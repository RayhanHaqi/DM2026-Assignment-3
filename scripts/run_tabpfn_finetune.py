#!/usr/bin/env python
"""TabPFN finetune — single bounded protocol with V3-aligned gates before any CSV write."""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.submission_audit import gate_submission
from model.sequence import load_test_sequences, load_train_sequences
from model.tabpfn_model import (
    finetuned_grouped_oof,
    make_finetuned_classifier,
    tabpfn_oof_predict,
)
from model.temporal_features import combine_base_and_temporal_features
from model.utils import generate_submission, load_test_data, load_train_data
from model.validation import (
    effective_group_kfold_splits,
    evaluate_submit_gates,
    prediction_distribution,
    required_oof_accuracy,
    smoke_slice_by_users,
)

VALID_LABELS = {0, 1, 2, 3, 4, 5}
DEFAULT_BASELINE = ROOT / "output" / "submission_tabpfn_v3_20260601_180045_01.csv"
DEFAULT_TRACKER = ROOT / "output" / "SUBMISSIONS.md"
EXPECTED_TEST_ROWS = 6849

# Finetune early-stopping uses TabPFN API metrics (log_loss | roc_auc); Kaggle gates use OOF accuracy.
FINETUNE_CONFIG = {
    "name": "tabpfn_ft_acc_30e",
    "epochs": 30,
    "lr": 2e-5,
    "patience": 8,
    "finetune_eval_metric": "log_loss",
    "n_est_finetune": 2,
    "n_est_validation": 2,
    "n_est_final": 16,
    "random_state": 42,
}


def _validate_frame(test_ids, preds, expected_len=EXPECTED_TEST_ROWS):
    frame = pd.DataFrame({"Id": test_ids, "Label": preds})
    if len(frame) != expected_len:
        raise ValueError(f"expected {expected_len} test rows, got {len(frame)}")
    bad = set(frame["Label"].astype(int).tolist()) - VALID_LABELS
    if bad:
        raise ValueError(f"invalid labels: {bad}")


def _write_sub(name, test_ids, preds, output_dir, model, features, notes):
    _validate_frame(test_ids, preds)
    return generate_submission(
        test_ids,
        preds,
        Path(output_dir) / f"submission_{name}.csv",
        model=model,
        features=features,
        notes=notes,
    )


def _audit_written_submission(
    candidate_path: Path,
    baseline_path: Path,
    tracker_path: Path,
    output_dir: Path,
    min_shift_pct: float,
    best_public_score: float,
) -> bool:
    row = gate_submission(
        candidate_path,
        baseline_path,
        tracker_path,
        output_dir,
        best_public_score=best_public_score,
        min_shift_pct=min_shift_pct,
        max_shift_pct=10.0,
    )
    if row.block_submit:
        print("  Phase 0 audit BLOCK:")
        for reason in row.block_reasons:
            print(f"    - {reason}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="TabPFN finetune (log_loss early-stop) with accuracy OOF gates vs V3."
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--baseline-csv", default=str(DEFAULT_BASELINE))
    parser.add_argument("--tracker", default=str(DEFAULT_TRACKER))
    parser.add_argument("--best-score", type=float, default=0.7830)
    parser.add_argument("--min-oof-margin", type=float, default=0.002)
    parser.add_argument("--min-shift-pct", type=float, default=2.0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Write CSV when OOF/shift gates fail (Phase 0 audit still enforced)",
    )
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        print("CUDA required for finetuning. Exiting.")
        sys.exit(1)

    baseline_path = Path(args.baseline_csv)
    if not baseline_path.exists():
        raise SystemExit(f"Baseline not found: {baseline_path}")

    train_path = Path(args.data_dir) / "train" / "train"
    test_path = Path(args.data_dir) / "test" / "test"

    print(f"Loading data from {train_path} and {test_path}...")
    X_train, y_train, _, users = load_train_data(str(train_path))
    X_test, test_ids, _ = load_test_data(str(test_path))
    X_seq, _, _, _ = load_train_sequences(str(train_path))
    X_test_seq, _, _ = load_test_sequences(str(test_path))

    if args.smoke:
        X_train, y_train, users, X_seq = smoke_slice_by_users(
            X_train, y_train, users, X_seq, n_users=6
        )
        n_splits = effective_group_kfold_splits(users, 3)
        print(
            f"SMOKE: {len(users)} rows, {users.nunique()} users, "
            f"n_splits={n_splits}; no submission file will be written"
        )
    else:
        n_splits = effective_group_kfold_splits(users, 5)

    X_train_t = combine_base_and_temporal_features(X_train, X_seq)
    X_test_t = combine_base_and_temporal_features(X_test, X_test_seq)
    print(f"Train: {X_train_t.shape}, Test: {X_test_t.shape}")

    baseline_test_preds = load_labels(baseline_path)
    tracker_path = Path(args.tracker)
    output_dir = Path(args.output_dir)
    class2_range = resolve_class2_range(
        sorted(output_dir.glob("submission_*.csv")),
        tracker_path,
        baseline_test_preds,
    )

    print("\n=== Baseline: TabPFN V3 grouped OOF (accuracy, canonical tabpfn_oof_predict) ===")
    baseline_result = tabpfn_oof_predict(
        X_train_t,
        y_train,
        users,
        X_test_t,
        device=args.device,
        seed=42,
        n_estimators=16,
        eval_metric="accuracy",
        model_version="V3",
        n_splits=n_splits,
    )
    baseline_oof = baseline_result.oof_accuracy
    baseline_fold_accs = baseline_result.fold_accuracies
    required_oof = required_oof_accuracy(baseline_oof, args.min_oof_margin)
    print(f"  V3 OOF accuracy: {baseline_oof:.4f}")
    print(f"  V3 worst-fold: {min(baseline_fold_accs):.4f}")
    print(f"  Required OOF for proceed/submit: {required_oof:.4f}")

    cfg = FINETUNE_CONFIG
    print(f"\n=== Finetuning OOF: {cfg['name']} (early-stop={cfg['finetune_eval_metric']}) ===")
    ft_result = finetuned_grouped_oof(
        X_train_t, y_train, users, cfg, args.device, n_splits=n_splits
    )
    for fold_i, acc in enumerate(ft_result.fold_accuracies, start=1):
        print(f"    FT fold {fold_i}: acc={acc:.4f}")
    print(f"  Finetune OOF accuracy: {ft_result.oof_accuracy:.4f}")
    print(f"  Finetune OOF macro-F1: {ft_result.oof_macro_f1:.4f}")
    print(
        f"  Finetune worst-fold: {min(ft_result.fold_accuracies):.4f} "
        f"vs V3 worst-fold {min(baseline_fold_accs):.4f}"
    )

    if ft_result.oof_accuracy < required_oof and not args.force:
        print(f"\nFinetune OOF below {required_oof:.4f}. No test predict / no CSV.")
        return

    print("\n=== Full-data finetune fit + test predict ===")
    y_arr = np.asarray(y_train)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train_t)
    X_te = scaler.transform(X_test_t)

    clf = make_finetuned_classifier(cfg, args.device)
    clf.fit(X_tr, y_arr)
    test_preds = clf.predict(X_te)

    gate = evaluate_submit_gates(
        test_preds,
        baseline_test_preds,
        candidate_oof=ft_result.oof_accuracy,
        baseline_oof=baseline_oof,
        candidate_fold_accs=ft_result.fold_accuracies,
        baseline_fold_accs=baseline_fold_accs,
        class2_range=class2_range,
        min_oof_margin=args.min_oof_margin,
        min_shift_pct=args.min_shift_pct,
    )

    print("\n=== Submit-ready checklist ===")
    if gate.accepted:
        print("  PASS — all pre-submit gates")
    else:
        print("  FAIL — do not submit:")
        for reason in gate.reasons:
            print(f"    - {reason}")

    if args.smoke:
        print("\nSMOKE complete (no file written).")
        return

    if not gate.accepted and not args.force:
        print("\nNo submission written (gates failed).")
        return

    notes = (
        f"epochs={cfg['epochs']}; lr={cfg['lr']}; finetune_metric={cfg['finetune_eval_metric']}; "
        f"OOF acc={ft_result.oof_accuracy:.4f}; V3 baseline OOF={baseline_oof:.4f}; "
        f"{'forced write (OOF/shift gates)' if args.force and not gate.accepted else 'passed submit gates'}"
    )
    path = _write_sub(
        cfg["name"],
        test_ids,
        test_preds,
        args.output_dir,
        model="TabPFN Finetuned",
        features="42 base + targeted temporal",
        notes=notes,
    )
    print(f"\nWrote: {path}")

    if not _audit_written_submission(
        path, baseline_path, tracker_path, output_dir, args.min_shift_pct, args.best_score
    ):
        path.unlink(missing_ok=True)
        print("Removed file — Phase 0 audit blocked (not overridable by --force).")
        return

    print("Phase 0 audit: no block flags.")
    print("\nCompare before upload:")
    print(
        f"  python scripts/compare_submissions.py --baseline {baseline_path} "
        f"--min-shift-pct {args.min_shift_pct} {path}"
    )
    print(f"  OOF delta vs V3: {ft_result.oof_accuracy - baseline_oof:+.4f}")


if __name__ == "__main__":
    main()
