#!/usr/bin/env python
"""TabPFN V3 probability ensemble — one CSV from multiple seeds."""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.sequence import load_test_sequences, load_train_sequences
from model.tabpfn_model import tabpfn_oof_predict, tabpfn_prob_ensemble_predict
from model.temporal_features import combine_base_and_temporal_features
from model.utils import generate_submission, load_test_data, load_train_data
from model.validation import (
    evaluate_paired_oof_gate,
    evaluate_submit_gates,
    prediction_shift,
    smoke_slice_by_users,
)

DEFAULT_BASELINE = ROOT / "output" / "submission_tabpfn_v3_20260601_180045_01.csv"
VALID_LABELS = {0, 1, 2, 3, 4, 5}


def _validate_frame(test_ids, preds):
    import pandas as pd

    frame = pd.DataFrame({"Id": test_ids, "Label": preds})
    assert len(frame) == 6849 or len(frame) <= 100
    assert set(frame["Label"].astype(int).tolist()) <= VALID_LABELS


def main():
    parser = argparse.ArgumentParser(description="TabPFN V3 probability ensemble runner.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--baseline-csv", default=str(DEFAULT_BASELINE))
    parser.add_argument("--seeds", default="0,11,23,42,71")
    parser.add_argument("--seed-search", default="0,11,20,23,42,65,71,80")
    parser.add_argument("--max-ensemble-seeds", type=int, default=5)
    parser.add_argument("--oof-band", type=float, default=0.003, help="Keep seeds within this OOF of best")
    parser.add_argument("--user-norm", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--min-shift-pct", type=float, default=2.0)
    parser.add_argument("--min-oof-margin", type=float, default=0.002)
    parser.add_argument("--max-class-prop-delta-pp", type=float, default=5.0)
    parser.add_argument("--skip-seed-search", action="store_true", help="Use --seeds only")
    args = parser.parse_args()

    train_path = Path(args.data_dir) / "train" / "train"
    test_path = Path(args.data_dir) / "test" / "test"

    print("Loading data...")
    X_train, y_train, _, users = load_train_data(str(train_path))
    X_test, test_ids, users_test = load_test_data(str(test_path))
    X_seq, _, _, _ = load_train_sequences(str(train_path))
    X_test_seq, _, _ = load_test_sequences(str(test_path))

    if args.smoke:
        X_train, y_train, users, X_seq = smoke_slice_by_users(
            X_train, y_train, users, X_seq=X_seq, n_users=4
        )
        n_test = 40
        X_test = X_test.iloc[:n_test]
        test_ids = test_ids.iloc[:n_test]
        users_test = users_test.iloc[:n_test]
        X_test_seq = X_test_seq[:n_test]

    X_train_t = combine_base_and_temporal_features(X_train, X_seq)
    X_test_t = combine_base_and_temporal_features(X_test, X_test_seq)

    knorm = {}
    if args.user_norm:
        knorm = dict(X_seq=X_seq, X_test_seq=X_test_seq, users_test=users_test, user_norm=True)

    clf_extra = {}
    if args.smoke:
        clf_extra["clf_kwargs"] = {"ignore_pretraining_limits": True}

    print("\n=== Baseline V3 (single seed=42, eval_metric=f1) ===")
    baseline = tabpfn_oof_predict(
        X_train_t,
        y_train,
        users,
        X_test_t,
        device=args.device,
        seed=42,
        n_estimators=16,
        eval_metric="f1",
        model_version="V3",
        n_splits=3 if args.smoke else 5,
        **knorm,
        **clf_extra,
    )
    baseline_preds = baseline.test_preds
    baseline_path = Path(args.baseline_csv)
    if baseline_path.exists() and not args.smoke:
        import pandas as pd

        baseline_preds = pd.read_csv(baseline_path)["Label"].astype(int).to_numpy()
    print(f"  Baseline OOF acc={baseline.oof_accuracy:.4f} worst-fold={min(baseline.fold_accuracies):.4f}")

    if args.skip_seed_search:
        selected = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    else:
        search_seeds = [int(s.strip()) for s in args.seed_search.split(",") if s.strip()]
        print(f"\n=== Seed search ({len(search_seeds)} seeds) ===")
        scored = []
        for seed in search_seeds:
            result = tabpfn_oof_predict(
                X_train_t,
                y_train,
                users,
                X_test_t,
                device=args.device,
                seed=seed,
                n_estimators=16,
                eval_metric="f1",
                model_version="V3",
                n_splits=3 if args.smoke else 5,
                **knorm,
                **clf_extra,
            )
            scored.append((seed, result.oof_accuracy))
            print(f"  seed={seed:3d}  OOF={result.oof_accuracy:.4f}")
        best_oof = max(o for _, o in scored)
        selected = [
            seed
            for seed, oof in sorted(scored, key=lambda x: (-x[1], x[0]))
            if oof >= best_oof - args.oof_band
        ][: args.max_ensemble_seeds]
    print(f"\n=== Ensemble seeds: {selected} ===")

    ensemble = tabpfn_prob_ensemble_predict(
        X_train_t,
        y_train,
        users,
        X_test_t,
        selected,
        device=args.device,
        n_estimators=16,
        eval_metric="f1",
        model_version="V3",
        n_splits=3 if args.smoke else 5,
        **knorm,
        **clf_extra,
    )

    paired = evaluate_paired_oof_gate(
        candidate_oof=ensemble.oof_accuracy,
        baseline_oof=baseline.oof_accuracy,
        candidate_fold_accs=ensemble.fold_accuracies,
        baseline_fold_accs=baseline.fold_accuracies,
        min_oof_margin=args.min_oof_margin,
    )
    shift = prediction_shift(ensemble.test_preds, baseline_preds)
    gate = evaluate_submit_gates(
        ensemble.test_preds,
        baseline_preds,
        candidate_oof=ensemble.oof_accuracy,
        baseline_oof=baseline.oof_accuracy,
        candidate_fold_accs=ensemble.fold_accuracies,
        baseline_fold_accs=baseline.fold_accuracies,
        class2_range=None,
        min_shift_pct=args.min_shift_pct,
        min_oof_margin=args.min_oof_margin,
        max_class_prop_delta_pp=args.max_class_prop_delta_pp,
    )
    print(
        f"\n  Ensemble OOF={ensemble.oof_accuracy:.4f} (delta {paired.oof_delta:+.4f}) "
        f"shift={shift.percent:.2f}% paired={'PASS' if paired.accepted else 'FAIL'} "
        f"submit={'PASS' if gate.accepted else 'FAIL'}"
    )
    if paired.reasons:
        print(f"  Paired: {'; '.join(paired.reasons)}")
    if gate.reasons:
        print(f"  Gates: {'; '.join(gate.reasons)}")

    if not gate.accepted and not args.force:
        print("\nNo submission written (use --force to override).")
        return

    _validate_frame(test_ids, ensemble.test_preds)
    tag = "tabpfn_v3_prob_ensemble_usernorm" if args.user_norm else "tabpfn_v3_prob_ensemble"
    notes = (
        f"seeds={selected}; n_est=16; eval_metric=f1; model_version=V3; "
        f"OOF={ensemble.oof_accuracy:.4f}; shift={shift.percent:.2f}%"
    )
    if args.user_norm:
        notes += "; +12 user-norm features"
    path = generate_submission(
        test_ids,
        ensemble.test_preds,
        Path(args.output_dir) / f"submission_{tag}.csv",
        model="TabPFN V3 prob ensemble",
        features="91 + user-norm" if args.user_norm else "91 targeted temporal",
        notes=notes,
    )
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
