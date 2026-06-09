#!/usr/bin/env python
"""
Revised TabPFN experiment runner (Codex-aligned plan).

- Grouped OOF with in-fold MI (no global feature leakage)
- Accuracy-first TabPFN V3 candidates
- Submission gates: OOF vs baseline, test label shift, class distribution
"""

import argparse
import hashlib
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.hjorth_features import build_hjorth_spectral_features
from model.sequence import load_test_sequences, load_train_sequences
from model.tabpfn_model import tabpfn_oof_predict
from model.temporal_features import combine_base_and_temporal_features
from model.utils import generate_submission, load_test_data, load_train_data
from model.validation import (
    evaluate_paired_oof_gate,
    evaluate_submit_gates,
    prediction_shift,
    smoke_slice_by_users,
)

VALID_LABELS = {0, 1, 2, 3, 4, 5}
DEFAULT_BASELINE = ROOT / "output" / "submission_tabpfn_v3_20260601_180045_01.csv"


def _validate_frame(test_ids, preds):
    frame = pd.DataFrame({"Id": test_ids, "Label": preds})
    assert len(frame) == 6849
    assert set(frame["Label"].astype(int).tolist()) <= VALID_LABELS


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


def _md5_preds(preds):
    digest = hashlib.md5()
    digest.update(np.asarray(preds, dtype=int).tobytes())
    return digest.hexdigest()[:8]


def _load_baseline_preds(baseline_path, test_ids):
    if baseline_path and Path(baseline_path).exists():
        frame = pd.read_csv(baseline_path)
        return frame["Label"].astype(int).to_numpy()
    return None


def _budget_full(written, args):
    """True when no more CSV writes are allowed this run."""
    if args.max_submissions <= 0:
        return False
    return len(written) >= args.max_submissions


def _run_candidate(
    name,
    result,
    *,
    written,
    baseline_oof,
    baseline_fold_accs,
    baseline_preds,
    test_ids,
    args,
    model,
    features,
    notes_extra,
):
    if _budget_full(written, args):
        print(f"  SKIP {name}: TabPFN submission budget full ({args.max_submissions})")
        return None
    shift = prediction_shift(result.test_preds, baseline_preds)
    paired = evaluate_paired_oof_gate(
        candidate_oof=result.oof_accuracy,
        baseline_oof=baseline_oof,
        candidate_fold_accs=result.fold_accuracies,
        baseline_fold_accs=baseline_fold_accs,
        min_oof_margin=args.min_oof_margin,
    )
    gate = evaluate_submit_gates(
        result.test_preds,
        baseline_preds,
        candidate_oof=result.oof_accuracy,
        baseline_oof=baseline_oof,
        candidate_fold_accs=result.fold_accuracies,
        baseline_fold_accs=baseline_fold_accs,
        class2_range=None,
        min_shift_pct=args.min_shift_pct,
        min_oof_margin=args.min_oof_margin,
        max_class_prop_delta_pp=args.max_class_prop_delta_pp,
    )
    md5 = _md5_preds(result.test_preds)
    gate_msg = "; ".join(gate.reasons) if gate.reasons else "passed submit gates"
    if paired.reasons:
        gate_msg = f"{gate_msg}; paired: {'; '.join(paired.reasons)}"
    submit_ok = gate.accepted and paired.accepted
    print(
        f"  {name}: OOF acc={result.oof_accuracy:.4f} f1={result.oof_macro_f1:.4f} "
        f"shift={shift.percent:.2f}% md5={md5} gate={'PASS' if submit_ok else 'SKIP'} ({gate_msg})"
    )
    if args.max_submissions <= 0 or args.smoke:
        return None
    if not submit_ok and not args.force:
        return None
    notes = (
        f"{notes_extra}; OOF acc={result.oof_accuracy:.4f}; OOF f1={result.oof_macro_f1:.4f}; "
        f"shift_vs_baseline={shift.percent:.2f}%; {gate_msg}"
    )
    path = _write_sub(name, test_ids, result.test_preds, args.output_dir, model, features, notes)
    written.append(path)
    print(f"    -> {path}  ({len(written)}/{args.max_submissions} slots used)")
    return path


def main():
    parser = argparse.ArgumentParser(description="Revised TabPFN runner with submission gates.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--baseline-csv", default=str(DEFAULT_BASELINE))
    parser.add_argument("--min-shift-pct", type=float, default=1.0)
    parser.add_argument("--min-oof-margin", type=float, default=-0.002)
    parser.add_argument("--max-class-prop-delta-pp", type=float, default=5.0)
    parser.add_argument("--force", action="store_true", help="Write CSV even when gates fail")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--smoke-users",
        type=int,
        default=6,
        help="Number of train users to keep in --smoke mode (need >= n_splits)",
    )
    parser.add_argument(
        "--max-submissions",
        type=int,
        default=2,
        help="Max Kaggle CSV files to write this run (default 2)",
    )
    parser.add_argument(
        "--only",
        default="cache_acc,v3_acc,seed_acc,v3_n32",
        help="Comma list: cache_acc,v3_acc,seed_acc,v3_n32,mi127,baseline,user_norm",
    )
    args = parser.parse_args()

    train_path = Path(args.data_dir) / "train" / "train"
    test_path = Path(args.data_dir) / "test" / "test"

    print("Loading data...")
    X_train, y_train, _, users = load_train_data(str(train_path))
    X_test, test_ids, users_test = load_test_data(str(test_path))
    X_seq, _, _, _ = load_train_sequences(str(train_path))
    X_test_seq, _, _ = load_test_sequences(str(test_path))

    if args.smoke:
        n_users = max(args.smoke_users, 3)
        X_train, y_train, users, X_seq = smoke_slice_by_users(
            X_train, y_train, users, X_seq=X_seq, n_users=n_users
        )
        n_test = 40
        X_test = X_test.iloc[:n_test]
        test_ids = test_ids.iloc[:n_test]
        users_test = users_test.iloc[:n_test]
        X_test_seq = X_test_seq[:n_test]
        print(f"  Smoke: {n_users} train users, {len(X_train)} train rows, {n_test} test rows")

    n_splits = 3 if args.smoke else 5

    X_train_t = combine_base_and_temporal_features(X_train, X_seq)
    X_test_t = combine_base_and_temporal_features(X_test, X_test_seq)

    hjorth_train = build_hjorth_spectral_features(X_seq)
    hjorth_test = build_hjorth_spectral_features(X_test_seq)
    X_full_train = pd.concat(
        [X_train_t.reset_index(drop=True), hjorth_train.reset_index(drop=True)], axis=1
    )
    X_full_test = pd.concat(
        [X_test_t.reset_index(drop=True), hjorth_test.reset_index(drop=True)], axis=1
    )

    only = {part.strip() for part in args.only.split(",")}
    if "all" in only:
        only = {"cache_acc", "v3_acc", "seed_acc", "v3_n32"}

    print(f"\n=== Kaggle budget: TabPFN max {args.max_submissions} submissions this run ===")
    print("  (GBDT blend: use run_gbdt_cache_blend.py — max 1 submission)")

    tuning_config = {
        "tune_decision_thresholds": True,
        "calibrate_temperature": True,
        "tuning_holdout_frac": 0.2,
    }

    # ── Baseline V3 repro ───────────────────────────────────────────────────
    print("\n=== Baseline: TabPFN V3 (grouped OOF, f1) ===")
    baseline_result = tabpfn_oof_predict(
        X_train_t,
        y_train,
        users,
        X_test_t,
        device=args.device,
        seed=42,
        n_estimators=16,
        eval_metric="f1",
        model_version="V3",
        n_splits=n_splits,
    )
    baseline_preds = None if args.smoke else _load_baseline_preds(args.baseline_csv, test_ids)
    if baseline_preds is None:
        baseline_preds = baseline_result.test_preds
        print("  (no baseline CSV — using fresh V3 OOF-averaged test preds)")
    else:
        repro_shift = prediction_shift(baseline_result.test_preds, baseline_preds)
        print(
            f"  Repro vs {args.baseline_csv}: shift={repro_shift.percent:.2f}% "
            f"(OOF acc={baseline_result.oof_accuracy:.4f})"
        )

    from model.validation import prediction_distribution

    baseline_dist = prediction_distribution(baseline_preds)
    baseline_oof = baseline_result.oof_accuracy
    baseline_fold_accs = baseline_result.fold_accuracies
    print(f"  Baseline OOF accuracy: {baseline_oof:.4f}")
    print(f"  Baseline worst-fold: {min(baseline_fold_accs):.4f}")
    print(f"  Baseline test distribution: {baseline_dist}")

    written = []

    if "baseline" in only and not _budget_full(written, args):
        _run_candidate(
            "tabpfn_v3_baseline_repro",
            baseline_result,
            written=written,
            baseline_oof=baseline_oof,
            baseline_preds=baseline_preds,
            baseline_fold_accs=baseline_fold_accs,
            test_ids=test_ids,
            args=args,
            model="TabPFN V3",
            features="42 base + targeted temporal",
            notes_extra="model_version=V3; n_est=16; eval_metric=accuracy; grouped OOF",
        )

    # ── fit_with_cache + accuracy (Phase 2) ───────────────────────────────
    if "cache_acc" in only and not _budget_full(written, args):
        print("\n=== Candidate: V3 + fit_mode=fit_with_cache + accuracy ===")
        result = tabpfn_oof_predict(
            X_train_t,
            y_train,
            users,
            X_test_t,
            device=args.device,
            seed=42,
            n_estimators=16,
            eval_metric="accuracy",
            model_version="V3",
            fit_mode="fit_with_cache",
            n_splits=n_splits,
        )
        _run_candidate(
            "tabpfn_v3_cache_acc",
            result,
            written=written,
            baseline_oof=baseline_oof,
            baseline_preds=baseline_preds,
            baseline_fold_accs=baseline_fold_accs,
            test_ids=test_ids,
            args=args,
            model="TabPFN V3",
            features="42 base + targeted temporal",
            notes_extra="fit_mode=fit_with_cache; eval_metric=accuracy",
        )

    # ── In-fold MI top-127 ────────────────────────────────────────────────
    if "mi127" in only and not _budget_full(written, args):
        print("\n=== Candidate: extended features, in-fold MI top-127 ===")
        result = tabpfn_oof_predict(
            X_full_train,
            y_train,
            users,
            X_full_test,
            device=args.device,
            seed=42,
            n_estimators=16,
            eval_metric="accuracy",
            model_version="V3",
            mi_top_k=127,
            n_splits=n_splits,
        )
        _run_candidate(
            "tabpfn_full_mi127_infold",
            result,
            written=written,
            baseline_oof=baseline_oof,
            baseline_preds=baseline_preds,
            baseline_fold_accs=baseline_fold_accs,
            test_ids=test_ids,
            args=args,
            model="TabPFN V3",
            features="42+targeted+Hjorth/spectral, in-fold MI top-127",
            notes_extra="in-fold mutual_info; 127 features",
        )

    # ── Accuracy + tuning_config ──────────────────────────────────────────
    if "v3_acc" in only and not _budget_full(written, args):
        print("\n=== Candidate: V3 + accuracy + tuning_config ===")
        result = tabpfn_oof_predict(
            X_train_t,
            y_train,
            users,
            X_test_t,
            device=args.device,
            seed=42,
            n_estimators=16,
            eval_metric="accuracy",
            model_version="V3",
            tuning_config=tuning_config,
            n_splits=n_splits,
        )
        _run_candidate(
            "tabpfn_v3_acc_tuned",
            result,
            written=written,
            baseline_oof=baseline_oof,
            baseline_preds=baseline_preds,
            baseline_fold_accs=baseline_fold_accs,
            test_ids=test_ids,
            args=args,
            model="TabPFN V3",
            features="42 base + targeted temporal",
            notes_extra="eval_metric=accuracy; tuning_config thresholds+temp",
        )

    # ── Seed search (accuracy) ──────────────────────────────────────────────
    if "seed_acc" in only and not _budget_full(written, args):
        print("\n=== Candidate: seed search (accuracy, V3) ===")
        seeds = [42, 65, 80, 20, 0, 11, 23, 71]
        best_result = None
        best_seed = None
        best_oof = -1.0
        for seed in seeds:
            result = tabpfn_oof_predict(
                X_train_t,
                y_train,
                users,
                X_test_t,
                device=args.device,
                seed=seed,
                n_estimators=16,
                eval_metric="accuracy",
                model_version="V3",
                n_splits=n_splits,
            )
            print(f"    seed={seed:3d}  OOF acc={result.oof_accuracy:.4f}")
            if result.oof_accuracy > best_oof:
                best_oof = result.oof_accuracy
                best_seed = seed
                best_result = result
        print(f"  Best seed: {best_seed} (OOF acc={best_oof:.4f})")
        _run_candidate(
            f"tabpfn_v3_seed{best_seed}_acc",
            best_result,
            written=written,
            baseline_oof=baseline_oof,
            baseline_preds=baseline_preds,
            baseline_fold_accs=baseline_fold_accs,
            test_ids=test_ids,
            args=args,
            model="TabPFN V3",
            features="42 base + targeted temporal",
            notes_extra=f"best_seed={best_seed}; eval_metric=accuracy",
        )

    # ── n_estimators=32 ───────────────────────────────────────────────────
    if "user_norm" in only and not _budget_full(written, args):
        print("\n=== Candidate: V3 + user-norm features (in-fold) ===")
        result = tabpfn_oof_predict(
            X_train_t,
            y_train,
            users,
            X_test_t,
            device=args.device,
            seed=42,
            n_estimators=16,
            eval_metric="f1",
            model_version="V3",
            n_splits=n_splits,
            X_seq=X_seq,
            X_test_seq=X_test_seq,
            users_test=users_test,
            user_norm=True,
        )
        _run_candidate(
            "tabpfn_v3_user_norm",
            result,
            written=written,
            baseline_oof=baseline_oof,
            baseline_preds=baseline_preds,
            baseline_fold_accs=baseline_fold_accs,
            test_ids=test_ids,
            args=args,
            model="TabPFN V3",
            features="91 + 12 user-norm",
            notes_extra="in-fold user z-score aggregates; eval_metric=f1",
        )

    if "v3_n32" in only and not _budget_full(written, args):
        print("\n=== Candidate: V3 n_estimators=32 ===")
        result = tabpfn_oof_predict(
            X_train_t,
            y_train,
            users,
            X_test_t,
            device=args.device,
            seed=42,
            n_estimators=32,
            eval_metric="accuracy",
            model_version="V3",
            n_splits=n_splits,
        )
        _run_candidate(
            "tabpfn_v3_n32_acc",
            result,
            written=written,
            baseline_oof=baseline_oof,
            baseline_preds=baseline_preds,
            baseline_fold_accs=baseline_fold_accs,
            test_ids=test_ids,
            args=args,
            model="TabPFN V3",
            features="42 base + targeted temporal",
            notes_extra="n_estimators=32; eval_metric=accuracy",
        )

    print("\n=== Done ===")
    if written:
        print("Written submissions:")
        for path in written:
            print(f"  {path}")
        print("\nCompare before Kaggle upload:")
        print(
            f"  python scripts/compare_submissions.py --baseline {args.baseline_csv} "
            + " ".join(str(p) for p in written)
        )
    else:
        print("No submissions passed gates (use --force to write anyway).")


if __name__ == "__main__":
    main()
