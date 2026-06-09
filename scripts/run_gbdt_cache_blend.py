#!/usr/bin/env python
"""
Cached OOF probability blend: XGBoost + LightGBM + CatBoost (1 Kaggle slot).

Tunes each model lightly (accuracy, grouped CV, cv SMOTE), caches OOF/test proba,
searches blend weights on OOF accuracy only, writes one gated submission.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.catboost_train import _require_catboost, tune_catboost
from model.oof import evaluate_oof_model, search_weighted_ensemble
from model.sequence import load_test_sequences, load_train_sequences
from model.temporal_features import combine_base_and_temporal_features
from model.train import _apply_smote, tune_lightgbm, tune_xgboost
from model.utils import generate_submission, load_test_data, load_train_data
from model.validation import prediction_distribution, prediction_shift, should_write_tabpfn_submission

DEFAULT_TABPFN_BASELINE = ROOT / "output" / "submission_tabpfn_v3_20260601_180045_01.csv"
DEFAULT_XGB_BASELINE = ROOT / "output" / "submission_xgb_targeted_temporal_20260521_143424_01.csv"
VALID_LABELS = {0, 1, 2, 3, 4, 5}


def _validate_frame(test_ids, preds):
    frame = pd.DataFrame({"Id": test_ids, "Label": preds})
    assert len(frame) == 6849
    assert set(frame["Label"].astype(int).tolist()) <= VALID_LABELS


def _load_labels(path):
    return pd.read_csv(path)["Label"].astype(int).to_numpy()


def _proba_sanity(name, proba, classes):
    if proba.shape[1] != len(classes):
        raise ValueError(f"{name}: proba columns {proba.shape[1]} != {len(classes)} classes")
    row_sums = proba.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-3):
        raise ValueError(f"{name}: probability rows do not sum to 1 (min={row_sums.min():.4f})")


def _diversity_report(y_true, oof_results):
    """Print solo OOF metrics and pairwise disagreement."""
    y_arr = np.asarray(y_true)
    print("\n=== Solo model OOF (grouped CV, accuracy) ===")
    for res in oof_results:
        print(
            f"  {res.name}: acc={res.accuracy:.4f} std={res.accuracy_std:.4f} "
            f"f1_macro={res.macro_f1:.4f}"
        )

    print("\n=== Pairwise test prediction disagreement (OOF argmax) ===")
    names = [r.name for r in oof_results]
    preds_list = [np.argmax(r.oof_proba, axis=1) for r in oof_results]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            disagree = float((preds_list[i] != preds_list[j]).mean() * 100.0)
            print(f"  {names[i]} vs {names[j]}: {disagree:.2f}% OOF rows differ")

    solo_best = max(oof_results, key=lambda r: r.accuracy)
    print(f"\n  Best solo OOF: {solo_best.name} ({solo_best.accuracy:.4f})")


def _cache_path(cache_dir, smoke):
    name = "gbdt_oof_smoke.npz" if smoke else "gbdt_oof.npz"
    return Path(cache_dir) / name


def _save_cache(path, oof_results, meta):
    arrays = {"classes": oof_results[0].classes}
    for res in oof_results:
        arrays[f"{res.name}_oof_proba"] = res.oof_proba
        arrays[f"{res.name}_test_proba"] = res.test_proba
    np.savez_compressed(path, **arrays)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  Cached -> {path}")


def main():
    parser = argparse.ArgumentParser(description="GBDT cached probability blend (max 1 Kaggle file).")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--cache-dir", default="output/cache")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--n-trials", type=int, default=25, help="Optuna trials per model (light tune)")
    parser.add_argument("--weight-step", type=float, default=0.1, help="Grid step for 3-model weights")
    parser.add_argument("--max-submissions", type=int, default=1)
    parser.add_argument("--tabpfn-baseline", default=str(DEFAULT_TABPFN_BASELINE))
    parser.add_argument("--xgb-baseline", default=str(DEFAULT_XGB_BASELINE))
    parser.add_argument("--min-shift-pct", type=float, default=1.0)
    parser.add_argument("--min-oof-margin", type=float, default=0.0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--cache-only", action="store_true", help="Load cache; skip model training")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    train_path = Path(args.data_dir) / "train" / "train"
    test_path = Path(args.data_dir) / "test" / "test"
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_path(cache_dir, args.smoke)

    print("Loading data...")
    X_train, y_train, _, users = load_train_data(str(train_path))
    X_test, test_ids, _ = load_test_data(str(test_path))
    X_seq, _, _, _ = load_train_sequences(str(train_path))
    X_test_seq, _, _ = load_test_sequences(str(test_path))

    if args.smoke:
        X_train = X_train.iloc[:80]
        y_train = y_train.iloc[:80]
        users = users.iloc[:80]
        X_test = X_test.iloc[:40]
        test_ids = test_ids.iloc[:40]
        X_seq = X_seq[:80]
        X_test_seq = X_test_seq[:40]

    X_train_t = combine_base_and_temporal_features(X_train, X_seq)
    X_test_t = combine_base_and_temporal_features(X_test, X_test_seq)
    y_arr = np.asarray(y_train)
    n_splits = 3 if args.smoke else 5
    features_note = "42 base + targeted temporal"

    oof_results = None

    if args.cache_only and cache_file.exists():
        print(f"Loading cache {cache_file}...")
        data = np.load(cache_file, allow_pickle=True)
        classes = data["classes"]
        oof_results = []
        for model_name in ("xgb", "lgb", "cat"):
            oof_proba = data[f"{model_name}_oof_proba"]
            test_proba = data[f"{model_name}_test_proba"]
            preds = classes[np.argmax(oof_proba, axis=1)]
            oof_results.append(
                type("CachedOOF", (), {
                    "name": model_name,
                    "accuracy": float(accuracy_score(y_arr, preds)),
                    "accuracy_std": 0.0,
                    "macro_f1": float(f1_score(y_arr, preds, average="macro", zero_division=0)),
                    "oof_proba": oof_proba,
                    "test_proba": test_proba,
                    "classes": classes,
                })()
            )
    else:
        print(f"\n=== Light tune + grouped OOF ({args.n_trials} trials/model, metric=accuracy) ===")

        print("  XGBoost...")
        xgb_params, _ = tune_xgboost(
            X_train_t, y_train, users,
            n_trials=args.n_trials, metric="accuracy", use_smote=True, n_jobs=args.n_jobs,
        )
        xgb_oof = evaluate_oof_model(
            XGBClassifier(**xgb_params), X_train_t, y_train, users, X_test_t,
            n_splits=n_splits, use_smote=True, name="xgb",
        )

        print("  LightGBM...")
        lgb_params, _ = tune_lightgbm(
            X_train_t, y_train, users,
            n_trials=args.n_trials, metric="accuracy", use_smote=True, n_jobs=args.n_jobs,
        )
        lgb_oof = evaluate_oof_model(
            LGBMClassifier(**lgb_params), X_train_t, y_train, users, X_test_t,
            n_splits=n_splits, use_smote=True, name="lgb",
        )

        print("  CatBoost...")
        CatBoostClassifier = _require_catboost()
        cat_params, _ = tune_catboost(
            X_train_t, y_train, users,
            n_trials=args.n_trials, metric="accuracy", use_smote=True, n_jobs=args.n_jobs,
        )
        # tune_catboost returns final_params with loss_function/eval_metric already set
        cat_template = CatBoostClassifier(**cat_params)
        cat_oof = evaluate_oof_model(
            cat_template, X_train_t, y_train, users, X_test_t,
            n_splits=n_splits, use_smote=True, name="cat",
        )

        oof_results = [xgb_oof, lgb_oof, cat_oof]
        for res in oof_results:
            _proba_sanity(res.name, res.oof_proba, res.classes)
            _proba_sanity(res.name, res.test_proba, res.classes)

        meta = {
            "n_trials": args.n_trials,
            "n_splits": n_splits,
            "xgb_params": {k: v for k, v in xgb_params.items() if k != "n_jobs"},
            "lgb_params": {k: v for k, v in lgb_params.items() if k not in ("n_jobs", "verbose")},
            "cat_params": cat_params,
        }
        _save_cache(cache_file, oof_results, meta)

    _diversity_report(y_arr, oof_results)

    print(f"\n=== Weight search (step={args.weight_step}, maximize OOF accuracy) ===")
    oof_probas = [r.oof_proba for r in oof_results]
    best_weights, best_oof_acc, _ = search_weighted_ensemble(oof_probas, y_arr, step=args.weight_step)
    names = [r.name for r in oof_results]
    print(f"  Best weights: {dict(zip(names, best_weights.round(3)))}")
    print(f"  Blended OOF accuracy: {best_oof_acc:.4f}")

    solo_xgb = next(r for r in oof_results if r.name == "xgb")
    print(f"  Solo XGB OOF accuracy: {solo_xgb.accuracy:.4f} (delta blend-xgb: {best_oof_acc - solo_xgb.accuracy:+.4f})")

    # Final test blend from cached fold-averaged test proba (same weights as OOF selection)
    test_blend = sum(w * r.test_proba for w, r in zip(best_weights, oof_results))
    classes = oof_results[0].classes
    test_preds = classes[np.argmax(test_blend, axis=1)].astype(int)

    # Optional: refit full train without SMOTE and reblend (Codex: keep OOF weights only)
  # Using fold-averaged test proba is consistent with revised TabPFN OOF protocol

    tabpfn_base = Path(args.tabpfn_baseline)
    xgb_base = Path(args.xgb_baseline)
    ref_preds = _load_labels(tabpfn_base) if tabpfn_base.exists() else test_preds
    ref_dist = prediction_distribution(ref_preds)

    shift = prediction_shift(test_preds, ref_preds)
    cand_dist = prediction_distribution(test_preds)

    gate = should_write_tabpfn_submission(
        best_oof_acc,
        solo_xgb.accuracy,
        shift,
        cand_dist,
        ref_dist,
        len(test_preds),
        min_shift_pct=args.min_shift_pct,
        min_oof_margin=args.min_oof_margin,
    )

    print(f"\n=== Submission gate (vs TabPFN V3: {tabpfn_base.name}) ===")
    print(f"  shift={shift.percent:.2f}%  gate={'PASS' if gate.accepted else 'SKIP'} ({gate.reason})")

    if xgb_base.exists():
        xgb_preds = _load_labels(xgb_base)
        shift_xgb = prediction_shift(test_preds, xgb_preds)
        print(f"  vs XGB solo CSV: shift={shift_xgb.percent:.2f}%")

    written = 0
    if gate.accepted or args.force:
        if written < args.max_submissions:
            w_str = "/".join(f"{w:.2f}" for w in best_weights)
            notes = (
                f"cached OOF blend xgb/lgb/cat weights=[{w_str}]; "
                f"OOF acc={best_oof_acc:.4f}; solo_xgb OOF={solo_xgb.accuracy:.4f}; "
                f"n_trials={args.n_trials}; cv_smote=True; {gate.reason}"
            )
            _validate_frame(test_ids, test_preds)
            path = generate_submission(
                test_ids,
                test_preds,
                Path(args.output_dir) / "submission_gbdt_cache_blend.csv",
                model="XGB+LGB+CatBoost cache blend",
                features=features_note,
                notes=notes,
            )
            written += 1
            print(f"\n  Wrote ({written}/{args.max_submissions}): {path}")
            print(
                f"\n  Compare:\n  python scripts/compare_submissions.py "
                f"--baseline {tabpfn_base} {path}"
            )
    else:
        print("\n  No submission written (gates failed). Use --force to override.")

    print("\n=== Kaggle budget reminder ===")
    print("  TabPFN workflow: max 2 slots (run_tabpfn_revised.py)")
    print("  GBDT blend:      max 1 slot (this script)")


if __name__ == "__main__":
    main()
