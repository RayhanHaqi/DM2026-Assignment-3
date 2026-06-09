#!/usr/bin/env python
"""Cache TabPFN V3 grouped-OOF test probabilities for calibration sweeps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.prob_cache import save_v3_prob_cache
from model.sequence import load_test_sequences, load_train_sequences
from model.tabpfn_model import tabpfn_oof_predict
from model.temporal_features import combine_base_and_temporal_features
from model.utils import load_test_data, load_train_data
from model.validation import prediction_shift, smoke_slice_by_users

DEFAULT_CACHE = ROOT / "output" / "prob_cache" / "tabpfn_v3_91f1.npz"
DEFAULT_BASELINE = ROOT / "output" / "submission_tabpfn_v3_20260601_180045_01.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache TabPFN V3 test probabilities.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-path", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--baseline-csv", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-users", type=int, default=6)
    args = parser.parse_args()

    train_path = Path(args.data_dir) / "train" / "train"
    test_path = Path(args.data_dir) / "test" / "test"

    print("Loading data...")
    X_train, y_train, _, users = load_train_data(str(train_path))
    X_test, test_ids, _ = load_test_data(str(test_path))
    X_seq, _, _, _ = load_train_sequences(str(train_path))
    X_test_seq, _, _ = load_test_sequences(str(test_path))

    if args.smoke:
        n_users = max(args.smoke_users, 3)
        X_train, y_train, users, X_seq = smoke_slice_by_users(
            X_train, y_train, users, X_seq=X_seq, n_users=n_users
        )
        X_test = X_test.iloc[:40]
        test_ids = test_ids.iloc[:40]
        X_test_seq = X_test_seq[:40]
        print(f"  Smoke: {n_users} users, {len(X_test)} test rows")

    n_splits = 3 if args.smoke else 5
    X_train_t = combine_base_and_temporal_features(X_train, X_seq)
    X_test_t = combine_base_and_temporal_features(X_test, X_test_seq)

    print("Running TabPFN V3 grouped OOF (this may take several minutes)...")
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
    )

    if args.baseline_csv.exists() and not args.smoke:
        import pandas as pd

        baseline_preds = pd.read_csv(args.baseline_csv)["Label"].astype(int).to_numpy()
        repro = prediction_shift(result.test_preds, baseline_preds)
        print(
            f"  Repro vs {args.baseline_csv.name}: shift={repro.percent:.2f}% "
            f"OOF={result.oof_accuracy:.4f}"
        )

    path = save_v3_prob_cache(
        args.output_path,
        test_proba=result.test_proba,
        test_preds=result.test_preds,
        classes=result.classes,
        test_ids=test_ids.to_numpy(),
        oof_accuracy=result.oof_accuracy,
        oof_macro_f1=result.oof_macro_f1,
        fold_accuracies=np.asarray(result.fold_accuracies),
        oof_proba=result.oof_proba,
    )
    print(f"Wrote cache: {path}")
    print(f"  test_proba {result.test_proba.shape}  OOF acc={result.oof_accuracy:.4f}")


if __name__ == "__main__":
    main()
