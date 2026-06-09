#!/usr/bin/env python
"""Fast TabPFN + XGB blend (regenerate after gbdt_blend bug fix)."""
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.tabpfn_model import fit_tabpfn_full, predict_tabpfn_proba
from model.train import tune_xgboost, _apply_smote
from model.temporal_features import combine_base_and_temporal_features
from model.utils import generate_submission, load_train_data, load_test_data
from model.sequence import load_train_sequences, load_test_sequences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=2)
    args = parser.parse_args()

    train_path = Path(args.data_dir) / "train" / "train"
    test_path = Path(args.data_dir) / "test" / "test"

    print("Loading data...")
    X_train, y_train, _, users = load_train_data(str(train_path))
    X_test, test_ids, _ = load_test_data(str(test_path))
    X_seq, _, _, _ = load_train_sequences(str(train_path))
    X_test_seq, _, _ = load_test_sequences(str(test_path))

    X_train_t = combine_base_and_temporal_features(X_train, X_seq)
    X_test_t = combine_base_and_temporal_features(X_test, X_test_seq)
    y_arr = np.asarray(y_train)
    classes = np.arange(6)

    # TabPFN seed ensemble
    print("TabPFN seed ensemble (5 seeds)...")
    seeds = [0, 11, 23, 42, 71]
    tabpfn_proba = np.zeros((len(X_test_t), 6), dtype=float)
    for seed in seeds:
        model, scaler = fit_tabpfn_full(
            X_train_t, y_arr, n_estimators=16, device=args.device, random_state=seed,
        )
        proba, _ = predict_tabpfn_proba((model, scaler), X_test_t)
        tabpfn_proba += proba / len(seeds)
    print(f"  TabPFN done. Shape: {tabpfn_proba.shape}")

    # XGB with fixed params (quick tune, 20 trials)
    print("Tuning XGB (20 trials)...")
    xgb_params, _ = tune_xgboost(
        X_train_t, y_arr, users, n_trials=20, metric="f1_macro",
        use_smote=True, n_jobs=args.n_jobs,
    )
    scaler_xgb = StandardScaler()
    X_tr_s = scaler_xgb.fit_transform(X_train_t)
    X_te_s = scaler_xgb.transform(X_test_t)
    X_tr_s, y_fit = _apply_smote(X_tr_s, y_arr)
    xgb = XGBClassifier(**{**xgb_params, "random_state": args.seed})
    xgb.fit(X_tr_s, y_fit)
    xgb_proba = xgb.predict_proba(X_te_s)
    print(f"  XGB done. Shape: {xgb_proba.shape}")

    # Generate blends at multiple ratios
    blends = [
        ("tabpfn_xgb_blend_90_10", 0.90, 0.10),
        ("tabpfn_xgb_blend_85_15", 0.85, 0.15),
        ("tabpfn_xgb_blend_80_20", 0.80, 0.20),
        ("tabpfn_xgb_blend_95_05", 0.95, 0.05),
    ]

    for name, w_t, w_x in blends:
        blend = w_t * tabpfn_proba + w_x * xgb_proba
        preds = classes[np.argmax(blend, axis=1)]
        path = generate_submission(
            test_ids, preds,
            Path(args.output_dir) / f"submission_{name}.csv",
            model=f"TabPFN {w_t:.2f} XGB {w_x:.2f} blend",
            features="42 base + targeted temporal",
            notes=f"TabPFN seed ensemble ({seeds}) + XGB (20 trials f1_macro); blend ratio {w_t:.2f}/{w_x:.2f}",
        )
        print(f"  {name}: {pd.Series(preds).value_counts().sort_index().to_dict()}")

    print("\nDone. Submit the best-scoring blend.")


if __name__ == "__main__":
    main()
