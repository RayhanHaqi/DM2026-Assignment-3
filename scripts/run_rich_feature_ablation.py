import argparse
from pathlib import Path
import sys

import pandas as pd
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.oof import evaluate_oof_model
from model.sequence import load_test_sequences, load_train_sequences
from model.temporal_features import combine_base_and_temporal_features
from model.rich_temporal_features import combine_base_and_rich_features
from model.utils import load_test_data, load_train_data
from scripts.run_balanced_candidates import _limit_by_user, _split_path


def ablation_feature_sets():
    return [
        ("base_42", None),
        ("targeted_temporal", "targeted"),
        ("segments", ["segments"]),
        ("segments_trend", ["segments", "trend"]),
        ("magnitude_diff", ["magnitude", "diff"]),
        ("fft", ["fft"]),
        ("rolling_autocorr", ["rolling", "autocorr"]),
        ("all_rich", ["segments", "trend", "diff", "magnitude", "fft", "rolling", "autocorr"]),
    ]


def _build_features(name, groups, X_base, X_seq):
    if groups is None:
        return X_base
    if groups == "targeted":
        return combine_base_and_temporal_features(X_base, X_seq)
    return combine_base_and_rich_features(X_base, X_seq, groups=groups)


def parse_args():
    parser = argparse.ArgumentParser(description="Run rich temporal feature ablations.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-user-limit", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        args.per_user_limit = 2
        args.n_splits = 3

    train_path = _split_path(args.data_dir, "train")
    test_path = _split_path(args.data_dir, "test")
    X_base, y, train_ids, users = load_train_data(train_path)
    X_test_base, test_ids, test_users = load_test_data(test_path)
    X_seq, y_seq, seq_ids, seq_users = load_train_sequences(train_path)
    X_test_seq, seq_test_ids, seq_test_users = load_test_sequences(test_path)
    X_base, y, train_ids, users = _limit_by_user(X_base, train_ids, users, y, args.per_user_limit)
    X_test_base, test_ids, test_users = _limit_by_user(X_test_base, test_ids, test_users, per_user_limit=args.per_user_limit)
    X_seq, y_seq, seq_ids, seq_users = _limit_by_user(X_seq, seq_ids, seq_users, y_seq, args.per_user_limit)
    X_test_seq, seq_test_ids, seq_test_users = _limit_by_user(X_test_seq, seq_test_ids, seq_test_users, per_user_limit=args.per_user_limit)

    rows = []
    for name, groups in ablation_feature_sets():
        X = _build_features(name, groups, X_base, X_seq)
        X_test = _build_features(name, groups, X_test_base, X_test_seq)
        model = XGBClassifier(n_estimators=120, max_depth=5, learning_rate=0.05, random_state=args.seed, n_jobs=-1)
        result = evaluate_oof_model(model, X, y, users, X_test, n_splits=args.n_splits, use_smote=True, name=name)
        rows.append({
            "name": name,
            "features": X.shape[1],
            "accuracy": result.accuracy,
            "accuracy_std": result.accuracy_std,
            "worst_accuracy": result.worst_accuracy,
            "macro_f1": result.macro_f1,
        })

    summary = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
