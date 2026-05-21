import argparse
from pathlib import Path
import sys

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.oof import evaluate_oof_model, search_weighted_ensemble
from model.rich_temporal_features import combine_base_and_rich_features
from model.sequence import load_test_sequences, load_train_sequences
from model.utils import generate_submission, load_test_data, load_train_data
from scripts.run_balanced_candidates import (
    _fit_tree_model,
    _limit_by_user,
    _split_path,
    validate_submission_frame,
)


def rich_candidate_names(include_pseudolabel=False):
    names = ["xgb_rich_temporal_selected", "extratrees_rich_temporal", "oof_weighted_tabular_ensemble"]
    if include_pseudolabel:
        names[-1] = "xgb_rich_temporal_pseudolabel"
    return names


def parse_group_list(text):
    return [part.strip() for part in text.split() if part.strip()]


def _write_submission(name, test_ids, preds, args, model, features, notes):
    validate_submission_frame(test_ids, preds, expected_rows=len(test_ids))
    if args.no_submit:
        return None
    return generate_submission(test_ids, preds, Path(args.output_dir) / f"submission_{name}.csv", model=model, features=features, notes=notes)


def _xgb_params(seed):
    return {
        "n_estimators": 700,
        "max_depth": 5,
        "learning_rate": 0.03,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
        "random_state": seed,
        "n_jobs": -1,
    }


def _fit_xgb_with_optional_pseudolabel(X, y, X_test, test_proba, threshold, seed):
    confidence = test_proba.max(axis=1)
    pseudo_mask = confidence >= threshold
    if pseudo_mask.sum() >= 500:
        X_fit = np.vstack([np.asarray(X), X_test[pseudo_mask]])
        y_fit = np.concatenate([np.asarray(y), test_proba[pseudo_mask].argmax(axis=1)])
    else:
        X_fit = np.asarray(X)
        y_fit = np.asarray(y)
    return _fit_tree_model(XGBClassifier, _xgb_params(seed), X_fit, y_fit, X_test, use_smote=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run rich tabular ensemble candidates.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--selected-groups", default="segments trend diff magnitude fft")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-user-limit", type=int, default=None)
    parser.add_argument("--pseudo-threshold", type=float, default=0.90)
    parser.add_argument("--include-pseudolabel", action="store_true")
    parser.add_argument("--no-submit", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        args.per_user_limit = 2
        args.n_splits = 3
        args.no_submit = True

    groups = parse_group_list(args.selected_groups)
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

    X = combine_base_and_rich_features(X_base, X_seq, groups=groups)
    X_test = combine_base_and_rich_features(X_test_base, X_test_seq, groups=groups)
    xgb = XGBClassifier(**_xgb_params(args.seed))
    extra = ExtraTreesClassifier(n_estimators=600, max_features="sqrt", class_weight="balanced", random_state=args.seed, n_jobs=-1)
    xgb_result = evaluate_oof_model(xgb, X, y, users, X_test, n_splits=args.n_splits, use_smote=True, name="xgb_rich_temporal_selected")
    extra_result = evaluate_oof_model(extra, X, y, users, X_test, n_splits=args.n_splits, use_smote=False, name="extratrees_rich_temporal")
    weights, ensemble_acc, ensemble_oof = search_weighted_ensemble([xgb_result.oof_proba, extra_result.oof_proba], y, step=0.1)
    ensemble_test = weights[0] * xgb_result.test_proba + weights[1] * extra_result.test_proba

    xgb_preds = _fit_tree_model(XGBClassifier, _xgb_params(args.seed), X, y, X_test, use_smote=True)
    extra_preds = _fit_tree_model(ExtraTreesClassifier, {"n_estimators": 600, "max_features": "sqrt", "class_weight": "balanced", "random_state": args.seed, "n_jobs": -1}, X, y, X_test, use_smote=False)
    ensemble_preds = ensemble_test.argmax(axis=1)

    print(f"xgb_rich_temporal_selected acc={xgb_result.accuracy:.4f} f1={xgb_result.macro_f1:.4f}")
    print(f"extratrees_rich_temporal acc={extra_result.accuracy:.4f} f1={extra_result.macro_f1:.4f}")
    print(f"oof_weighted_tabular_ensemble acc={ensemble_acc:.4f} weights={weights.tolist()}")

    _write_submission("xgb_rich_temporal_selected", test_ids, xgb_preds, args, "XGBoost", "42 base + rich temporal", f"groups={groups}")
    _write_submission("extratrees_rich_temporal", test_ids, extra_preds, args, "ExtraTrees", "42 base + rich temporal", f"groups={groups}")
    if args.include_pseudolabel:
        pseudo_preds = _fit_xgb_with_optional_pseudolabel(np.asarray(X), y, np.asarray(X_test), ensemble_test, args.pseudo_threshold, args.seed)
        _write_submission("xgb_rich_temporal_pseudolabel", test_ids, pseudo_preds, args, "XGBoost", "42 base + rich temporal + pseudo-label", f"threshold={args.pseudo_threshold}")
    else:
        _write_submission("oof_weighted_tabular_ensemble", test_ids, ensemble_preds, args, "OOF weighted ensemble", "42 base + rich temporal", f"weights={weights.tolist()}")


if __name__ == "__main__":
    main()
