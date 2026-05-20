import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.cnn import fit_cnn_full, predict_cnn, train_cnn_candidate
from model.sequence import load_test_sequences, load_train_sequences
from model.train import _apply_smote, cv_evaluate, tune_lightgbm, tune_xgboost
from model.utils import generate_submission, load_test_data, load_train_data


VALID_LABELS = {0, 1, 2, 3, 4, 5}


def daily_tree_candidate_names():
    return ["lgb_macro_smote_refresh", "xgb_macro_smote_refresh"]


def validate_submission_frame(file_ids, preds, expected_rows=6849):
    frame = pd.DataFrame({"Id": file_ids, "Label": preds})
    if len(frame) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, got {len(frame)}")
    if frame.isna().any().any():
        raise ValueError("Submission contains null values")
    labels = set(frame["Label"].astype(int).tolist())
    if not labels <= VALID_LABELS:
        raise ValueError(f"Submission contains invalid labels: {sorted(labels - VALID_LABELS)}")
    return frame


def _split_path(data_dir, split):
    path = Path(data_dir) / split
    nested = path / split
    if nested.exists():
        return nested
    return path


def _limit_by_user(X, ids, users, y=None, per_user_limit=None):
    if per_user_limit is None:
        if y is None:
            return X, ids, users
        return X, y, ids, users

    keep = users.groupby(users).cumcount() < per_user_limit
    if hasattr(X, "iloc"):
        X_limited = X.loc[keep].reset_index(drop=True)
    else:
        X_limited = X[np.asarray(keep)]
    ids_limited = ids.loc[keep].reset_index(drop=True)
    users_limited = users.loc[keep].reset_index(drop=True)
    if y is None:
        return X_limited, ids_limited, users_limited
    return X_limited, y.loc[keep].reset_index(drop=True), ids_limited, users_limited


def _fit_tree_model(model_cls, params, X_train, y_train, X_test, use_smote):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    y_fit = y_train
    if use_smote:
        X_train_scaled, y_fit = _apply_smote(X_train_scaled, y_train)
    model = model_cls(**params)
    model.fit(X_train_scaled, y_fit)
    return model.predict(X_test_scaled)


def _write_submission(name, test_ids, preds, output_dir, no_submit, model, features, notes):
    validate_submission_frame(test_ids, preds, expected_rows=len(test_ids))
    if no_submit:
        return None
    return generate_submission(
        test_ids,
        preds,
        Path(output_dir) / f"submission_{name}.csv",
        model=model,
        features=features,
        notes=notes,
    )


def _run_lgb_candidate(name, X_train, y_train, users, X_test, test_ids, args, use_smote, metric="accuracy"):
    print(f"\nTuning {name}...")
    params, _ = tune_lightgbm(
        X_train,
        y_train,
        users,
        n_trials=args.tree_trials,
        metric=metric,
        use_smote=use_smote,
    )
    scores, acc, acc_std = cv_evaluate(
        LGBMClassifier(**params),
        X_train,
        y_train,
        users,
        metric="accuracy",
        use_smote=use_smote,
    )
    _, f1, _ = cv_evaluate(
        LGBMClassifier(**params),
        X_train,
        y_train,
        users,
        metric="f1_macro",
        use_smote=use_smote,
    )
    preds = _fit_tree_model(LGBMClassifier, params, X_train, y_train, X_test, use_smote=use_smote)
    path = _write_submission(
        name,
        test_ids,
        preds,
        args.output_dir,
        args.no_submit,
        model="LightGBM",
        features="42 base",
        notes=f"{metric}-tuned; {args.tree_trials} trials; use_smote={use_smote}",
    )
    return {"name": name, "accuracy": acc, "accuracy_std": acc_std, "f1_macro": f1, "file": path, "scores": scores}


def _run_xgb_candidate(name, X_train, y_train, users, X_test, test_ids, args, use_smote=False, metric="accuracy"):
    print(f"\nTuning {name}...")
    params, _ = tune_xgboost(
        X_train,
        y_train,
        users,
        n_trials=args.tree_trials,
        metric=metric,
        use_smote=use_smote,
    )
    scores, acc, acc_std = cv_evaluate(
        XGBClassifier(**params),
        X_train,
        y_train,
        users,
        metric="accuracy",
        use_smote=use_smote,
    )
    _, f1, _ = cv_evaluate(
        XGBClassifier(**params),
        X_train,
        y_train,
        users,
        metric="f1_macro",
        use_smote=use_smote,
    )
    preds = _fit_tree_model(XGBClassifier, params, X_train, y_train, X_test, use_smote=use_smote)
    path = _write_submission(
        name,
        test_ids,
        preds,
        args.output_dir,
        args.no_submit,
        model="XGBoost",
        features="42 base",
        notes=f"{metric}-tuned; {args.tree_trials} trials; use_smote={use_smote}",
    )
    return {"name": name, "accuracy": acc, "accuracy_std": acc_std, "f1_macro": f1, "file": path, "scores": scores}


def _run_daily_tree_candidates(X_train, y_train, users, X_test, test_ids, args):
    lgb_name, xgb_name = daily_tree_candidate_names()
    return [
        _run_lgb_candidate(lgb_name, X_train, y_train, users, X_test, test_ids, args, use_smote=True, metric="f1_macro"),
        _run_xgb_candidate(xgb_name, X_train, y_train, users, X_test, test_ids, args, use_smote=True, metric="f1_macro"),
    ]


def _run_cnn_candidate(X_seq, y, users, X_test_seq, test_ids, args, name="cnn_raw_sequence", variant="small", normalize=False):
    print(f"\nTraining {name}...")
    result = train_cnn_candidate(
        X_seq,
        y,
        users,
        epochs=args.cnn_epochs,
        batch_size=args.cnn_batch_size,
        patience=args.cnn_patience,
        device=args.device,
        seed=args.seed,
        variant=variant,
        normalize=normalize,
    )
    full_epochs = max(1, result.best_epoch)
    model = fit_cnn_full(
        X_seq,
        y,
        epochs=full_epochs,
        batch_size=args.cnn_batch_size,
        device=args.device,
        seed=args.seed,
        variant=variant,
        normalize=normalize,
    )
    preds = predict_cnn(model, X_test_seq, device=args.device, normalize=normalize)
    path = _write_submission(
        name,
        test_ids,
        preds,
        args.output_dir,
        args.no_submit,
        model="1D CNN",
        features="raw 300x6 sequence",
        notes=f"variant={variant}; normalize={normalize}; epochs={full_epochs}; validation from grouped split",
    )
    return {
        "name": name,
        "accuracy": result.accuracy,
        "accuracy_std": 0.0,
        "f1_macro": result.f1_macro,
        "file": path,
        "scores": [result.accuracy],
    }


def _print_summary(results):
    ranked = sorted(results, key=lambda row: row["accuracy"], reverse=True)
    print("\nCandidate summary:")
    for i, row in enumerate(ranked, start=1):
        file_text = row["file"] if row["file"] is not None else "not written (--no-submit)"
        print(
            f"{i}. {row['name']:<20} "
            f"acc={row['accuracy']:.4f} (+/- {row['accuracy_std']:.4f}) "
            f"f1={row['f1_macro']:.4f} file={file_text}"
        )
    print("\nRecommended order:")
    for row in ranked[:3]:
        print(f"- {row['name']}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run balanced ASG3 candidates.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--tree-trials", type=int, default=30)
    parser.add_argument("--cnn-epochs", type=int, default=30)
    parser.add_argument("--cnn-batch-size", type=int, default=128)
    parser.add_argument("--cnn-patience", type=int, default=5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-user-limit", type=int, default=None)
    parser.add_argument("--include-xgb", action="store_true")
    parser.add_argument(
        "--daily-20260520",
        action="store_true",
        help="Run today's approved candidates: LGB macro/SMOTE, XGB macro/SMOTE, improved CNN.",
    )
    parser.add_argument("--no-submit", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        args.tree_trials = 1
        args.cnn_epochs = 1
        args.cnn_patience = 1
        args.per_user_limit = 2
        args.no_submit = True

    train_path = _split_path(args.data_dir, "train")
    test_path = _split_path(args.data_dir, "test")

    print(f"Loading aggregate data from {train_path} and {test_path}...")
    X_train, y_train, train_ids, users = load_train_data(train_path)
    X_test, test_ids, test_users = load_test_data(test_path)
    X_train, y_train, train_ids, users = _limit_by_user(X_train, train_ids, users, y_train, args.per_user_limit)
    X_test, test_ids, test_users = _limit_by_user(X_test, test_ids, test_users, per_user_limit=args.per_user_limit)

    if args.daily_20260520:
        results = _run_daily_tree_candidates(X_train, y_train, users, X_test, test_ids, args)
    else:
        results = [
            _run_lgb_candidate("lgb_acc_no_smote", X_train, y_train, users, X_test, test_ids, args, use_smote=False),
            _run_lgb_candidate("lgb_acc_smote", X_train, y_train, users, X_test, test_ids, args, use_smote=True),
        ]

        if args.include_xgb:
            results.append(
                _run_xgb_candidate(
                    "xgb_acc_no_smote",
                    X_train,
                    y_train,
                    users,
                    X_test,
                    test_ids,
                    args,
                    use_smote=False,
                    metric="accuracy",
                )
            )

    print(f"\nLoading sequence data from {train_path} and {test_path}...")
    X_seq, y_seq, seq_ids, seq_users = load_train_sequences(train_path)
    X_test_seq, seq_test_ids, seq_test_users = load_test_sequences(test_path)
    X_seq, y_seq, seq_ids, seq_users = _limit_by_user(X_seq, seq_ids, seq_users, y_seq, args.per_user_limit)
    X_test_seq, seq_test_ids, seq_test_users = _limit_by_user(
        X_test_seq,
        seq_test_ids,
        seq_test_users,
        per_user_limit=args.per_user_limit,
    )
    cnn_name = "cnn_improved_sequence" if args.daily_20260520 else "cnn_raw_sequence"
    cnn_variant = "improved" if args.daily_20260520 else "small"
    cnn_normalize = bool(args.daily_20260520)
    results.append(
        _run_cnn_candidate(
            X_seq,
            y_seq.to_numpy(),
            seq_users.to_numpy(),
            X_test_seq,
            seq_test_ids,
            args,
            name=cnn_name,
            variant=cnn_variant,
            normalize=cnn_normalize,
        )
    )

    _print_summary(results)


if __name__ == "__main__":
    main()
