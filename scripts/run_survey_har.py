#!/usr/bin/env python
"""Survey HAR slice: shallow DeepConvLSTM + TabPFN with spectral sequence features."""

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.lstm import (
    fit_deepconv_lstm_full,
    predict_deepconv_lstm,
    predict_deepconv_lstm_proba,
    train_deepconv_lstm_candidate,
)
from model.sequence import load_test_sequences, load_train_sequences
from model.sequence_oof import evaluate_sequence_oof_model
from model.tabpfn_model import fit_tabpfn_full, fit_tabpfn_oof, predict_tabpfn
from model.temporal_features import combine_base_temporal_spectral_features
from model.utils import generate_submission, load_test_data, load_train_data

VALID_LABELS = {0, 1, 2, 3, 4, 5}
MULTIROCKET_OOF_BASELINE = 0.74


def _smoke_subset_by_users(X_train, y_train, users, X_test, test_ids, X_seq, X_test_seq, n_users=4):
    """Keep full rows for the first n_users train users (group-safe smoke slice)."""
    users_arr = np.asarray(users)
    picked = sorted(np.unique(users_arr))[:n_users]
    mask = np.isin(users_arr, picked)
    idx = np.flatnonzero(mask)
    n_test = max(8, len(test_ids) // 200)
    return (
        X_train.iloc[idx].reset_index(drop=True),
        y_train.iloc[idx].reset_index(drop=True),
        users.iloc[idx].reset_index(drop=True),
        X_test.iloc[:n_test].reset_index(drop=True),
        test_ids.iloc[:n_test].reset_index(drop=True),
        X_seq[idx],
        X_test_seq[:n_test],
        len(idx),
        n_test,
        len(picked),
    )


def _validate_preds(test_ids, preds):
    assert len(test_ids) == len(preds)
    assert set(np.asarray(preds, dtype=int).tolist()) <= VALID_LABELS


def _write_submission(name, test_ids, preds, output_dir, no_submit, model, features, notes):
    _validate_preds(test_ids, preds)
    path = generate_submission(
        test_ids,
        preds,
        Path(output_dir) / f"submission_{name}.csv",
        model=model,
        features=features,
        notes=notes,
    )
    if no_submit:
        print(f"  (no-submit) wrote {path}")
    return path


def _print_class_diagnostics(y_true, preds, title):
    cm = confusion_matrix(y_true, preds, labels=sorted(VALID_LABELS))
    print(f"\n{title} confusion matrix (rows=true, cols=pred):")
    print(cm)
    for label in sorted(VALID_LABELS):
        mask = np.asarray(y_true) == label
        if mask.any():
            recall = float((np.asarray(preds)[mask] == label).mean())
            print(f"  class {label} recall: {recall:.3f} ({mask.sum()} rows)")


def run_deepconv_lstm(X_seq, y_train, users, X_test_seq, test_ids, args):
    print("\n=== Shallow DeepConvLSTM (grouped OOF) ===")
    epochs = 1 if args.smoke else args.epochs
    patience = 1 if args.smoke else args.patience
    n_splits = 2 if args.smoke else args.n_splits

    result = train_deepconv_lstm_candidate(
        X_seq,
        y_train,
        users,
        epochs=epochs,
        batch_size=args.batch_size,
        patience=patience,
        device=args.device,
        seed=args.seed,
        normalize=True,
    )
    print(
        f"  Holdout (grouped 80/20): acc={result.accuracy:.4f} "
        f"f1={result.f1_macro:.4f} best_epoch={result.best_epoch}"
    )

    def fit_fn(X_fold, y_fold):
        return fit_deepconv_lstm_full(
            X_fold,
            y_fold,
            epochs=max(1, result.best_epoch),
            batch_size=args.batch_size,
            device=args.device,
            seed=args.seed,
            normalize=True,
        )

    def predict_proba_fn(model, X_fold):
        return predict_deepconv_lstm_proba(
            model,
            X_fold,
            device=args.device,
            normalize=True,
        )

    oof_result = evaluate_sequence_oof_model(
        fit_fn,
        predict_proba_fn,
        np.asarray(X_seq),
        np.asarray(y_train),
        np.asarray(users),
        np.asarray(X_test_seq),
        n_splits=n_splits,
        random_state=args.seed,
        name="deepconv_lstm_shallow",
    )
    print(
        f"  Grouped OOF: acc={oof_result.accuracy:.4f} "
        f"std={oof_result.accuracy_std:.4f} worst={oof_result.worst_accuracy:.4f} "
        f"macro_f1={oof_result.macro_f1:.4f}"
    )
    print(f"  Fold accuracies: {[round(x, 4) for x in oof_result.fold_accuracy]}")
    oof_preds = oof_result.classes[np.argmax(oof_result.oof_proba, axis=1)]
    _print_class_diagnostics(y_train, oof_preds, "DeepConvLSTM OOF")

    full_model = fit_fn(np.asarray(X_seq), np.asarray(y_train))
    preds = predict_deepconv_lstm(
        full_model,
        np.asarray(X_test_seq),
        device=args.device,
        normalize=True,
    )
    submit = (not args.no_submit) and oof_result.accuracy > MULTIROCKET_OOF_BASELINE
    path = None
    if submit or args.force_submit:
        path = _write_submission(
            "deepconv_lstm_shallow",
            test_ids,
            preds,
            args.output_dir,
            args.no_submit,
            model="DeepConvLSTM shallow",
            features="raw 300x6 sequence",
            notes=(
                f"epochs={max(1, result.best_epoch)}; grouped_oof={oof_result.accuracy:.4f}; "
                f"fold_std={oof_result.accuracy_std:.4f}; macro_f1={oof_result.macro_f1:.4f}"
            ),
        )
    else:
        print(
            f"  Skipping submission (OOF {oof_result.accuracy:.4f} <= "
            f"MultiRocket public baseline {MULTIROCKET_OOF_BASELINE:.2f})"
        )

    return {
        "name": "deepconv_lstm_shallow",
        "oof_accuracy": oof_result.accuracy,
        "oof_std": oof_result.accuracy_std,
        "macro_f1": oof_result.macro_f1,
        "file": path,
    }


def run_tabpfn_spectral(X_train, y_train, users, X_test, test_ids, X_seq, X_test_seq, args):
    print("\n=== TabPFN + base + temporal + spectral FFT features ===")
    X_train_full = combine_base_temporal_spectral_features(X_train, X_seq)
    X_test_full = combine_base_temporal_spectral_features(X_test, X_test_seq)
    print(f"  Feature matrix: train {X_train_full.shape}, test {X_test_full.shape}")

    n_splits = 2 if args.smoke else args.n_splits
    try:
        oof_proba, oof_preds = fit_tabpfn_oof(
            X_train_full,
            np.asarray(y_train),
            np.asarray(users),
            n_splits=n_splits,
            n_estimators=args.tabpfn_estimators,
            device=args.device,
            random_state=args.seed,
        )
    except ImportError as exc:
        print(f"  TabPFN unavailable: {exc}")
        return {"name": "tabpfn_spectral", "oof_accuracy": 0.0, "file": None}

    oof_acc = float(accuracy_score(np.asarray(y_train), oof_preds))
    oof_f1 = float(f1_score(np.asarray(y_train), oof_preds, average="macro"))
    print(f"  TabPFN OOF: acc={oof_acc:.4f} macro_f1={oof_f1:.4f}")
    _print_class_diagnostics(y_train, oof_preds, "TabPFN spectral OOF")

    if args.smoke:
        print("  Smoke: OOF complete; skipping TabPFN full fit and CSV (use full run with --device cuda)")
        return {"name": "tabpfn_spectral_fft", "oof_accuracy": oof_acc, "macro_f1": oof_f1, "file": None}

    model, scaler = fit_tabpfn_full(
        X_train_full,
        np.asarray(y_train),
        n_estimators=args.tabpfn_estimators,
        device=args.device,
        random_state=args.seed,
    )
    preds = predict_tabpfn((model, scaler), X_test_full)
    path = _write_submission(
        "tabpfn_spectral_fft",
        test_ids,
        preds,
        args.output_dir,
        args.no_submit,
        model="TabPFN",
        features="42 base + targeted temporal + spectral FFT",
        notes=f"n_estimators={args.tabpfn_estimators}; oof_acc={oof_acc:.4f}; oof_f1={oof_f1:.4f}",
    )
    return {"name": "tabpfn_spectral_fft", "oof_accuracy": oof_acc, "macro_f1": oof_f1, "file": path}


def main():
    parser = argparse.ArgumentParser(description="Survey HAR: DeepConvLSTM + TabPFN spectral features.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--tabpfn-estimators", type=int, default=16)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-submit", action="store_true", help="Write CSVs but mark as dry-run in logs")
    parser.add_argument("--force-submit", action="store_true", help="Write DeepConv CSV even if OOF gate fails")
    parser.add_argument("--skip-deepconv", action="store_true")
    parser.add_argument("--skip-tabpfn", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.device = "cpu"
        args.batch_size = min(args.batch_size, 32)

    train_path = Path(args.data_dir) / "train" / "train"
    test_path = Path(args.data_dir) / "test" / "test"
    print(f"Loading data from {train_path} and {test_path}...")
    X_train, y_train, _, users = load_train_data(str(train_path))
    X_test, test_ids, _ = load_test_data(str(test_path))
    X_seq, _, _, _ = load_train_sequences(str(train_path))
    X_test_seq, _, _ = load_test_sequences(str(test_path))

    if args.smoke:
        (
            X_train,
            y_train,
            users,
            X_test,
            test_ids,
            X_seq,
            X_test_seq,
            n_train,
            n_test,
            n_users,
        ) = _smoke_subset_by_users(X_train, y_train, users, X_test, test_ids, X_seq, X_test_seq)
        print(f"Smoke subset: train={n_train} rows from {n_users} users, test={n_test}")

    results = []
    if not args.skip_deepconv:
        results.append(
            run_deepconv_lstm(X_seq, y_train, users, X_test_seq, test_ids, args)
        )
    if not args.skip_tabpfn:
        results.append(
            run_tabpfn_spectral(X_train, y_train, users, X_test, test_ids, X_seq, X_test_seq, args)
        )

    print("\n=== Summary ===")
    for item in results:
        file_note = item.get("file") or "(no file)"
        print(
            f"  {item['name']}: oof={item.get('oof_accuracy', 0):.4f} "
            f"file={file_note}"
        )
    print("\nDone.")


if __name__ == "__main__":
    main()
