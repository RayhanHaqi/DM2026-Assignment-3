#!/usr/bin/env python
"""TabPFN v2 — tuning_config, eval_metric variants, seed search, finetuning."""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.tabpfn_model import _require_tabpfn, fit_tabpfn_full, fit_tabpfn_oof
from model.temporal_features import combine_base_and_temporal_features
from model.train import _apply_smote
from model.utils import generate_submission, load_test_data, load_train_data
from model.sequence import load_test_sequences, load_train_sequences
from model.hjorth_features import build_hjorth_spectral_features


VALID_LABELS = {0, 1, 2, 3, 4, 5}
N_CLASSES = 6


def _validate_frame(test_ids, preds):
    frame = pd.DataFrame({"Id": test_ids, "Label": preds})
    assert len(frame) == 6849
    assert set(frame["Label"].astype(int).tolist()) <= VALID_LABELS
    return frame


def _write_sub(name, test_ids, preds, output_dir, model, features, notes):
    _validate_frame(test_ids, preds)
    return generate_submission(
        test_ids, preds,
        Path(output_dir) / f"submission_{name}.csv",
        model=model, features=features, notes=notes,
    )


def _fit_tabpfn_with_config(X_train, y_train, X_test, device, seed,
                             eval_metric="f1", n_estimators=16,
                             tune_thresholds=False, calibrate_temp=False):
    TabPFNClassifier = _require_tabpfn()
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    kwargs = dict(
        n_estimators=n_estimators,
        device=device,
        random_state=seed,
        eval_metric=eval_metric,
    )
    if tune_thresholds or calibrate_temp:
        kwargs["tuning_config"] = {
            "tune_decision_thresholds": tune_thresholds,
            "calibrate_temperature": calibrate_temp,
            "tuning_holdout_frac": 0.2,
        }

    clf = TabPFNClassifier(**kwargs)
    clf.fit(X_tr, y_train)
    proba = clf.predict_proba(X_te)
    preds = clf.classes_[np.argmax(proba, axis=1)]
    return preds, proba, clf.classes_


def _tabpfn_oof_eval(X_train, y_train, users, device, seed,
                     eval_metric="f1", n_estimators=16,
                     tune_thresholds=False, calibrate_temp=False,
                     n_splits=5):
    """Evaluate TabPFN with OOF, return OOF accuracy."""
    TabPFNClassifier = _require_tabpfn()
    y_arr = np.asarray(y_train)
    groups_arr = np.asarray(users)
    kf = GroupKFold(n_splits=n_splits)
    oof_preds = np.zeros(len(y_arr), dtype=int)

    for tr_idx, val_idx in kf.split(X_train, y_arr, groups_arr):
        X_tr = X_train.iloc[tr_idx] if hasattr(X_train, "iloc") else X_train[tr_idx]
        X_val = X_train.iloc[val_idx] if hasattr(X_train, "iloc") else X_train[val_idx]
        y_tr = y_arr[tr_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        kwargs = dict(
            n_estimators=n_estimators,
            device=device,
            random_state=seed,
            eval_metric=eval_metric,
        )
        if tune_thresholds or calibrate_temp:
            kwargs["tuning_config"] = {
                "tune_decision_thresholds": tune_thresholds,
                "calibrate_temperature": calibrate_temp,
                "tuning_holdout_frac": 0.2,
            }

        clf = TabPFNClassifier(**kwargs)
        clf.fit(X_tr_s, y_tr)
        preds = clf.predict(X_val_s)
        oof_preds[val_idx] = preds

    return float(accuracy_score(y_arr, oof_preds))


# ── Experiment 1: tuning_config + eval_metric grid ──────────────────────────

def experiment_tuning_config(X_train, y_train, X_test, test_ids, users, args):
    print("\n=== Experiment 1: tuning_config + eval_metric grid ===")

    configs = [
        # (name, eval_metric, tune_thresholds, calibrate_temp)
        ("tabpfn_tuned_f1",           "f1",                True,  True),
        ("tabpfn_tuned_acc",          "accuracy",          True,  True),
        ("tabpfn_tuned_bal",          "balanced_accuracy", True,  True),
        ("tabpfn_tuned_logloss",      "log_loss",          True,  True),
        ("tabpfn_tuned_roc",          "roc_auc",           True,  True),
        ("tabpfn_thresh_only_f1",     "f1",                True,  False),
        ("tabpfn_temp_only_f1",       "f1",                False, True),
        ("tabpfn_thresh_only_acc",    "accuracy",          True,  False),
    ]

    results = []
    for name, metric, thresh, temp in configs:
        print(f"  {name} (metric={metric}, thresh={thresh}, temp={temp})...")
        try:
            preds, _, _ = _fit_tabpfn_with_config(
                X_train, y_train, X_test, args.device, args.seed,
                eval_metric=metric, tune_thresholds=thresh, calibrate_temp=temp,
            )
            path = _write_sub(
                name, test_ids, preds, args.output_dir,
                model="TabPFN",
                features="42 base + targeted temporal",
                notes=f"eval_metric={metric}; tune_thresholds={thresh}; calibrate_temp={temp}",
            )
            dist = {i: int((preds == i).sum()) for i in range(6)}
            best_dist = pd.Series(
                pd.read_csv("output/submission_tabpfn_20260529_000430_01.csv")["Label"]
            ).value_counts().sort_index().to_dict()
            changed = sum(1 for a, b in zip(
                pd.read_csv("output/submission_tabpfn_20260529_000430_01.csv")["Label"],
                preds
            ) if a != b)
            print(f"    Dist: {dist}  ΔvsBest: {changed}")
            results.append((name, path, changed))
        except Exception as e:
            print(f"    FAILED: {e}")

    return results


# ── Experiment 2: Seed search via OOF ───────────────────────────────────────

def experiment_seed_search(X_train, y_train, X_test, test_ids, users, args):
    print("\n=== Experiment 2: Seed search (OOF accuracy, 20 seeds) ===")

    seeds = list(range(0, 200, 10))  # [0, 10, 20, ..., 190]
    best_oof_acc = -1.0
    best_seed = args.seed

    for seed in seeds:
        oof_acc = _tabpfn_oof_eval(
            X_train, y_train, users, args.device, seed,
            eval_metric="f1", n_estimators=16,
            tune_thresholds=True, calibrate_temp=True,
            n_splits=3 if args.smoke else 5,
        )
        if oof_acc > best_oof_acc:
            best_oof_acc = oof_acc
            best_seed = seed
        print(f"  seed={seed:3d}  OOF acc={oof_acc:.4f}  {'← BEST' if seed == best_seed else ''}")

    print(f"\n  Best seed: {best_seed} (OOF acc={best_oof_acc:.4f})")
    print(f"  Original seed 42 OOF acc: {_tabpfn_oof_eval(X_train, y_train, users, args.device, 42, eval_metric='f1', n_estimators=16, tune_thresholds=True, calibrate_temp=True, n_splits=3 if args.smoke else 5):.4f}")

    # Fit with best seed
    preds, _, _ = _fit_tabpfn_with_config(
        X_train, y_train, X_test, args.device, best_seed,
        eval_metric="f1", tune_thresholds=True, calibrate_temp=True,
    )
    path = _write_sub(
        "tabpfn_best_seed_tuned", test_ids, preds, args.output_dir,
        model="TabPFN",
        features="42 base + targeted temporal",
        notes=f"best_seed={best_seed} (OOF acc={best_oof_acc:.4f}); tuning_config enabled",
    )
    return best_seed, best_oof_acc, path


# ── Experiment 3: TabPFN on different feature sets ──────────────────────────

def experiment_feature_sets(X_base_train, y_train, X_base_test, test_ids,
                            X_seq, X_test_seq, users, args):
    print("\n=== Experiment 3: Different feature sets ===")

    X_targ_train = combine_base_and_temporal_features(X_base_train, X_seq)
    X_targ_test = combine_base_and_temporal_features(X_base_test, X_test_seq)

    feature_sets = [
        ("tabpfn_base42",       X_base_train, X_base_test, "42 base"),
        ("tabpfn_targeted91",   X_targ_train, X_targ_test, "42 base + targeted temporal"),
    ]

    # Also try with Hjorth features
    print("  Building Hjorth + spectral features...")
    hjorth_train = build_hjorth_spectral_features(X_seq)
    hjorth_test = build_hjorth_spectral_features(X_test_seq)
    X_full_train = pd.concat([X_targ_train.reset_index(drop=True), hjorth_train.reset_index(drop=True)], axis=1)
    X_full_test = pd.concat([X_targ_test.reset_index(drop=True), hjorth_test.reset_index(drop=True)], axis=1)
    feature_sets.append(("tabpfn_full127", X_full_train, X_full_test, "42 base + targeted temporal + Hjorth/spectral"))

    results = []
    for name, X_tr, X_te, features in feature_sets:
        print(f"  {name} ({X_tr.shape[1]} features)...")
        preds, _, _ = _fit_tabpfn_with_config(
            X_tr, y_train, X_te, args.device, args.seed,
            eval_metric="accuracy", tune_thresholds=True, calibrate_temp=True,
        )
        path = _write_sub(
            name, test_ids, preds, args.output_dir,
            model="TabPFN", features=features,
            notes=f"eval_metric=accuracy; tuning_config enabled; {X_tr.shape[1]} features",
        )
        dist = {i: int((preds == i).sum()) for i in range(6)}
        print(f"    Dist: {dist}")
        results.append((name, path))

    return results


# ── Experiment 4: TabPFN finetuning ─────────────────────────────────────────

def experiment_finetuning(X_train, y_train, X_test, test_ids, args):
    print("\n=== Experiment 4: TabPFN finetuning ===")
    try:
        import torch
        if not torch.cuda.is_available():
            print("  CUDA not available — skipping finetuning")
            return None
    except ImportError:
        print("  PyTorch not available — skipping finetuning")
        return None

    try:
        from tabpfn.finetuning.finetuned_classifier import FinetunedTabPFNClassifier
    except ImportError:
        print("  FinetunedTabPFNClassifier not available — skipping")
        return None

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    configs = [
        ("tabpfn_finetuned_30e", dict(epochs=30, learning_rate=2e-5, early_stopping_patience=8)),
        ("tabpfn_finetuned_50e", dict(epochs=50, learning_rate=1e-5, early_stopping_patience=12)),
    ]

    results = []
    for name, cfg in configs:
        print(f"  {name} (epochs={cfg['epochs']}, lr={cfg['learning_rate']})...")
        try:
            clf = FinetunedTabPFNClassifier(
                device="cuda",
                epochs=cfg["epochs"],
                learning_rate=cfg["learning_rate"],
                weight_decay=0.01,
                n_estimators_finetune=2,
                n_estimators_validation=2,
                n_estimators_final_inference=16,
                early_stopping=True,
                early_stopping_patience=cfg["early_stopping_patience"],
                eval_metric="log_loss",
                random_state=args.seed,
            )
            clf.fit(X_tr, np.asarray(y_train))
            preds = clf.predict(X_te)
            path = _write_sub(
                name, test_ids, preds, args.output_dir,
                model="TabPFN Finetuned",
                features="42 base + targeted temporal",
                notes=f"epochs={cfg['epochs']}; lr={cfg['learning_rate']}; early_stopping; eval_metric=log_loss",
            )
            dist = {i: int((preds == i).sum()) for i in range(6)}
            print(f"    Dist: {dist}")
            results.append((name, path))
        except Exception as e:
            print(f"    FAILED: {e}")

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--no-submit", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.no_submit = True

    train_path = Path(args.data_dir) / "train" / "train"
    test_path = Path(args.data_dir) / "test" / "test"

    print(f"Loading data from {train_path} and {test_path}...")
    X_train, y_train, _, users = load_train_data(str(train_path))
    X_test, test_ids, _ = load_test_data(str(test_path))
    X_seq, _, _, _ = load_train_sequences(str(train_path))
    X_test_seq, _, _ = load_test_sequences(str(test_path))

    if args.smoke:
        X_train = X_train.iloc[:60]
        y_train = y_train.iloc[:60]
        users = users.iloc[:60]
        X_test = X_test.iloc[:20]
        test_ids = test_ids.iloc[:20]
        X_seq = X_seq[:60]
        X_test_seq = X_test_seq[:20]

    X_train_t = combine_base_and_temporal_features(X_train, X_seq)
    X_test_t = combine_base_and_temporal_features(X_test, X_test_seq)
    print(f"Train: {X_train_t.shape}, Test: {X_test_t.shape}")

    # Experiment 1: tuning_config + eval_metric grid
    tuning_results = experiment_tuning_config(X_train_t, y_train, X_test_t, test_ids, users, args)

    # Experiment 2: Seed search
    best_seed, best_oof, _ = experiment_seed_search(X_train_t, y_train, X_test_t, test_ids, users, args)

    # Experiment 3: Feature sets
    feature_results = experiment_feature_sets(
        X_train, y_train, X_test, test_ids, X_seq, X_test_seq, users, args,
    )

    # Experiment 4: Finetuning
    ft_results = experiment_finetuning(X_train_t, y_train, X_test_t, test_ids, args)

    print("\n=== Summary ===")
    print(f"Best seed: {best_seed} (OOF acc={best_oof:.4f})")
    print(f"Tuning config variants: {len(tuning_results)} submitted")
    print(f"Feature set variants: {len(feature_results)} submitted")
    if ft_results:
        print(f"Finetuned variants: {len(ft_results)} submitted")


if __name__ == "__main__":
    main()
