#!/usr/bin/env python
"""TabPFN Aggressive — Optuna tuning, OOF predictions, feature selection, model versions."""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.tabpfn_model import _require_tabpfn, tabpfn_oof_predict
from model.temporal_features import combine_base_and_temporal_features
from model.utils import generate_submission, load_test_data, load_train_data
from model.sequence import load_test_sequences, load_train_sequences
from model.hjorth_features import build_hjorth_spectral_features

VALID_LABELS = {0, 1, 2, 3, 4, 5}
N_CLASSES = 6


def _validate_frame(test_ids, preds):
    frame = pd.DataFrame({"Id": test_ids, "Label": preds})
    assert len(frame) == 6849
    assert set(frame["Label"].astype(int).tolist()) <= VALID_LABELS


def _write_sub(name, test_ids, preds, output_dir, model, features, notes):
    _validate_frame(test_ids, preds)
    return generate_submission(
        test_ids, preds,
        Path(output_dir) / f"submission_{name}.csv",
        model=model, features=features, notes=notes,
    )


def _tabpfn_oof_predict(X_train, y_train, users, X_test, device, seed,
                        n_estimators=16, eval_metric="f1", n_splits=5,
                        scaler_cls=StandardScaler, model_version=None, **clf_kwargs):
    """Generate OOF test predictions (averaged across folds)."""
    TabPFNClassifier = _require_tabpfn()
    y_arr = np.asarray(y_train)
    groups_arr = np.asarray(users)
    kf = GroupKFold(n_splits=n_splits)
    oof_preds = np.zeros(len(y_arr), dtype=int)
    test_proba = np.zeros((len(X_test), N_CLASSES), dtype=float)

    for tr_idx, val_idx in kf.split(X_train, y_arr, groups_arr):
        X_tr = X_train.iloc[tr_idx]
        X_val = X_train.iloc[val_idx]
        y_tr = y_arr[tr_idx]

        scaler = scaler_cls()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)
        X_te_s = scaler.transform(X_test)

        kwargs = dict(n_estimators=n_estimators, device=device, random_state=seed,
                      eval_metric=eval_metric, **clf_kwargs)
        if model_version:
            from tabpfn.constants import ModelVersion
            ver = getattr(ModelVersion, model_version)
            clf = TabPFNClassifier.create_default_for_version(ver, **kwargs)
        else:
            clf = TabPFNClassifier(**kwargs)

        clf.fit(X_tr_s, y_tr)
        oof_preds[val_idx] = clf.predict(X_val_s)
        proba = clf.predict_proba(X_te_s)
        # Align classes
        for i, c in enumerate(clf.classes_):
            test_proba[:, int(c)] += proba[:, i] / n_splits

    oof_acc = float(accuracy_score(y_arr, oof_preds))
    test_preds = np.argmax(test_proba, axis=1)
    return oof_acc, test_preds, test_proba


def _tabpfn_full_predict(X_train, y_train, X_test, device, seed,
                         n_estimators=16, eval_metric="f1",
                         scaler_cls=StandardScaler, model_version=None, **clf_kwargs):
    """Fit on full training data, predict test."""
    TabPFNClassifier = _require_tabpfn()
    y_arr = np.asarray(y_train)

    scaler = scaler_cls()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    kwargs = dict(n_estimators=n_estimators, device=device, random_state=seed,
                  eval_metric=eval_metric, **clf_kwargs)
    if model_version:
        from tabpfn.constants import ModelVersion
        ver = getattr(ModelVersion, model_version)
        clf = TabPFNClassifier.create_default_for_version(ver, **kwargs)
    else:
        clf = TabPFNClassifier(**kwargs)

    clf.fit(X_tr, y_arr)
    preds = clf.predict(X_te)
    return preds


# ── Experiment 1: Optuna hyperparameter search ──────────────────────────────

def experiment_optuna(X_train, y_train, X_test, test_ids, users, args):
    print("\n=== Experiment 1: Optuna hyperparameter search ===")
    import optuna

    y_arr = np.asarray(y_train)
    groups_arr = np.asarray(users)
    kf = GroupKFold(n_splits=5)
    folds = list(kf.split(X_train, y_arr, groups_arr))

    def objective(trial):
        n_est = trial.suggest_categorical("n_estimators", [4, 8, 16, 32])
        metric = trial.suggest_categorical("eval_metric", ["f1", "accuracy", "balanced_accuracy", "log_loss"])
        scaler_name = trial.suggest_categorical("scaler", ["standard", "minmax", "robust"])
        seed = trial.suggest_int("seed", 0, 100)

        scaler_map = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}
        scaler_cls = scaler_map[scaler_name]

        TabPFNClassifier = _require_tabpfn()
        fold_accs = []

        for tr_idx, val_idx in folds:
            X_tr = X_train.iloc[tr_idx]
            X_val = X_train.iloc[val_idx]
            y_tr = y_arr[tr_idx]
            y_val = y_arr[val_idx]

            scaler = scaler_cls()
            X_tr_s = scaler.fit_transform(X_tr)
            X_val_s = scaler.transform(X_val)

            try:
                clf = TabPFNClassifier(
                    n_estimators=n_est, device=args.device,
                    random_state=seed, eval_metric=metric,
                )
                clf.fit(X_tr_s, y_tr)
                preds = clf.predict(X_val_s)
                fold_accs.append(float(accuracy_score(y_val, preds)))
            except Exception:
                fold_accs.append(0.0)

            trial.report(np.mean(fold_accs), len(fold_accs) - 1)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(fold_accs)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=30, n_jobs=1)

    best = study.best_params
    print(f"  Best params: {best}")
    print(f"  Best OOF accuracy: {study.best_value:.4f}")

    # Fit with best params
    scaler_map = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}
    preds = _tabpfn_full_predict(
        X_train, y_arr, X_test, args.device, best["seed"],
        n_estimators=best["n_estimators"], eval_metric=best["eval_metric"],
        scaler_cls=scaler_map[best["scaler"]],
    )
    path = _write_sub(
        "tabpfn_optuna_best", test_ids, preds, args.output_dir,
        model="TabPFN (Optuna tuned)", features="42 base + targeted temporal",
        notes=f"params={best}; OOF={study.best_value:.4f}",
    )
    dist = {i: int((preds == i).sum()) for i in range(6)}
    print(f"  Dist: {dist}")
    print(f"  File: {path}")
    return best, study.best_value


# ── Experiment 2: Model versions ────────────────────────────────────────────

def experiment_model_versions(X_train, y_train, X_test, test_ids, users, args):
    print("\n=== Experiment 2: TabPFN model versions ===")

    from tabpfn.constants import ModelVersion
    versions = ["V2", "V2_5", "V2_6", "V3"]

    for ver in versions:
        print(f"  Version {ver}...")
        try:
            oof_acc, preds, _ = _tabpfn_oof_predict(
                X_train, y_train, users, X_test, args.device, 42,
                n_estimators=16, eval_metric="f1", n_splits=5,
                model_version=ver,
            )
            path = _write_sub(
                f"tabpfn_{ver.lower()}", test_ids, preds, args.output_dir,
                model=f"TabPFN {ver}", features="42 base + targeted temporal",
                notes=f"model_version={ver}; n_est=16; f1; OOF={oof_acc:.4f}",
            )
            dist = {i: int((preds == i).sum()) for i in range(6)}
            print(f"    OOF={oof_acc:.4f}  Dist: {dist}  File: {path}")
        except Exception as e:
            print(f"    FAILED: {e}")


# ── Experiment 3: Feature selection ─────────────────────────────────────────

def experiment_feature_selection(X_train, y_train, X_test, test_ids, users, args):
    print("\n=== Experiment 3: Feature selection (in-fold mutual information) ===")

    y_arr = np.asarray(y_train)

    for k in [50, 70, 80, 91]:
        if k > X_train.shape[1]:
            continue
        print(f"  In-fold MI top-{k}...")
        result = tabpfn_oof_predict(
            X_train, y_arr, users, X_test,
            device=args.device, seed=42,
            n_estimators=16, eval_metric="f1", n_splits=5,
            mi_top_k=k,
        )
        path = _write_sub(
            f"tabpfn_mi_top{k}_infold", test_ids, result.test_preds, args.output_dir,
            model="TabPFN", features=f"in-fold MI top-{k} of 91 features",
            notes=f"in-fold mutual_info top-{k}; n_est=16; f1; OOF acc={result.oof_accuracy:.4f}",
        )
        dist = result.prediction_distribution
        print(f"    OOF acc={result.oof_accuracy:.4f}  Dist: {dist}  File: {path}")


# ── Experiment 4: Different scalers ─────────────────────────────────────────

def experiment_scalers(X_train, y_train, X_test, test_ids, users, args):
    print("\n=== Experiment 4: Different scalers ===")

    scalers = [
        ("standard", StandardScaler),
        ("minmax", MinMaxScaler),
        ("robust", RobustScaler),
        ("none", None),
    ]

    y_arr = np.asarray(y_train)

    for name, scaler_cls in scalers:
        print(f"  Scaler: {name}...")
        if scaler_cls is None:
            # No scaling — TabPFN handles its own preprocessing
            TabPFNClassifier = _require_tabpfn()
            groups_arr = np.asarray(users)
            kf = GroupKFold(n_splits=5)
            oof_preds = np.zeros(len(y_arr), dtype=int)
            test_proba = np.zeros((len(X_test), N_CLASSES), dtype=float)

            for tr_idx, val_idx in kf.split(X_train, y_arr, groups_arr):
                X_tr = X_train.iloc[tr_idx]
                X_val = X_train.iloc[val_idx]
                y_tr = y_arr[tr_idx]

                clf = TabPFNClassifier(n_estimators=16, device=args.device, random_state=42, eval_metric="f1")
                clf.fit(np.asarray(X_tr), y_tr)
                oof_preds[val_idx] = clf.predict(np.asarray(X_val))
                proba = clf.predict_proba(np.asarray(X_test))
                for i, c in enumerate(clf.classes_):
                    test_proba[:, int(c)] += proba[:, i] / 5

            oof_acc = float(accuracy_score(y_arr, oof_preds))
            preds = np.argmax(test_proba, axis=1)
        else:
            oof_acc, preds, _ = _tabpfn_oof_predict(
                X_train, y_arr, users, X_test, args.device, 42,
                n_estimators=16, eval_metric="f1", n_splits=5,
                scaler_cls=scaler_cls,
            )

        path = _write_sub(
            f"tabpfn_{name}", test_ids, preds, args.output_dir,
            model="TabPFN", features="42 base + targeted temporal",
            notes=f"scaler={name}; n_est=16; f1; OOF={oof_acc:.4f}",
        )
        dist = {i: int((preds == i).sum()) for i in range(6)}
        print(f"    OOF={oof_acc:.4f}  Dist: {dist}")


# ── Experiment 5: Gaussian noise augmentation ───────────────────────────────

def experiment_augmentation(X_train, y_train, X_test, test_ids, users, args):
    print("\n=== Experiment 5: Gaussian noise augmentation ===")

    y_arr = np.asarray(y_train)
    noise_levels = [0.01, 0.02, 0.05]

    for noise in noise_levels:
        print(f"  Noise std={noise}...")
        # Add noise to training data
        rng = np.random.RandomState(42)
        X_noisy = np.asarray(X_train) + rng.normal(0, noise, np.asarray(X_train).shape)
        X_noisy_df = pd.DataFrame(X_noisy, columns=X_train.columns)

        oof_acc, preds, _ = _tabpfn_oof_predict(
            X_noisy_df, y_arr, users, X_test, args.device, 42,
            n_estimators=16, eval_metric="f1", n_splits=5,
        )
        path = _write_sub(
            f"tabpfn_noise_{noise}", test_ids, preds, args.output_dir,
            model="TabPFN (noisy train)", features="42 base + targeted temporal",
            notes=f"gaussian noise std={noise}; n_est=16; f1; OOF={oof_acc:.4f}",
        )
        dist = {i: int((preds == i).sum()) for i in range(6)}
        print(f"    OOF={oof_acc:.4f}  Dist: {dist}")


# ── Experiment 6: OOF cross-validated predictions ───────────────────────────

def experiment_oof_predictions(X_train, y_train, X_test, test_ids, users, args):
    print("\n=== Experiment 6: OOF cross-validated predictions ===")

    y_arr = np.asarray(y_train)
    groups_arr = np.asarray(users)

    # Use OOF predictions instead of full-data fit
    oof_acc, preds, proba = _tabpfn_oof_predict(
        X_train, y_arr, users, X_test, args.device, 42,
        n_estimators=16, eval_metric="f1", n_splits=5,
    )
    path = _write_sub(
        "tabpfn_oof_cv", test_ids, preds, args.output_dir,
        model="TabPFN (OOF CV)", features="42 base + targeted temporal",
        notes=f"OOF cross-validated predictions (5-fold); n_est=16; f1; OOF={oof_acc:.4f}",
    )
    dist = {i: int((preds == i).sum()) for i in range(6)}
    print(f"  OOF={oof_acc:.4f}  Dist: {dist}  File: {path}")


# ── Experiment 7: Extended feature set ──────────────────────────────────────

def experiment_extended_features(X_base_train, y_train, X_base_test, test_ids,
                                 X_seq, X_test_seq, users, args):
    print("\n=== Experiment 7: Extended features ===")

    # Hjorth + spectral
    hjorth_train = build_hjorth_spectral_features(X_seq)
    hjorth_test = build_hjorth_spectral_features(X_test_seq)

    X_targ_train = combine_base_and_temporal_features(X_base_train, X_seq)
    X_targ_test = combine_base_and_temporal_features(X_base_test, X_test_seq)

    X_full_train = pd.concat([X_targ_train.reset_index(drop=True), hjorth_train.reset_index(drop=True)], axis=1)
    X_full_test = pd.concat([X_targ_test.reset_index(drop=True), hjorth_test.reset_index(drop=True)], axis=1)

    y_arr = np.asarray(y_train)

    for k in [80, 100, 127]:
        print(f"  Full features, in-fold MI top-{k}...")
        result = tabpfn_oof_predict(
            X_full_train, y_arr, users, X_full_test,
            device=args.device, seed=42,
            n_estimators=16, eval_metric="f1", n_splits=5,
            mi_top_k=k,
        )
        path = _write_sub(
            f"tabpfn_full_mi{k}_infold", test_ids, result.test_preds, args.output_dir,
            model="TabPFN", features=f"42+targeted+Hjorth/spectral, in-fold MI top-{k}",
            notes=f"extended (127 cols), in-fold MI top-{k}; n_est=16; f1; OOF acc={result.oof_accuracy:.4f}",
        )
        print(f"    OOF acc={result.oof_accuracy:.4f}  Dist: {result.prediction_distribution}  File: {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--start-from", type=int, default=1, help="Start from experiment N (1-7)")
    args = parser.parse_args()

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

    # Run experiments (skip if --start-from > experiment number)
    if args.start_from <= 1:
        experiment_optuna(X_train_t, y_train, X_test_t, test_ids, users, args)
    if args.start_from <= 2:
        experiment_model_versions(X_train_t, y_train, X_test_t, test_ids, users, args)
    if args.start_from <= 3:
        experiment_feature_selection(X_train_t, y_train, X_test_t, test_ids, users, args)
    if args.start_from <= 4:
        experiment_scalers(X_train_t, y_train, X_test_t, test_ids, users, args)
    if args.start_from <= 5:
        experiment_augmentation(X_train_t, y_train, X_test_t, test_ids, users, args)
    if args.start_from <= 6:
        experiment_oof_predictions(X_train_t, y_train, X_test_t, test_ids, users, args)
    if args.start_from <= 7:
        experiment_extended_features(X_train, y_train, X_test, test_ids, X_seq, X_test_seq, users, args)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
