from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from model.user_norm_features import build_user_norm_features, fit_user_norm_stats
from model.validation import (
    confusion_matrix_dict,
    effective_group_kfold_splits,
    mutual_info_top_k_columns,
    prediction_distribution,
)

N_CLASSES = 6
VALID_LABELS = (0, 1, 2, 3, 4, 5)


def _require_tabpfn():
    """Import TabPFN; auth uses env, ~/.cache/tabpfn/auth_token, or browser on model load."""
    try:
        from tabpfn import TabPFNClassifier
        from tabpfn.browser_auth import get_cached_token
    except ImportError as exc:
        raise ImportError(
            "tabpfn is required. Install with: pip install tabpfn"
        ) from exc
    if get_cached_token() is None:
        import warnings

        warnings.warn(
            "No TabPFN token in TABPFN_TOKEN or ~/.cache/tabpfn/auth_token. "
            "Loading V2.5/V2.6/V3 will open a browser for Prior Labs login, or set "
            "TABPFN_TOKEN from https://ux.priorlabs.ai/account",
            stacklevel=2,
        )
    return TabPFNClassifier


def _take_rows(X, idx):
    return X.iloc[idx] if hasattr(X, "iloc") else X[idx]


def _take_seq(X_seq, idx):
    return X_seq[idx]


def _take_users(users, idx):
    return users.iloc[idx] if hasattr(users, "iloc") else users[idx]


def _build_classifier(TabPFNClassifier, n_estimators, device, seed, eval_metric, model_version, clf_kwargs):
    kwargs = dict(
        n_estimators=n_estimators,
        device=device,
        random_state=seed,
        eval_metric=eval_metric,
        **clf_kwargs,
    )
    if model_version:
        from tabpfn.constants import ModelVersion
        ver = getattr(ModelVersion, model_version)
        return TabPFNClassifier.create_default_for_version(ver, **kwargs)
    return TabPFNClassifier(**kwargs)


@dataclass
class TabPFNOOFResult:
    oof_accuracy: float
    oof_macro_f1: float
    fold_accuracies: list[float]
    oof_preds: np.ndarray
    oof_proba: np.ndarray
    test_preds: np.ndarray
    test_proba: np.ndarray
    classes: np.ndarray
    confusion: dict
    prediction_distribution: dict
    member_seeds: list[int] | None = None


@dataclass
class FinetuneOOFResult:
    oof_accuracy: float
    oof_macro_f1: float
    fold_accuracies: list[float]
    oof_preds: np.ndarray


def tabpfn_oof_predict(
    X_train,
    y_train,
    users,
    X_test,
    *,
    device="cuda",
    seed=42,
    n_estimators=16,
    eval_metric="accuracy",
    n_splits=5,
    model_version=None,
    mi_top_k=None,
    fit_mode=None,
    tuning_config=None,
    clf_kwargs=None,
    X_seq=None,
    X_test_seq=None,
    users_test=None,
    user_norm=False,
):
    """
    Grouped OOF evaluation with optional in-fold MI feature selection.

    Scaler and MI are fit on each training fold only. Test predictions are the
    average of fold-wise probability vectors.
    """
    TabPFNClassifier = _require_tabpfn()
    clf_kwargs = dict(clf_kwargs or {})
    if fit_mode is not None:
        clf_kwargs["fit_mode"] = fit_mode
    if tuning_config is not None:
        clf_kwargs["tuning_config"] = tuning_config

    if user_norm and (X_seq is None or X_test_seq is None or users_test is None):
        raise ValueError("user_norm=True requires X_seq, X_test_seq, and users_test")

    y_arr = np.asarray(y_train)
    groups_arr = np.asarray(users)
    classes = np.unique(y_arr)
    n_splits = effective_group_kfold_splits(groups_arr, n_splits)
    kf = GroupKFold(n_splits=n_splits)

    oof_preds = np.zeros(len(y_arr), dtype=int)
    oof_proba = np.zeros((len(y_arr), len(classes)), dtype=float)
    test_proba = np.zeros((len(X_test), len(classes)), dtype=float)
    fold_accuracies: list[float] = []

    for tr_idx, val_idx in kf.split(X_train, y_arr, groups_arr):
        X_tr = _take_rows(X_train, tr_idx)
        X_val = _take_rows(X_train, val_idx)
        X_te = X_test
        y_tr = y_arr[tr_idx]

        if user_norm:
            stats = fit_user_norm_stats(_take_seq(X_seq, tr_idx), _take_users(users, tr_idx))
            unnorm_tr = build_user_norm_features(_take_seq(X_seq, tr_idx), _take_users(users, tr_idx), stats)
            unnorm_val = build_user_norm_features(_take_seq(X_seq, val_idx), _take_users(users, val_idx), stats)
            unnorm_te = build_user_norm_features(X_test_seq, users_test, stats)
            X_tr = pd.concat([X_tr.reset_index(drop=True), unnorm_tr.reset_index(drop=True)], axis=1)
            X_val = pd.concat([X_val.reset_index(drop=True), unnorm_val.reset_index(drop=True)], axis=1)
            X_te = pd.concat([X_test.reset_index(drop=True), unnorm_te.reset_index(drop=True)], axis=1)
        else:
            X_te = X_test

        if mi_top_k is not None:
            cols = mutual_info_top_k_columns(X_tr, y_tr, mi_top_k, random_state=seed)
            X_tr = X_tr[cols]
            X_val = X_val[cols]
            X_te = X_te[cols]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)
        X_te_s = scaler.transform(X_te)

        clf = _build_classifier(
            TabPFNClassifier, n_estimators, device, seed, eval_metric, model_version, clf_kwargs
        )
        clf.fit(X_tr_s, y_tr)
        val_proba_raw = clf.predict_proba(X_val_s)
        fold_preds = classes[np.argmax(
            _align_proba(val_proba_raw, clf.classes_, classes), axis=1
        )].astype(int)
        oof_preds[val_idx] = fold_preds
        oof_proba[val_idx] = _align_proba(val_proba_raw, clf.classes_, classes)
        fold_accuracies.append(float(accuracy_score(y_arr[val_idx], fold_preds)))
        proba = clf.predict_proba(X_te_s)
        test_proba += _align_proba(proba, clf.classes_, classes) / n_splits

    test_preds = classes[np.argmax(test_proba, axis=1)].astype(int)
    return TabPFNOOFResult(
        oof_accuracy=float(accuracy_score(y_arr, oof_preds)),
        oof_macro_f1=float(f1_score(y_arr, oof_preds, average="macro", zero_division=0)),
        fold_accuracies=fold_accuracies,
        oof_preds=oof_preds,
        oof_proba=oof_proba,
        test_preds=test_preds,
        test_proba=test_proba,
        classes=classes,
        confusion=confusion_matrix_dict(y_arr, oof_preds),
        prediction_distribution=prediction_distribution(test_preds),
    )


def _align_proba(proba_raw, fold_classes, all_classes):
    aligned = np.zeros((proba_raw.shape[0], len(all_classes)), dtype=float)
    for slot, class_label in enumerate(fold_classes):
        class_idx = int(np.where(all_classes == class_label)[0][0])
        aligned[:, class_idx] = proba_raw[:, slot]
    return aligned


def tabpfn_prob_ensemble_predict(
    X_train,
    y_train,
    users,
    X_test,
    seeds,
    *,
    device="cuda",
    n_estimators=16,
    eval_metric="f1",
    n_splits=5,
    model_version="V3",
    mi_top_k=None,
    fit_mode=None,
    tuning_config=None,
    clf_kwargs=None,
    X_seq=None,
    X_test_seq=None,
    users_test=None,
    user_norm=False,
):
    """Grouped OOF with test/OOF probabilities averaged across seeds (one submission)."""
    TabPFNClassifier = _require_tabpfn()
    clf_kwargs = dict(clf_kwargs or {})
    if fit_mode is not None:
        clf_kwargs["fit_mode"] = fit_mode
    if tuning_config is not None:
        clf_kwargs["tuning_config"] = tuning_config
    if user_norm and (X_seq is None or X_test_seq is None or users_test is None):
        raise ValueError("user_norm=True requires X_seq, X_test_seq, and users_test")

    seeds = [int(s) for s in seeds]
    if not seeds:
        raise ValueError("seeds must be non-empty")

    y_arr = np.asarray(y_train)
    groups_arr = np.asarray(users)
    classes = np.unique(y_arr)
    n_splits = effective_group_kfold_splits(groups_arr, n_splits)
    kf = GroupKFold(n_splits=n_splits)

    oof_proba = np.zeros((len(y_arr), len(classes)), dtype=float)
    test_proba = np.zeros((len(X_test), len(classes)), dtype=float)
    fold_accuracies: list[float] = []

    for tr_idx, val_idx in kf.split(X_train, y_arr, groups_arr):
        fold_oof = np.zeros((len(val_idx), len(classes)), dtype=float)
        fold_test = np.zeros((len(X_test), len(classes)), dtype=float)

        for seed in seeds:
            X_tr = _take_rows(X_train, tr_idx)
            X_val = _take_rows(X_train, val_idx)
            X_te = X_test
            y_tr = y_arr[tr_idx]

            if user_norm:
                stats = fit_user_norm_stats(_take_seq(X_seq, tr_idx), _take_users(users, tr_idx))
                unnorm_tr = build_user_norm_features(_take_seq(X_seq, tr_idx), _take_users(users, tr_idx), stats)
                unnorm_val = build_user_norm_features(_take_seq(X_seq, val_idx), _take_users(users, val_idx), stats)
                unnorm_te = build_user_norm_features(X_test_seq, users_test, stats)
                X_tr = pd.concat([X_tr.reset_index(drop=True), unnorm_tr.reset_index(drop=True)], axis=1)
                X_val = pd.concat([X_val.reset_index(drop=True), unnorm_val.reset_index(drop=True)], axis=1)
                X_te = pd.concat([X_test.reset_index(drop=True), unnorm_te.reset_index(drop=True)], axis=1)

            if mi_top_k is not None:
                cols = mutual_info_top_k_columns(X_tr, y_tr, mi_top_k, random_state=seed)
                X_tr = X_tr[cols]
                X_val = X_val[cols]
                X_te = X_te[cols]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_val_s = scaler.transform(X_val)
            X_te_s = scaler.transform(X_te)

            clf = _build_classifier(
                TabPFNClassifier,
                n_estimators,
                device,
                seed,
                eval_metric,
                model_version,
                clf_kwargs,
            )
            clf.fit(X_tr_s, y_tr)
            val_raw = clf.predict_proba(X_val_s)
            test_raw = clf.predict_proba(X_te_s)
            fold_oof += _align_proba(val_raw, clf.classes_, classes) / len(seeds)
            fold_test += _align_proba(test_raw, clf.classes_, classes) / len(seeds)

        oof_proba[val_idx] = fold_oof
        test_proba += fold_test / n_splits
        fold_preds = classes[np.argmax(fold_oof, axis=1)].astype(int)
        fold_accuracies.append(float(accuracy_score(y_arr[val_idx], fold_preds)))

    oof_preds = classes[np.argmax(oof_proba, axis=1)].astype(int)
    test_preds = classes[np.argmax(test_proba, axis=1)].astype(int)
    return TabPFNOOFResult(
        oof_accuracy=float(accuracy_score(y_arr, oof_preds)),
        oof_macro_f1=float(f1_score(y_arr, oof_preds, average="macro", zero_division=0)),
        fold_accuracies=fold_accuracies,
        oof_preds=oof_preds,
        oof_proba=oof_proba,
        test_preds=test_preds,
        test_proba=test_proba,
        classes=classes,
        confusion=confusion_matrix_dict(y_arr, oof_preds),
        prediction_distribution=prediction_distribution(test_preds),
        member_seeds=seeds,
    )


def make_finetuned_classifier(cfg, device):
    """Build FinetunedTabPFNClassifier (finetune loss: log_loss or roc_auc per TabPFN API)."""
    from tabpfn.finetuning.finetuned_classifier import FinetunedTabPFNClassifier

    return FinetunedTabPFNClassifier(
        device=device,
        epochs=cfg["epochs"],
        learning_rate=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0.01),
        n_estimators_finetune=cfg.get("n_est_finetune", 2),
        n_estimators_validation=cfg.get("n_est_validation", 2),
        n_estimators_final_inference=cfg.get("n_est_final", 16),
        early_stopping=True,
        early_stopping_patience=cfg.get("patience", 8),
        eval_metric=cfg.get("finetune_eval_metric", "log_loss"),
        random_state=cfg.get("random_state", 42),
    )


def finetuned_grouped_oof(X_train, y_train, users, cfg, device, n_splits=5):
    """Grouped OOF for finetuned TabPFN; submit gates should use OOF accuracy, not finetune eval_metric."""
    y_arr = np.asarray(y_train)
    groups_arr = np.asarray(users)
    kf = GroupKFold(n_splits=n_splits)
    oof_preds = np.zeros(len(y_arr), dtype=int)
    fold_accuracies: list[float] = []

    for tr_idx, val_idx in kf.split(X_train, y_arr, groups_arr):
        X_tr = _take_rows(X_train, tr_idx)
        X_val = _take_rows(X_train, val_idx)
        y_tr = y_arr[tr_idx]
        y_val = y_arr[val_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        clf = make_finetuned_classifier(cfg, device)
        clf.fit(X_tr_s, y_tr)
        fold_preds = clf.predict(X_val_s)
        oof_preds[val_idx] = fold_preds
        fold_accuracies.append(float(accuracy_score(y_val, fold_preds)))

    return FinetuneOOFResult(
        oof_accuracy=float(accuracy_score(y_arr, oof_preds)),
        oof_macro_f1=float(f1_score(y_arr, oof_preds, average="macro", zero_division=0)),
        fold_accuracies=fold_accuracies,
        oof_preds=oof_preds,
    )


def fit_tabpfn_oof(X, y, groups, n_splits=5, n_estimators=16, device="cuda", random_state=42):
    """Legacy API: grouped OOF class probabilities (eval_metric=f1)."""
    TabPFNClassifier = _require_tabpfn()
    y_arr = np.asarray(y)
    groups_arr = np.asarray(groups)
    classes = np.unique(y_arr)
    oof_proba = np.zeros((len(y_arr), len(classes)), dtype=float)

    kf = GroupKFold(n_splits=n_splits)
    for train_idx, val_idx in kf.split(X, y_arr, groups_arr):
        X_tr = _take_rows(X, train_idx)
        X_val = _take_rows(X, val_idx)
        y_tr = y_arr[train_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        clf = TabPFNClassifier(
            n_estimators=n_estimators,
            device=device,
            random_state=random_state,
            eval_metric="f1",
        )
        clf.fit(X_tr_s, y_tr)
        val_proba_raw = clf.predict_proba(X_val_s)
        fold_classes = clf.classes_

        val_proba = np.zeros((len(val_proba_raw), len(classes)), dtype=float)
        for slot, c in enumerate(fold_classes):
            idx = np.where(classes == c)[0][0]
            val_proba[:, idx] = val_proba_raw[:, slot]
        oof_proba[val_idx] = val_proba

    oof_preds = classes[np.argmax(oof_proba, axis=1)]
    return oof_proba, oof_preds


def fit_tabpfn_full(X, y, n_estimators=16, device="cuda", random_state=42, eval_metric="f1"):
    TabPFNClassifier = _require_tabpfn()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = TabPFNClassifier(
        n_estimators=n_estimators,
        device=device,
        random_state=random_state,
        eval_metric=eval_metric,
    )
    clf.fit(X_scaled, np.asarray(y))
    return clf, scaler


def predict_tabpfn(model_scaler, X_test):
    model, scaler = model_scaler
    X_scaled = scaler.transform(X_test)
    return model.predict(X_scaled).astype(int)


def predict_tabpfn_proba(model_scaler, X_test):
    model, scaler = model_scaler
    X_scaled = scaler.transform(X_test)
    return model.predict_proba(X_scaled), model.classes_
