import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

import optuna
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def _score_predictions(y_true, preds, metric):
    if metric == "f1_macro":
        return f1_score(y_true, preds, average="macro")
    if metric == "accuracy":
        return accuracy_score(y_true, preds)
    raise ValueError("metric must be 'f1_macro' or 'accuracy'")


def _score_from_proba(y_true, proba, classes, metric):
    from sklearn.metrics import log_loss
    if metric == "neg_log_loss":
        unique_labels = np.unique(np.asarray(y_true))
        if len(unique_labels) < 2:
            return 0.0
        return -log_loss(y_true, proba, labels=classes)
    raise ValueError("_score_from_proba only supports 'neg_log_loss'")


def _apply_smote(X, y):
    from imblearn.over_sampling import SMOTE
    class_counts = np.bincount(np.asarray(y))
    present_counts = class_counts[class_counts > 0]
    if len(present_counts) < 2:
        return X, y
    min_count = present_counts.min()
    if min_count < 2:
        return X, y
    smote = SMOTE(random_state=42, k_neighbors=min(5, min_count - 1))
    return smote.fit_resample(X, y)


def _take_row(X, idx):
    return X.iloc[idx] if hasattr(X, "iloc") else X[idx]


def cv_evaluate(model, X, y, groups, n_splits=5, metric="f1_macro", use_smote=False):
    """GroupKFold CV returning per-fold scores and mean/std."""
    kf = GroupKFold(n_splits=n_splits)
    scores = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y, groups)):
        X_tr = X.iloc[train_idx] if hasattr(X, "iloc") else X[train_idx]
        X_val = X.iloc[val_idx] if hasattr(X, "iloc") else X[val_idx]
        y_tr = y.iloc[train_idx] if hasattr(y, "iloc") else y[train_idx]
        y_val = y.iloc[val_idx] if hasattr(y, "iloc") else y[val_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        if use_smote:
            X_tr_s, y_tr = _apply_smote(X_tr_s, y_tr)

        fold_model = clone(model)
        fold_model.fit(X_tr_s, y_tr)
        preds = fold_model.predict(X_val_s)
        score = _score_predictions(y_val, preds, metric)
        scores.append(score)
    return scores, np.mean(scores), np.std(scores)


def _xgb_objective(trial, X, y, groups, metric="f1_macro", use_smote=True, n_jobs=-1, n_repeats=1):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": 42,
        "n_jobs": n_jobs,
    }
    model = XGBClassifier(**params)
    fold_scores = []

    for repeat_i in range(n_repeats):
        if n_repeats > 1:
            try:
                kf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42 + repeat_i)
            except (NameError, TypeError):
                kf = GroupKFold(n_splits=5)
        else:
            kf = GroupKFold(n_splits=5)

        for fold_i, (train_idx, val_idx) in enumerate(kf.split(X, y, groups)):
            X_tr = _take_row(X, train_idx)
            X_val = _take_row(X, val_idx)
            y_tr = _take_row(y, train_idx)
            y_val = _take_row(y, val_idx)

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_val_s = scaler.transform(X_val)

            if use_smote:
                X_tr_s, y_tr = _apply_smote(X_tr_s, y_tr)

            model.fit(X_tr_s, y_tr)

            if metric in ("f1_macro", "accuracy"):
                preds = model.predict(X_val_s)
                fold_scores.append(_score_predictions(y_val, preds, metric))
            elif metric == "neg_log_loss":
                proba = model.predict_proba(X_val_s)
                fold_scores.append(_score_from_proba(y_val, proba, model.classes_, metric))

            trial.report(np.mean(fold_scores), repeat_i * 5 + fold_i)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return np.mean(fold_scores)


def tune_xgboost(X, y, groups, n_trials=50, metric="f1_macro", use_smote=True, n_jobs=-1, n_repeats=1):
    if metric not in ("f1_macro", "accuracy", "neg_log_loss"):
        raise ValueError(f"Unknown metric: {metric}")

    if n_repeats > 1:
        n_trials = max(n_trials, int(n_trials * 1.2))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(
        lambda trial: _xgb_objective(
            trial, X, y, groups, metric=metric, use_smote=use_smote,
            n_jobs=n_jobs, n_repeats=n_repeats,
        ),
        n_trials=n_trials,
        n_jobs=1,
    )
    best_params = study.best_params
    best_params.update({"random_state": 42, "n_jobs": n_jobs})
    best_model = XGBClassifier(**best_params)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    if use_smote:
        X_scaled, y = _apply_smote(X_scaled, y)
    best_model.fit(X_scaled, y)
    return best_params, best_model


def _lgb_objective(trial, X, y, groups, metric="f1_macro", use_smote=True, n_jobs=-1, n_repeats=1):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": n_jobs,
        "verbose": -1,
    }
    model = LGBMClassifier(**params)
    fold_scores = []

    for repeat_i in range(n_repeats):
        if n_repeats > 1:
            try:
                kf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42 + repeat_i)
            except (NameError, TypeError):
                kf = GroupKFold(n_splits=5)
        else:
            kf = GroupKFold(n_splits=5)

        for fold_i, (train_idx, val_idx) in enumerate(kf.split(X, y, groups)):
            X_tr = _take_row(X, train_idx)
            X_val = _take_row(X, val_idx)
            y_tr = _take_row(y, train_idx)
            y_val = _take_row(y, val_idx)

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_val_s = scaler.transform(X_val)

            if use_smote:
                X_tr_s, y_tr = _apply_smote(X_tr_s, y_tr)

            model.fit(X_tr_s, y_tr)
            preds = model.predict(X_val_s)
            fold_scores.append(_score_predictions(y_val, preds, metric))

            trial.report(np.mean(fold_scores), repeat_i * 5 + fold_i)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return np.mean(fold_scores)


def tune_lightgbm(X, y, groups, n_trials=50, metric="f1_macro", use_smote=True, n_jobs=-1, n_repeats=1):
    if n_repeats > 1:
        n_trials = max(n_trials, int(n_trials * 1.2))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(
        lambda trial: _lgb_objective(
            trial, X, y, groups, metric=metric, use_smote=use_smote,
            n_jobs=n_jobs, n_repeats=n_repeats,
        ),
        n_trials=n_trials,
        n_jobs=1,
    )
    best_params = study.best_params
    best_params.update({"class_weight": "balanced", "random_state": 42, "n_jobs": n_jobs, "verbose": -1})
    best_model = LGBMClassifier(**best_params)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    if use_smote:
        X_scaled, y = _apply_smote(X_scaled, y)
    best_model.fit(X_scaled, y)
    return best_params, best_model
