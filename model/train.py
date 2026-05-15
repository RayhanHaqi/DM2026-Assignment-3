import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

import optuna
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def cv_evaluate(model, X, y, groups, n_splits=5):
    """GroupKFold CV returning per-fold F1-macro and mean/std."""
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

        model.fit(X_tr_s, y_tr)
        preds = model.predict(X_val_s)
        score = f1_score(y_val, preds, average="macro")
        scores.append(score)
    return scores, np.mean(scores), np.std(scores)


def _xgb_objective(trial, X, y, groups):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": 42,
        "n_jobs": -1,
    }
    model = XGBClassifier(**params)
    kf = GroupKFold(n_splits=5)
    fold_scores = []
    for fold_i, (train_idx, val_idx) in enumerate(kf.split(X, y, groups)):
        X_tr = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_tr = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        model.fit(X_tr_s, y_tr)
        preds = model.predict(X_val_s)
        fold_f1 = f1_score(y_val, preds, average="macro")
        fold_scores.append(fold_f1)

        trial.report(np.mean(fold_scores), fold_i)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(fold_scores)


def tune_xgboost(X, y, groups, n_trials=50):
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(
        lambda trial: _xgb_objective(trial, X, y, groups),
        n_trials=n_trials,
        n_jobs=1,
    )
    best_params = study.best_params
    best_params.update({"random_state": 42, "n_jobs": -1})
    best_model = XGBClassifier(**best_params)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    best_model.fit(X_scaled, y)
    return best_params, best_model


def _lgb_objective(trial, X, y, groups):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    model = LGBMClassifier(**params)
    kf = GroupKFold(n_splits=5)
    fold_scores = []
    for fold_i, (train_idx, val_idx) in enumerate(kf.split(X, y, groups)):
        X_tr = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_tr = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        model.fit(X_tr_s, y_tr)
        preds = model.predict(X_val_s)
        fold_f1 = f1_score(y_val, preds, average="macro")
        fold_scores.append(fold_f1)

        trial.report(np.mean(fold_scores), fold_i)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(fold_scores)


def tune_lightgbm(X, y, groups, n_trials=50):
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(
        lambda trial: _lgb_objective(trial, X, y, groups),
        n_trials=n_trials,
        n_jobs=1,
    )
    best_params = study.best_params
    best_params.update({"class_weight": "balanced", "random_state": 42, "n_jobs": -1, "verbose": -1})
    best_model = LGBMClassifier(**best_params)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    best_model.fit(X_scaled, y)
    return best_params, best_model
