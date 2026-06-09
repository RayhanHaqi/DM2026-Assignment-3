import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import optuna

from model.train import _apply_smote, _score_predictions


def _require_catboost():
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise ImportError("catboost is required for CatBoost experiments. Install with: pip install catboost") from exc
    return CatBoostClassifier


def catboost_default_params(n_jobs=4, random_state=42, iterations=500):
    return {
        "loss_function": "MultiClass",
        "eval_metric": "Accuracy",
        "iterations": iterations,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 3.0,
        "random_strength": 1.0,
        "bagging_temperature": 1.0,
        "random_seed": random_state,
        "thread_count": n_jobs,
        "verbose": False,
        "allow_writing_files": False,
    }


def fit_catboost_full(X, y, n_jobs=4, random_state=42, iterations=500, params=None):
    CatBoostClassifier = _require_catboost()
    fit_params = catboost_default_params(n_jobs=n_jobs, random_state=random_state, iterations=iterations)
    if params is not None:
        fit_params.update(params)
    model = CatBoostClassifier(**fit_params)
    model.fit(X, np.asarray(y))
    return model


def predict_catboost(model, X_test):
    return model.predict(X_test).reshape(-1).astype(int)


def predict_catboost_proba(model, X_test):
    return model.predict_proba(X_test)


def _cat_objective(trial, X, y, groups, metric="f1_macro", use_smote=True, n_jobs=4):
    CatBoostClassifier = _require_catboost()
    params = {
        "loss_function": "MultiClass",
        "eval_metric": "Accuracy",
        "iterations": trial.suggest_int("iterations", 100, 800),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
        "random_strength": trial.suggest_float("random_strength", 0.5, 10.0),
        "random_seed": 42,
        "thread_count": n_jobs,
        "verbose": False,
        "allow_writing_files": False,
    }
    model = CatBoostClassifier(**params)
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

        if use_smote:
            X_tr_s, y_tr = _apply_smote(X_tr_s, y_tr)

        try:
            model.fit(X_tr_s, y_tr)
            preds = model.predict(X_val_s)
            fold_scores.append(_score_predictions(y_val, preds, metric))
        except Exception:
            fold_scores.append(0.0)

        trial.report(np.mean(fold_scores), fold_i)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(fold_scores)


def tune_catboost(X, y, groups, n_trials=50, metric="f1_macro", use_smote=True, n_jobs=4):
    CatBoostClassifier = _require_catboost()
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(
        lambda trial: _cat_objective(trial, X, y, groups, metric=metric, use_smote=use_smote, n_jobs=n_jobs),
        n_trials=n_trials,
        n_jobs=1,
    )
    best_params = study.best_params
    final_params = {
        "loss_function": "MultiClass",
        "eval_metric": "Accuracy",
        **best_params,
        "random_seed": 42,
        "thread_count": n_jobs,
        "verbose": False,
        "allow_writing_files": False,
    }
    best_model = CatBoostClassifier(**final_params)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    try:
        best_model.fit(X_scaled, np.asarray(y))
    except Exception:
        best_model.fit(X_scaled, np.asarray(y).astype(float))
    return final_params, best_model
