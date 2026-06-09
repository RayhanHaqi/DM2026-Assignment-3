import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def dart_default_params(n_jobs=4, random_state=42, n_estimators=500):
    return {
        "booster": "dart",
        "n_estimators": n_estimators,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
        "rate_drop": 0.1,
        "skip_drop": 0.5,
        "sample_type": "uniform",
        "normalize_type": "tree",
        "random_state": random_state,
        "n_jobs": n_jobs,
    }


def fit_dart_full(X, y, n_jobs=4, random_state=42, n_estimators=500, params=None):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    fit_params = dart_default_params(n_jobs=n_jobs, random_state=random_state, n_estimators=n_estimators)
    if params is not None:
        fit_params.update(params)
    model = XGBClassifier(**fit_params)
    model.fit(X_scaled, np.asarray(y))
    model._har_scaler = scaler
    return model


def predict_dart(model, X_test):
    X_scaled = model._har_scaler.transform(X_test)
    return model.predict(X_scaled).astype(int)


def predict_dart_proba(model, X_test):
    X_scaled = model._har_scaler.transform(X_test)
    return model.predict_proba(X_scaled)
