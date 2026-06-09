import numpy as np
import pandas as pd

from model.dart_train import dart_default_params, fit_dart_full, predict_dart


def test_dart_default_params_enable_dart_booster():
    params = dart_default_params(n_jobs=2, random_state=7)

    assert params["booster"] == "dart"
    assert params["n_jobs"] == 2
    assert params["random_state"] == 7
    assert "rate_drop" in params


def test_fit_dart_full_predicts_labels():
    X = pd.DataFrame({"a": [0.0, 0.1, 1.0, 1.1, 2.0, 2.1], "b": [1.0, 1.1, 0.0, 0.1, 2.0, 2.1]})
    y = pd.Series([0, 0, 1, 1, 2, 2])
    X_test = pd.DataFrame({"a": [0.05, 1.05, 2.05], "b": [1.05, 0.05, 2.05]})

    model = fit_dart_full(X, y, n_jobs=1, random_state=42, n_estimators=10)
    preds = predict_dart(model, X_test)

    assert preds.shape == (3,)
    assert set(preds.tolist()) <= {0, 1, 2}
