import importlib.util

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("catboost") is None, reason="catboost not installed")


def test_catboost_default_params_respect_cpu_limit():
    from model.catboost_train import catboost_default_params

    params = catboost_default_params(n_jobs=3, random_state=9)

    assert params["thread_count"] == 3
    assert params["random_seed"] == 9
    assert params["loss_function"] == "MultiClass"


def test_fit_catboost_full_predicts_labels():
    from model.catboost_train import fit_catboost_full, predict_catboost

    X = pd.DataFrame({"a": [0.0, 0.1, 1.0, 1.1, 2.0, 2.1], "b": [1.0, 1.1, 0.0, 0.1, 2.0, 2.1]})
    y = pd.Series([0, 0, 1, 1, 2, 2])
    X_test = pd.DataFrame({"a": [0.05, 1.05, 2.05], "b": [1.05, 0.05, 2.05]})

    model = fit_catboost_full(X, y, n_jobs=1, random_state=42, iterations=10)
    preds = predict_catboost(model, X_test)

    assert preds.shape == (3,)
    assert set(preds.tolist()) <= {0, 1, 2}
