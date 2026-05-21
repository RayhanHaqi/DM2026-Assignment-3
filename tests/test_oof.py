import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

from model.oof import evaluate_oof_model, prediction_distribution, search_weighted_ensemble


def test_prediction_distribution_counts_all_labels():
    assert prediction_distribution([0, 1, 1, 5]) == {0: 1, 1: 2, 2: 0, 3: 0, 4: 0, 5: 1}


def test_evaluate_oof_model_returns_probabilities_and_metrics():
    X = pd.DataFrame({"a": np.linspace(0, 1, 18), "b": np.linspace(1, 0, 18)})
    y = pd.Series([0, 1, 2] * 6)
    groups = pd.Series([f"u{i // 3}" for i in range(18)])
    X_test = pd.DataFrame({"a": [0.1, 0.5], "b": [0.9, 0.5]})
    model = ExtraTreesClassifier(n_estimators=10, random_state=42)

    result = evaluate_oof_model(model, X, y, groups, X_test, n_splits=3)

    assert result.oof_proba.shape == (18, 3)
    assert result.test_proba.shape == (2, 3)
    assert len(result.fold_accuracy) == 3
    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.macro_f1 <= 1.0
    assert result.worst_accuracy == min(result.fold_accuracy)


def test_search_weighted_ensemble_prefers_better_probabilities():
    y_true = np.array([0, 1, 0, 1])
    good = np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9]])
    bad = np.array([[0.1, 0.9], [0.8, 0.2], [0.2, 0.8], [0.9, 0.1]])

    weights, score, blended = search_weighted_ensemble([good, bad], y_true, step=0.5)

    assert weights[0] > weights[1]
    assert score == 1.0
    assert blended.shape == good.shape
