import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from model.train import _apply_smote, cv_evaluate


def test_cv_evaluate_accuracy_returns_valid_scores():
    X = pd.DataFrame(
        {
            "a": [0.0, 0.1, 0.2, 1.0, 1.1, 1.2],
            "b": [1.0, 0.9, 0.8, 0.0, 0.1, 0.2],
        }
    )
    y = pd.Series([0, 0, 0, 1, 1, 1])
    groups = pd.Series([0, 0, 1, 1, 2, 2])
    model = LogisticRegression(random_state=42)

    scores, mean, std = cv_evaluate(model, X, y, groups, n_splits=3, metric="accuracy", use_smote=False)

    assert len(scores) == 3
    assert all(0.0 <= score <= 1.0 for score in scores)
    assert 0.0 <= mean <= 1.0
    assert std >= 0.0


def test_cv_evaluate_rejects_unknown_metric():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    groups = np.array([0, 1, 2, 3])
    model = DummyClassifier(strategy="most_frequent")

    with pytest.raises(ValueError, match="metric"):
        cv_evaluate(model, X, y, groups, n_splits=2, metric="unknown")


def test_apply_smote_skips_single_class_training_fold():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0, 0, 0])

    X_resampled, y_resampled = _apply_smote(X, y)

    assert X_resampled is X
    assert y_resampled is y
