import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from model.validation import (
    PredictionShift,
    RepeatedGroupCVResult,
    confusion_matrix_dict,
    evaluate_paired_oof_gate,
    is_near_duplicate_predictions,
    prediction_distribution,
    prediction_shift,
    repeated_group_cv,
    should_submit_candidate,
)


def test_prediction_distribution_counts_all_labels():
    assert prediction_distribution([0, 1, 1, 5]) == {0: 1, 1: 2, 2: 0, 3: 0, 4: 0, 5: 1}


def test_prediction_shift_compares_reference_predictions():
    shift = prediction_shift([0, 1, 2, 2], [0, 1, 1, 2])

    assert isinstance(shift, PredictionShift)
    assert shift.changed == 1
    assert shift.total == 4
    assert shift.percent == 25.0


def test_repeated_group_cv_returns_metrics():
    X = pd.DataFrame({"a": np.linspace(0, 1, 24), "b": np.linspace(1, 0, 24)})
    y = pd.Series([0, 1, 2, 3] * 6)
    groups = pd.Series([f"u{i // 4}" for i in range(24)])

    def factory():
        return LogisticRegression(max_iter=200, multi_class="auto")

    result = repeated_group_cv(factory, X, y, groups, n_repeats=2, n_splits=3, random_state=7)

    assert isinstance(result, RepeatedGroupCVResult)
    assert len(result.fold_accuracy) == 6
    assert 0.0 <= result.mean_accuracy <= 1.0
    assert 0.0 <= result.mean_macro_f1 <= 1.0
    assert result.worst_accuracy == min(result.fold_accuracy)
    assert set(result.prediction_distribution) == {0, 1, 2, 3, 4, 5}


def test_should_submit_candidate_rejects_accuracy_drop():
    candidate = RepeatedGroupCVResult(
        name="candidate",
        mean_accuracy=0.75,
        accuracy_std=0.02,
        worst_accuracy=0.70,
        mean_macro_f1=0.60,
        fold_accuracy=[0.75],
        fold_macro_f1=[0.60],
        prediction_distribution={0: 10, 1: 10, 2: 0, 3: 0, 4: 0, 5: 0},
    )
    baseline = RepeatedGroupCVResult(
        name="baseline",
        mean_accuracy=0.78,
        accuracy_std=0.01,
        worst_accuracy=0.74,
        mean_macro_f1=0.58,
        fold_accuracy=[0.78],
        fold_macro_f1=[0.58],
        prediction_distribution={0: 10, 1: 10, 2: 0, 3: 0, 4: 0, 5: 0},
    )

    decision = should_submit_candidate(candidate, baseline, min_accuracy_margin=-0.005)

    assert decision.accepted is False
    assert "accuracy" in decision.reason


def test_should_submit_candidate_accepts_small_accuracy_gain():
    candidate = RepeatedGroupCVResult(
        name="candidate",
        mean_accuracy=0.785,
        accuracy_std=0.02,
        worst_accuracy=0.74,
        mean_macro_f1=0.60,
        fold_accuracy=[0.785],
        fold_macro_f1=[0.60],
        prediction_distribution={0: 10, 1: 10, 2: 0, 3: 0, 4: 0, 5: 0},
    )
    baseline = RepeatedGroupCVResult(
        name="baseline",
        mean_accuracy=0.780,
        accuracy_std=0.01,
        worst_accuracy=0.74,
        mean_macro_f1=0.58,
        fold_accuracy=[0.780],
        fold_macro_f1=[0.58],
        prediction_distribution={0: 10, 1: 10, 2: 0, 3: 0, 4: 0, 5: 0},
    )

    decision = should_submit_candidate(candidate, baseline, min_accuracy_margin=-0.005)

    assert decision.accepted is True


def test_confusion_matrix_dict_shape():
    matrix = confusion_matrix_dict([0, 1, 1, 2], [0, 1, 2, 2])
    assert matrix[1][1] == 1
    assert matrix[1][2] == 1


def test_is_near_duplicate_predictions():
    baseline = np.array([0, 1, 2, 3, 4, 5] * 10)
    near = baseline.copy()
    far = baseline.copy()
    far[:50] = (far[:50] + 1) % 6
    assert is_near_duplicate_predictions(near, baseline, max_shift_pct=0.3) is True
    assert is_near_duplicate_predictions(far, baseline, max_shift_pct=0.3) is False


def test_evaluate_paired_oof_gate():
    ok = evaluate_paired_oof_gate(
        candidate_oof=0.895,
        baseline_oof=0.890,
        candidate_fold_accs=[0.88, 0.90],
        baseline_fold_accs=[0.87, 0.89],
        min_oof_margin=0.002,
    )
    assert ok.accepted is True
    assert ok.oof_delta == pytest.approx(0.005)

    bad = evaluate_paired_oof_gate(
        candidate_oof=0.891,
        baseline_oof=0.890,
        candidate_fold_accs=[0.86, 0.90],
        baseline_fold_accs=[0.87, 0.89],
        min_oof_margin=0.002,
    )
    assert bad.accepted is False
