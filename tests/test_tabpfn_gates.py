import numpy as np
import pandas as pd

from model.validation import (
    GateDecision,
    distribution_max_delta_pp,
    mutual_info_top_k_columns,
    should_write_tabpfn_submission,
)
from model.validation import prediction_distribution, prediction_shift


def test_mutual_info_top_k_uses_training_slice_only():
    rng = np.random.RandomState(0)
    X = pd.DataFrame(
        {
            "signal": rng.normal(size=40),
            "noise": rng.normal(size=40),
        }
    )
    y = (X["signal"] > 0).astype(int)

    cols = mutual_info_top_k_columns(X.iloc[:30], y[:30], k=1, random_state=42)
    assert len(cols) == 1
    assert cols[0] in X.columns


def test_should_write_tabpfn_submission_rejects_low_shift_duplicate():
    baseline_preds = np.array([0, 1, 2, 3, 4, 5] * 10)
    candidate_preds = baseline_preds.copy()
    shift = prediction_shift(candidate_preds, baseline_preds)
    dist = prediction_distribution(candidate_preds)

    decision = should_write_tabpfn_submission(
        oof_accuracy=0.90,
        baseline_oof_accuracy=0.88,
        shift=shift,
        candidate_dist=dist,
        baseline_dist=dist,
        total=len(baseline_preds),
        min_shift_pct=1.0,
    )

    assert isinstance(decision, GateDecision)
    assert decision.accepted is False


def test_should_write_tabpfn_submission_accepts_shifted_candidate():
    baseline_preds = np.array([0, 1, 2, 3, 4, 5] * 100)
    candidate_preds = baseline_preds.copy()
    candidate_preds[:80] = (candidate_preds[:80] + 1) % 6
    shift = prediction_shift(candidate_preds, baseline_preds)
    base_dist = prediction_distribution(baseline_preds)
    cand_dist = prediction_distribution(candidate_preds)

    decision = should_write_tabpfn_submission(
        oof_accuracy=0.90,
        baseline_oof_accuracy=0.88,
        shift=shift,
        candidate_dist=cand_dist,
        baseline_dist=base_dist,
        total=len(baseline_preds),
        min_shift_pct=1.0,
    )

    assert decision.accepted is True


def test_distribution_max_delta_pp():
    dist_a = {0: 100, 1: 100, 2: 0, 3: 0, 4: 0, 5: 0}
    dist_b = {0: 150, 1: 50, 2: 0, 3: 0, 4: 0, 5: 0}
    delta = distribution_max_delta_pp(dist_a, dist_b, total=200)
    assert delta == 25.0
