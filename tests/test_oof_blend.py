import numpy as np

from model.oof import search_weighted_ensemble


def test_search_weighted_ensemble_weights_sum_to_one():
    rng = np.random.RandomState(0)
    y = np.array([0, 1, 0, 1, 2, 2])
    p1 = rng.dirichlet([1, 1, 1], size=len(y))
    p2 = rng.dirichlet([1, 1, 1], size=len(y))
    weights, score, blend = search_weighted_ensemble([p1, p2], y, step=0.5)
    assert weights is not None
    assert np.isclose(weights.sum(), 1.0)
    assert 0.0 <= score <= 1.0
    assert blend.shape == p1.shape
