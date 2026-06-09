import numpy as np

from model.prob_blend import blend_two_proba, boost180_calibrated_proba, search_anchor_blend
from model.v3_probability import decode_labels


def test_blend_two_proba_rows_sum_to_one():
    anchor = np.array([[0.7, 0.2, 0.1], [0.2, 0.5, 0.3]], dtype=float)
    partner = np.array([[0.1, 0.2, 0.7], [0.6, 0.3, 0.1]], dtype=float)

    blended = blend_two_proba(anchor, partner, 0.95)

    assert blended.shape == (2, 3)
    np.testing.assert_allclose(blended.sum(axis=1), np.ones(2), atol=1e-6)


def test_search_anchor_blend_prefers_pure_anchor_on_identical_partner():
    classes = np.array([0, 1, 2])
    y = np.array([0, 1, 2, 0, 1, 2])
    anchor_oof = np.eye(3)[y]
    partner_oof = np.full((len(y), 3), 1.0 / 3.0)

    rows = search_anchor_blend(anchor_oof, partner_oof, y, classes, [0.99, 0.50])

    assert rows[0]["anchor_weight"] == 0.99
    assert rows[0]["oof_accuracy"] == 1.0


def test_boost180_calibrated_proba_changes_argmax_on_flat_rows():
    flat = np.full((2, 6), 1.0 / 6.0, dtype=float)
    flat[0, 2] = 0.22
    flat[0, 4] = 0.21
    flat[0, 5] = 0.21

    calibrated = boost180_calibrated_proba(flat)
    preds = decode_labels(calibrated, np.arange(6))

    assert preds.shape == (2,)
    assert calibrated.sum(axis=1).tolist()[0] == 1.0
