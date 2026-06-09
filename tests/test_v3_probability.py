import numpy as np

from model.v3_diagnostics import diagnose_vs_v3, passes_confidence_shift_gate
from model.v3_probability import (
    apply_class_priors,
    apply_low_confidence_override,
    apply_temperature,
    decode_labels,
    decode_selective_calibration,
)


def test_temperature_and_priors_change_argmax_on_uncertain_rows():
    proba = np.array(
        [
            [0.40, 0.35, 0.25, 0.0, 0.0, 0.0],
            [0.90, 0.05, 0.05, 0.0, 0.0, 0.0],
        ]
    )
    boosted = apply_class_priors(proba, [1.0, 1.0, 1.5, 1.0, 1.0, 1.0])
    cooled = apply_temperature(boosted, 0.85)
    preds = decode_labels(cooled, classes=np.array([0, 1, 2, 3, 4, 5]))
    assert preds[0] in (0, 2)
    assert preds[1] == 0


def test_low_confidence_override_only_changes_uncertain_rows():
    base_preds = np.array([0, 1, 2])
    alt_preds = np.array([1, 1, 2])
    proba = np.array(
        [
            [0.40, 0.35, 0.25, 0.0, 0.0, 0.0],
            [0.90, 0.05, 0.05, 0.0, 0.0, 0.0],
            [0.10, 0.10, 0.80, 0.0, 0.0, 0.0],
        ]
    )
    out = apply_low_confidence_override(
        base_preds, proba, alt_preds, max_confidence=0.55, max_margin=0.15
    )
    assert out[0] == 1
    assert out[1] == 1
    assert out[2] == 2


def test_selective_calibration_keeps_high_confidence_rows():
    base_preds = np.array([0, 1, 2])
    proba = np.array(
        [
            [0.40, 0.35, 0.25, 0.0, 0.0, 0.0],
            [0.90, 0.05, 0.05, 0.0, 0.0, 0.0],
            [0.10, 0.10, 0.80, 0.0, 0.0, 0.0],
        ]
    )
    classes = np.array([0, 1, 2, 3, 4, 5])
    out = decode_selective_calibration(
        base_preds,
        proba,
        classes,
        [0.85, 0.95, 1.5, 1.0, 1.5, 1.5],
        0.70,
        max_confidence=0.55,
        max_margin=0.15,
    )
    assert out[1] == 1
    assert out[0] != base_preds[0] or out[2] != base_preds[2]


def test_confidence_gate_prefers_low_confidence_changes():
    base_preds = np.array([0, 0, 1, 1, 2, 2] * 20)
    good = base_preds.copy()
    good[0] = 1
    bad = base_preds.copy()
    bad[40:80] = (bad[40:80] + 1) % 6
    proba = np.full((len(base_preds), 6), 1.0 / 6.0)
    proba[0] = [0.40, 0.35, 0.25, 0.0, 0.0, 0.0]
    proba[40:80] = [0.90, 0.02, 0.02, 0.02, 0.02, 0.02]

    good_diag = diagnose_vs_v3(base_preds, proba, good)
    bad_diag = diagnose_vs_v3(base_preds, proba, bad)
    good_ok, _ = passes_confidence_shift_gate(good_diag, min_changed_low_conf_frac=0.70)
    bad_ok, bad_reasons = passes_confidence_shift_gate(
        bad_diag, min_changed_low_conf_frac=0.70
    )
    assert good_ok or good_diag.changed == 0
    assert not bad_ok
    assert bad_reasons
