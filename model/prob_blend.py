"""Probability blending helpers for calibrated TabPFN + partner models."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from model.v3_probability import apply_class_priors, apply_temperature, decode_labels, normalize_proba

BOOST180_MULTIPLIERS = [0.80, 0.90, 1.80, 1.0, 1.80, 1.80]
BOOST180_TEMPERATURE = 0.60


def calibrated_proba(
    proba: np.ndarray,
    class_multipliers: list[float] | np.ndarray | None = None,
    temperature: float | None = None,
) -> np.ndarray:
    """Apply optional class priors and temperature to a probability matrix."""
    out = normalize_proba(proba)
    if class_multipliers is not None:
        out = apply_class_priors(out, class_multipliers)
    if temperature is not None:
        out = apply_temperature(out, temperature)
    return out


def boost180_calibrated_proba(proba: np.ndarray) -> np.ndarray:
    return calibrated_proba(proba, BOOST180_MULTIPLIERS, BOOST180_TEMPERATURE)


def blend_two_proba(anchor: np.ndarray, partner: np.ndarray, anchor_weight: float) -> np.ndarray:
    weight = float(anchor_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"anchor_weight must be in [0, 1], got {weight}")
    blended = weight * normalize_proba(anchor) + (1.0 - weight) * normalize_proba(partner)
    return normalize_proba(blended)


def search_anchor_blend(
    anchor_oof: np.ndarray,
    partner_oof: np.ndarray,
    y_true: np.ndarray,
    classes: np.ndarray,
    anchor_weights: list[float],
) -> list[dict]:
    """Return blend metrics sorted by OOF accuracy (desc), then macro F1."""
    y_true = np.asarray(y_true)
    classes = np.asarray(classes, dtype=int)
    rows = []
    for weight in anchor_weights:
        blended = blend_two_proba(anchor_oof, partner_oof, weight)
        preds = decode_labels(blended, classes)
        rows.append(
            {
                "anchor_weight": float(weight),
                "partner_weight": float(1.0 - weight),
                "oof_accuracy": float(accuracy_score(y_true, preds)),
                "oof_macro_f1": float(f1_score(y_true, preds, average="macro", zero_division=0)),
                "oof_preds": preds,
                "oof_proba": blended,
            }
        )
    rows.sort(key=lambda row: (row["oof_accuracy"], row["oof_macro_f1"]), reverse=True)
    return rows


def load_gbdt_partner_cache(path, partner: str = "xgb") -> dict:
    """Load OOF/test probabilities for one partner from gbdt_oof.npz."""
    with np.load(path, allow_pickle=False) as data:
        prefix = f"{partner}_"
        return {
            "name": partner,
            "classes": np.asarray(data["classes"], dtype=int),
            "oof_proba": np.asarray(data[f"{prefix}oof_proba"], dtype=float),
            "test_proba": np.asarray(data[f"{prefix}test_proba"], dtype=float),
        }
