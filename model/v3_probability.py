"""TabPFN V3 probability post-processing: priors, temperature, confidence."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model.validation import VALID_LABELS


@dataclass(frozen=True)
class ConfidenceStats:
    max_proba: np.ndarray
    margin: np.ndarray
    entropy: np.ndarray


def normalize_proba(proba: np.ndarray) -> np.ndarray:
    arr = np.asarray(proba, dtype=float)
    arr = np.clip(arr, 1e-12, None)
    return arr / arr.sum(axis=1, keepdims=True)


def confidence_stats(proba: np.ndarray) -> ConfidenceStats:
    arr = normalize_proba(proba)
    order = np.argsort(arr, axis=1)
    top1 = arr[np.arange(len(arr)), order[:, -1]]
    top2 = arr[np.arange(len(arr)), order[:, -2]]
    margin = top1 - top2
    entropy = -np.sum(arr * np.log(arr), axis=1)
    return ConfidenceStats(max_proba=top1, margin=margin, entropy=entropy)


def apply_temperature(proba: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if abs(temperature - 1.0) < 1e-9:
        return normalize_proba(proba)
    logits = np.log(np.clip(normalize_proba(proba), 1e-12, None))
    scaled = np.exp(logits / temperature)
    return normalize_proba(scaled)


def apply_class_priors(
    proba: np.ndarray,
    class_multipliers: np.ndarray | list[float],
) -> np.ndarray:
    mult = np.asarray(class_multipliers, dtype=float).reshape(1, -1)
    if mult.shape[1] != proba.shape[1]:
        raise ValueError(
            f"class_multipliers length {mult.shape[1]} != proba columns {proba.shape[1]}"
        )
    adjusted = normalize_proba(proba) * mult
    return normalize_proba(adjusted)


def decode_labels(proba: np.ndarray, classes: np.ndarray | None = None) -> np.ndarray:
    if classes is None:
        classes = np.asarray(VALID_LABELS, dtype=int)
    idx = np.argmax(normalize_proba(proba), axis=1)
    return classes[idx].astype(int)


def low_confidence_mask(
    proba: np.ndarray,
    *,
    max_confidence: float = 0.55,
    max_margin: float = 0.15,
) -> np.ndarray:
    stats = confidence_stats(proba)
    return (stats.max_proba < max_confidence) | (stats.margin < max_margin)


def decode_selective_calibration(
    base_preds: np.ndarray,
    base_proba: np.ndarray,
    classes: np.ndarray,
    class_multipliers: np.ndarray | list[float],
    temperature: float,
    *,
    max_confidence: float = 0.70,
    max_margin: float = 0.15,
) -> np.ndarray:
    """Recalibrate and decode only low-confidence rows; keep V3 labels elsewhere."""
    base_preds = np.asarray(base_preds, dtype=int)
    mask = low_confidence_mask(
        base_proba, max_confidence=max_confidence, max_margin=max_margin
    )
    result = base_preds.copy()
    if not mask.any():
        return result
    adjusted = apply_temperature(
        apply_class_priors(base_proba[mask], class_multipliers), temperature
    )
    result[mask] = decode_labels(adjusted, classes)
    return result


def apply_low_confidence_override(
    base_preds: np.ndarray,
    base_proba: np.ndarray,
    alt_preds: np.ndarray,
    *,
    max_confidence: float = 0.55,
    max_margin: float = 0.15,
) -> np.ndarray:
    """Replace V3 labels only on low-confidence rows using alternate hard labels."""
    base_preds = np.asarray(base_preds, dtype=int)
    alt_preds = np.asarray(alt_preds, dtype=int)
    if len(base_preds) != len(alt_preds):
        raise ValueError("base_preds and alt_preds length mismatch")
    stats = confidence_stats(base_proba)
    low = (stats.max_proba < max_confidence) | (stats.margin < max_margin)
    result = base_preds.copy()
    change = low & (alt_preds != base_preds)
    result[change] = alt_preds[change]
    return result


def default_class_multipliers() -> np.ndarray:
    return np.ones(len(VALID_LABELS), dtype=float)
