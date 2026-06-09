"""Diagnostics for V3-centered candidate submissions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model.validation import VALID_LABELS, prediction_distribution, prediction_shift
from model.v3_probability import confidence_stats


@dataclass(frozen=True)
class V3ShiftDiagnostic:
    shift_pct: float
    changed: int
    total: int
    changed_low_confidence_frac: float
    changed_mean_max_proba: float
    unchanged_mean_max_proba: float
    per_class_changed: dict[int, int]
    per_class_delta_pp: dict[int, float]
    candidate_dist: dict
    baseline_dist: dict


def diagnose_vs_v3(
    baseline_preds: np.ndarray,
    baseline_proba: np.ndarray,
    candidate_preds: np.ndarray,
    *,
    low_confidence_threshold: float = 0.55,
    low_margin_threshold: float = 0.15,
) -> V3ShiftDiagnostic:
    baseline_preds = np.asarray(baseline_preds, dtype=int)
    candidate_preds = np.asarray(candidate_preds, dtype=int)
    shift = prediction_shift(candidate_preds, baseline_preds)
    changed_mask = baseline_preds != candidate_preds
    stats = confidence_stats(baseline_proba)
    low = (stats.max_proba < low_confidence_threshold) | (
        stats.margin < low_margin_threshold
    )

    if changed_mask.any():
        changed_low_frac = float(low[changed_mask].mean())
        changed_mean_conf = float(stats.max_proba[changed_mask].mean())
    else:
        changed_low_frac = 0.0
        changed_mean_conf = 0.0

    unchanged_mask = ~changed_mask
    if unchanged_mask.any():
        unchanged_mean_conf = float(stats.max_proba[unchanged_mask].mean())
    else:
        unchanged_mean_conf = 0.0

    per_class_changed: dict[int, int] = {int(c): 0 for c in VALID_LABELS}
    for base_label, cand_label in zip(baseline_preds[changed_mask], candidate_preds[changed_mask]):
        per_class_changed[int(cand_label)] = per_class_changed.get(int(cand_label), 0) + 1

    baseline_dist = prediction_distribution(baseline_preds)
    candidate_dist = prediction_distribution(candidate_preds)
    total = len(baseline_preds)
    per_class_delta_pp = {
        int(label): 100.0 * (
            candidate_dist.get(int(label), 0) - baseline_dist.get(int(label), 0)
        )
        / total
        for label in VALID_LABELS
    }

    return V3ShiftDiagnostic(
        shift_pct=shift.percent,
        changed=shift.changed,
        total=shift.total,
        changed_low_confidence_frac=changed_low_frac,
        changed_mean_max_proba=changed_mean_conf,
        unchanged_mean_max_proba=unchanged_mean_conf,
        per_class_changed=per_class_changed,
        per_class_delta_pp=per_class_delta_pp,
        candidate_dist=candidate_dist,
        baseline_dist=baseline_dist,
    )


def passes_confidence_shift_gate(
    diagnostic: V3ShiftDiagnostic,
    *,
    min_changed_low_conf_frac: float = 0.70,
    max_single_class_delta_pp: float = 1.5,
    exclude_classes: tuple[int, ...] = (),
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if diagnostic.changed == 0:
        reasons.append("no label changes vs V3")
    elif diagnostic.changed_low_confidence_frac < min_changed_low_conf_frac:
        reasons.append(
            f"only {diagnostic.changed_low_confidence_frac:.1%} of changed rows are "
            f"low-confidence (need >= {min_changed_low_conf_frac:.1%})"
        )

    for label, delta_pp in diagnostic.per_class_delta_pp.items():
        if label in exclude_classes:
            continue
        if abs(delta_pp) > max_single_class_delta_pp:
            reasons.append(
                f"class {label} distribution delta {delta_pp:+.2f} pp "
                f"exceeds {max_single_class_delta_pp:.2f} pp"
            )

    return not reasons, reasons
