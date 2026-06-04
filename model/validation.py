from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


VALID_LABELS = (0, 1, 2, 3, 4, 5)


@dataclass
class PredictionShift:
    changed: int
    total: int
    percent: float


@dataclass
class GateDecision:
    accepted: bool
    reason: str


@dataclass
class SubmitGateResult:
    accepted: bool
    reasons: list[str]


@dataclass
class PairedOOFGate:
    accepted: bool
    reasons: list[str]
    oof_delta: float
    worst_fold_delta: float


def required_oof_accuracy(baseline_oof: float, min_oof_margin: float) -> float:
    return baseline_oof + min_oof_margin


def effective_group_kfold_splits(groups, n_splits: int) -> int:
    """Cap n_splits so GroupKFold never exceeds the number of unique groups."""
    n_groups = len(np.unique(np.asarray(groups)))
    if n_groups < 2:
        raise ValueError(
            f"GroupKFold needs at least 2 groups, got {n_groups}. "
            "Slice smoke data by multiple users, not a flat row count."
        )
    return max(2, min(int(n_splits), n_groups))


@dataclass
class RepeatedGroupCVResult:
    name: str
    mean_accuracy: float
    accuracy_std: float
    worst_accuracy: float
    mean_macro_f1: float
    fold_accuracy: list
    fold_macro_f1: list
    prediction_distribution: dict


def prediction_distribution(preds, labels=VALID_LABELS):
    values, counts = np.unique(np.asarray(preds, dtype=int), return_counts=True)
    distribution = {int(label): 0 for label in labels}
    distribution.update({int(label): int(count) for label, count in zip(values, counts)})
    return distribution


def prediction_shift(candidate_preds, reference_preds):
    candidate = np.asarray(candidate_preds, dtype=int)
    reference = np.asarray(reference_preds, dtype=int)
    if len(candidate) != len(reference):
        raise ValueError(f"Prediction lengths differ: {len(candidate)} != {len(reference)}")
    changed = int((candidate != reference).sum())
    total = int(len(candidate))
    percent = float(changed / total * 100.0) if total else 0.0
    return PredictionShift(changed=changed, total=total, percent=percent)


def label_hamming_shift(candidate_preds, reference_preds):
    """Hamming distance on discrete labels (same as prediction_shift)."""
    return prediction_shift(candidate_preds, reference_preds)


def is_near_duplicate_predictions(
    candidate_preds,
    reference_preds,
    *,
    max_shift_pct: float = 0.3,
) -> bool:
    return label_hamming_shift(candidate_preds, reference_preds).percent < max_shift_pct


def evaluate_paired_oof_gate(
    *,
    candidate_oof: float,
    baseline_oof: float,
    candidate_fold_accs: list[float],
    baseline_fold_accs: list[float],
    min_oof_margin: float = 0.002,
) -> PairedOOFGate:
    """Paired grouped-OOF gate: same splits, accuracy-first."""
    reasons: list[str] = []
    oof_delta = float(candidate_oof - baseline_oof)
    cand_worst = float(min(candidate_fold_accs))
    base_worst = float(min(baseline_fold_accs))
    worst_fold_delta = cand_worst - base_worst

    if candidate_oof < baseline_oof + min_oof_margin:
        reasons.append(
            f"paired OOF {candidate_oof:.4f} below baseline "
            f"{baseline_oof + min_oof_margin:.4f} (delta {oof_delta:+.4f})"
        )
    if cand_worst < base_worst:
        reasons.append(
            f"paired worst-fold {cand_worst:.4f} below baseline {base_worst:.4f} "
            f"(delta {worst_fold_delta:+.4f})"
        )

    return PairedOOFGate(
        accepted=not reasons,
        reasons=reasons,
        oof_delta=oof_delta,
        worst_fold_delta=worst_fold_delta,
    )


def _splitter(n_splits, random_state):
    try:
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    except TypeError:
        return GroupKFold(n_splits=n_splits)


def _take_rows(X, idx):
    return X.iloc[idx] if hasattr(X, "iloc") else X[idx]


def smoke_slice_by_users(X, y, groups, X_seq=None, n_users: int = 6):
    """Keep rows from the first n_users (for smoke runs with valid GroupKFold)."""
    groups_s = pd.Series(groups).reset_index(drop=True)
    keep = list(groups_s.unique()[:n_users])
    mask = groups_s.isin(keep).to_numpy()
    idx = np.where(mask)[0]
    if X_seq is None:
        return _take_rows(X, idx), _take_rows(y, idx), groups_s.iloc[idx]
    return _take_rows(X, idx), _take_rows(y, idx), groups_s.iloc[idx], X_seq[idx]


def repeated_group_cv(model_factory, X, y, groups, n_repeats=3, n_splits=5, random_state=42, name="model"):
    y_array = np.asarray(y)
    groups_array = np.asarray(groups)
    fold_accuracy = []
    fold_macro_f1 = []
    all_preds = []

    for repeat_i in range(n_repeats):
        splitter = _splitter(n_splits=n_splits, random_state=random_state + repeat_i)
        for train_idx, val_idx in splitter.split(X, y_array, groups_array):
            X_train = _take_rows(X, train_idx)
            X_val = _take_rows(X, val_idx)
            y_train = y_array[train_idx]
            y_val = y_array[val_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            model = clone(model_factory())
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_val_scaled)
            all_preds.extend(preds.tolist())
            fold_accuracy.append(float(accuracy_score(y_val, preds)))
            fold_macro_f1.append(float(f1_score(y_val, preds, average="macro", zero_division=0)))

    return RepeatedGroupCVResult(
        name=name,
        mean_accuracy=float(np.mean(fold_accuracy)),
        accuracy_std=float(np.std(fold_accuracy)),
        worst_accuracy=float(np.min(fold_accuracy)),
        mean_macro_f1=float(np.mean(fold_macro_f1)),
        fold_accuracy=fold_accuracy,
        fold_macro_f1=fold_macro_f1,
        prediction_distribution=prediction_distribution(all_preds),
    )


def should_submit_candidate(candidate, baseline, min_accuracy_margin=0.0, max_accuracy_std=None):
    if candidate.mean_accuracy < baseline.mean_accuracy + min_accuracy_margin:
        return GateDecision(
            accepted=False,
            reason=f"accuracy {candidate.mean_accuracy:.4f} below required {baseline.mean_accuracy + min_accuracy_margin:.4f}",
        )
    if max_accuracy_std is not None and candidate.accuracy_std > max_accuracy_std:
        return GateDecision(
            accepted=False,
            reason=f"accuracy std {candidate.accuracy_std:.4f} above limit {max_accuracy_std:.4f}",
        )
    return GateDecision(accepted=True, reason="passed repeated grouped accuracy gate")


def mutual_info_top_k_columns(X, y, k, random_state=42):
    """Rank features by MI on the provided training slice only (leakage-safe)."""
    if k is None or k >= X.shape[1]:
        return list(X.columns) if hasattr(X, "columns") else list(range(X.shape[1]))
    mi = mutual_info_classif(np.asarray(X), np.asarray(y), random_state=random_state)
    if hasattr(X, "columns"):
        order = np.argsort(mi)[::-1][:k]
        return X.columns[order].tolist()
    order = np.argsort(mi)[::-1][:k]
    return order.tolist()


def confusion_matrix_dict(y_true, y_pred, labels=VALID_LABELS):
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    matrix = confusion_matrix(y_true_arr, y_pred_arr, labels=list(labels))
    return {int(label): {int(pred): int(matrix[i, j]) for j, pred in enumerate(labels)} for i, label in enumerate(labels)}


def distribution_max_delta_pp(candidate_dist, reference_dist, total, labels=VALID_LABELS):
    """Max absolute percentage-point gap in class proportions."""
    if total <= 0:
        return 0.0
    deltas = []
    for label in labels:
        cand_pct = 100.0 * candidate_dist.get(int(label), 0) / total
        ref_pct = 100.0 * reference_dist.get(int(label), 0) / total
        deltas.append(abs(cand_pct - ref_pct))
    return float(max(deltas))


def should_write_tabpfn_submission(
    oof_accuracy,
    baseline_oof_accuracy,
    shift,
    candidate_dist,
    baseline_dist,
    total,
    min_shift_pct=1.0,
    min_oof_margin=0.0,
    max_class_prop_delta_pp=5.0,
):
    """Gate TabPFN CSV writes: avoid near-duplicate subs and optimistic OOF-only wins."""
    if oof_accuracy < baseline_oof_accuracy + min_oof_margin:
        return GateDecision(
            accepted=False,
            reason=(
                f"OOF accuracy {oof_accuracy:.4f} below baseline "
                f"{baseline_oof_accuracy + min_oof_margin:.4f}"
            ),
        )
    prop_delta = distribution_max_delta_pp(candidate_dist, baseline_dist, total)
    if shift.percent < min_shift_pct:
        if prop_delta <= max_class_prop_delta_pp:
            return GateDecision(
                accepted=False,
                reason=(
                    f"test shift {shift.percent:.2f}% < {min_shift_pct:.2f}% "
                    f"and class distribution too similar (max delta {prop_delta:.2f} pp)"
                ),
            )
        return GateDecision(
            accepted=False,
            reason=f"test shift {shift.percent:.2f}% < {min_shift_pct:.2f}%",
        )
    if prop_delta > max_class_prop_delta_pp:
        return GateDecision(
            accepted=True,
            reason=(
                f"passed with {shift.percent:.2f}% label shift despite "
                f"class distribution delta {prop_delta:.2f} pp"
            ),
        )
    return GateDecision(accepted=True, reason="passed OOF and submission gates")


def evaluate_submit_gates(
    candidate_preds,
    baseline_preds,
    *,
    candidate_oof: float,
    baseline_oof: float,
    candidate_fold_accs: list[float],
    baseline_fold_accs: list[float],
    class2_range: tuple[int, int] | None,
    min_oof_margin: float = 0.002,
    min_shift_pct: float = 2.0,
    max_class_prop_delta_pp: float = 5.0,
) -> SubmitGateResult:
    """Pre-submit checklist: OOF margin, worst-fold, shift, optional class-2 range."""
    reasons: list[str] = []
    required_oof = required_oof_accuracy(baseline_oof, min_oof_margin)
    if candidate_oof < required_oof:
        reasons.append(
            f"OOF accuracy {candidate_oof:.4f} below required {required_oof:.4f} "
            f"(baseline {baseline_oof:.4f} + margin {min_oof_margin:.4f})"
        )

    cand_worst = float(min(candidate_fold_accs))
    base_worst = float(min(baseline_fold_accs))
    if cand_worst < base_worst:
        reasons.append(
            f"worst-fold accuracy {cand_worst:.4f} below baseline worst-fold {base_worst:.4f}"
        )

    candidate_preds = np.asarray(candidate_preds, dtype=int)
    baseline_preds = np.asarray(baseline_preds, dtype=int)
    shift = prediction_shift(candidate_preds, baseline_preds)
    cand_dist = prediction_distribution(candidate_preds)
    base_dist = prediction_distribution(baseline_preds)
    tabpfn_gate = should_write_tabpfn_submission(
        candidate_oof,
        baseline_oof,
        shift,
        cand_dist,
        base_dist,
        len(baseline_preds),
        min_shift_pct=min_shift_pct,
        min_oof_margin=0.0,
        max_class_prop_delta_pp=max_class_prop_delta_pp,
    )
    if not tabpfn_gate.accepted:
        reasons.append(tabpfn_gate.reason)

    if class2_range is not None:
        low, high = class2_range
        c2 = int((candidate_preds == 2).sum())
        if c2 < low or c2 > high:
            reasons.append(f"class-2 count {c2} outside good-public range [{low}, {high}]")

    return SubmitGateResult(accepted=not reasons, reasons=reasons)
