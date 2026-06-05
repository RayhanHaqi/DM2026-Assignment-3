"""Hard-label consensus from prior public-good TabPFN submissions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from model.submission_audit import (
    load_labels,
    matches_denylist,
    md5_file,
    parse_submissions_tracker,
)
from model.validation import VALID_LABELS, prediction_shift

DEFAULT_BASELINE_NAME = "submission_tabpfn_v3_20260601_180045_01.csv"


@dataclass(frozen=True)
class GoodSubmission:
    path: Path
    filename: str
    labels: np.ndarray
    md5: str
    public_score: float


def _tie_break_label(votes: np.ndarray, anchor_label: int) -> int:
    """Pick label with highest vote count; ties prefer anchor_label."""
    best_label = int(anchor_label)
    best_count = -1
    for label in VALID_LABELS:
        count = int((votes == label).sum())
        if count > best_count:
            best_count = count
            best_label = int(label)
        elif count == best_count and label == anchor_label:
            best_label = int(label)
    return best_label


def select_public_good_submissions(
    output_dir: Path,
    tracker_path: Path,
    *,
    min_good_score: float = 0.7780,
    baseline_name: str = DEFAULT_BASELINE_NAME,
) -> list[GoodSubmission]:
    """Load MD5-distinct submissions with known public score >= min_good_score."""
    tracker = parse_submissions_tracker(tracker_path)
    by_md5: dict[str, GoodSubmission] = {}

    for filename, entry in tracker.items():
        if entry.kaggle_score is None or entry.kaggle_score < min_good_score:
            continue
        if matches_denylist(filename):
            continue
        path = output_dir / filename
        if not path.exists():
            continue
        labels = load_labels(path)
        digest = md5_file(path)
        existing = by_md5.get(digest)
        if existing is None or entry.kaggle_score > existing.public_score:
            by_md5[digest] = GoodSubmission(
                path=path,
                filename=filename,
                labels=labels,
                md5=digest,
                public_score=entry.kaggle_score,
            )

    selected = sorted(by_md5.values(), key=lambda item: item.public_score, reverse=True)
    if not selected:
        raise ValueError(
            f"No local submissions with public score >= {min_good_score:.4f} in {tracker_path}"
        )

    baseline_path = output_dir / baseline_name
    if baseline_path.exists():
        baseline_md5 = md5_file(baseline_path)
        if baseline_md5 not in by_md5:
            labels = load_labels(baseline_path)
            entry = tracker.get(baseline_name)
            score = entry.kaggle_score if entry and entry.kaggle_score is not None else min_good_score
            selected.append(
                GoodSubmission(
                    path=baseline_path,
                    filename=baseline_name,
                    labels=labels,
                    md5=baseline_md5,
                    public_score=float(score),
                )
            )
            selected.sort(key=lambda item: item.public_score, reverse=True)

    lengths = {len(item.labels) for item in selected}
    if len(lengths) != 1:
        raise ValueError(f"Mismatched submission lengths: {lengths}")

    return selected


def resolve_anchor(submissions: list[GoodSubmission], baseline_name: str) -> GoodSubmission:
    for item in submissions:
        if item.filename == baseline_name:
            return item
    return submissions[0]


def hard_label_majority(
    submissions: list[GoodSubmission],
    *,
    baseline_name: str = DEFAULT_BASELINE_NAME,
) -> np.ndarray:
    """Majority vote; ties break toward the anchor (V3) label."""
    anchor = resolve_anchor(submissions, baseline_name)
    stack = np.stack([item.labels for item in submissions], axis=0)
    n_samples = stack.shape[1]
    result = np.empty(n_samples, dtype=int)
    anchor_labels = anchor.labels
    for idx in range(n_samples):
        result[idx] = _tie_break_label(stack[:, idx], int(anchor_labels[idx]))
    return result


def v3_disagreement_override(
    submissions: list[GoodSubmission],
    *,
    baseline_name: str = DEFAULT_BASELINE_NAME,
    min_agree: int = 2,
) -> np.ndarray:
    """Keep anchor labels unless min_agree non-anchor models agree on another class."""
    anchor = resolve_anchor(submissions, baseline_name)
    others = [item for item in submissions if item.filename != anchor.filename]
    if len(others) < min_agree:
        raise ValueError(
            f"Need at least {min_agree} non-anchor sources, got {len(others)}"
        )

    result = anchor.labels.copy()
    stack = np.stack([item.labels for item in others], axis=0)
    for idx in range(len(result)):
        anchor_label = int(result[idx])
        column = stack[:, idx]
        best_label = anchor_label
        best_count = 0
        for label in VALID_LABELS:
            if label == anchor_label:
                continue
            count = int((column == label).sum())
            if count >= min_agree and count > best_count:
                best_count = count
                best_label = int(label)
        result[idx] = best_label
    return result


def consensus_shift_pct(preds: np.ndarray, anchor_labels: np.ndarray) -> float:
    return prediction_shift(preds, anchor_labels).percent
