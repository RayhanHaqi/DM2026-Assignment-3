"""Audit local submission CSVs against a baseline and SUBMISSIONS.md tracker."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from model.validation import (
    distribution_max_delta_pp,
    is_near_duplicate_predictions,
    prediction_distribution,
    prediction_shift,
)

VALID_LABELS = (0, 1, 2, 3, 4, 5)
SCORE_PATTERN = re.compile(r"(\d+\.\d+)")

# Known-failed motifs from public ablations — block even when OOF looks good.
DENYLIST_FILENAME_PATTERNS = (
    "user_norm",
    "_ft_sub_",
    "ft_sub_",
    "noise_0.",
    "xgb_blend",
    "gbdt_blend",
    "oof_stacking",
    "prob_ensemble",
    "gbdt_cache_blend",
)


@dataclass
class TrackerEntry:
    filename: str
    md5_prefix: str | None
    kaggle_score: float | None
    model: str
    notes: str


@dataclass
class SubmissionAuditRow:
    path: Path
    filename: str
    md5: str
    md5_prefix: str
    shift_pct: float
    changed: int
    total: int
    class2_count: int
    max_class_prop_delta_pp: float
    candidate_dist: dict
    tracker_score: float | None
    nearest_file: str | None
    nearest_shift_pct: float | None
    flags: list[str] = field(default_factory=list)
    block_submit: bool = False
    block_reasons: list[str] = field(default_factory=list)


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        digest.update(handle.read())
    return digest.hexdigest()


def load_labels(path: Path) -> np.ndarray:
    frame = pd.read_csv(path)
    if "Label" not in frame.columns:
        raise ValueError(f"{path} missing Label column")
    return frame["Label"].astype(int).to_numpy()


def parse_kaggle_score(cell: str) -> float | None:
    text = cell.strip()
    if not text or text in {"?", "-"}:
        return None
    match = SCORE_PATTERN.search(text.replace("*", ""))
    if not match:
        return None
    return float(match.group(1))


def parse_submissions_tracker(tracker_path: Path) -> dict[str, TrackerEntry]:
    """Parse output/SUBMISSIONS.md table rows keyed by filename."""
    entries: dict[str, TrackerEntry] = {}
    if not tracker_path.exists():
        return entries

    for line in tracker_path.read_text().splitlines():
        if not line.startswith("| submission_"):
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) < 7:
            continue
        filename, _date, md5_prefix, score_cell, model, _features, notes = parts[:7]
        md5_val = md5_prefix if md5_prefix not in {"", "-"} else None
        entries[filename] = TrackerEntry(
            filename=filename,
            md5_prefix=md5_val,
            kaggle_score=parse_kaggle_score(score_cell),
            model=model,
            notes=notes,
        )
    return entries


def class2_count(labels: np.ndarray) -> int:
    return int((labels == 2).sum())


def matches_denylist(filename: str) -> bool:
    name = filename.lower()
    return any(pattern in name for pattern in DENYLIST_FILENAME_PATTERNS)


def build_md5_index(
    submission_paths: list[Path],
    *,
    expected_len: int | None = None,
) -> dict[str, str]:
    """Map full MD5 hex digest -> filename for local submission CSVs."""
    index: dict[str, str] = {}
    for path in submission_paths:
        if not path.exists():
            continue
        try:
            labels = load_labels(path)
        except (ValueError, OSError):
            continue
        if len(labels) == 0:
            continue
        if expected_len is not None and len(labels) != expected_len:
            continue
        index[md5_file(path)] = path.name
    return index


def load_labels_index(
    submission_paths: list[Path],
    *,
    expected_len: int,
) -> dict[str, np.ndarray]:
    """Load labels for full-size submissions only (skip smoke/partial CSVs)."""
    all_labels: dict[str, np.ndarray] = {}
    for path in submission_paths:
        if not path.exists():
            continue
        try:
            labels = load_labels(path)
        except (ValueError, OSError):
            continue
        if len(labels) == expected_len:
            all_labels[path.name] = labels
    return all_labels


def class2_range_from_good_submissions(
    submission_paths: list[Path],
    min_good_score: float,
    tracker: dict[str, TrackerEntry],
) -> tuple[int, int] | None:
    counts: list[int] = []
    for path in submission_paths:
        entry = tracker.get(path.name)
        if entry is None or entry.kaggle_score is None:
            continue
        if entry.kaggle_score < min_good_score:
            continue
        if not path.exists():
            continue
        counts.append(class2_count(load_labels(path)))
    if not counts:
        return None
    return int(min(counts)), int(max(counts))


def _append_block(row: SubmissionAuditRow, reason: str, flag: str) -> None:
    row.block_submit = True
    if reason not in row.block_reasons:
        row.block_reasons.append(reason)
    if flag not in row.flags:
        row.flags.append(flag)


def audit_submission(
    candidate_path: Path,
    baseline_labels: np.ndarray,
    baseline_md5: str,
    baseline_dist: dict,
    tracker: dict[str, TrackerEntry],
    *,
    best_public_score: float,
    class2_range: tuple[int, int] | None,
    min_shift_pct: float,
    all_labels: dict[str, np.ndarray],
    max_near_dup_shift_pct: float = 0.3,
    max_shift_pct: float = 10.0,
    md5_index: dict[str, str] | None = None,
) -> SubmissionAuditRow | None:
    labels = load_labels(candidate_path)
    if len(labels) != len(baseline_labels):
        return None
    md5 = md5_file(candidate_path)
    shift = prediction_shift(labels, baseline_labels)
    cand_dist = prediction_distribution(labels)
    prop_delta = distribution_max_delta_pp(cand_dist, baseline_dist, len(baseline_labels))
    entry = tracker.get(candidate_path.name)
    tracker_score = entry.kaggle_score if entry else None

    nearest_file = None
    nearest_shift = None
    for name, other_labels in all_labels.items():
        if name == candidate_path.name:
            continue
        other_shift = prediction_shift(labels, other_labels)
        if nearest_shift is None or other_shift.percent < nearest_shift:
            nearest_shift = other_shift.percent
            nearest_file = name

    row = SubmissionAuditRow(
        path=candidate_path,
        filename=candidate_path.name,
        md5=md5,
        md5_prefix=md5[:8],
        shift_pct=shift.percent,
        changed=shift.changed,
        total=shift.total,
        class2_count=class2_count(labels),
        max_class_prop_delta_pp=prop_delta,
        candidate_dist=cand_dist,
        tracker_score=tracker_score,
        nearest_file=nearest_file,
        nearest_shift_pct=nearest_shift,
    )

    if matches_denylist(candidate_path.name):
        _append_block(
            row,
            f"filename matches denylist motif ({candidate_path.name})",
            "DENYLIST",
        )

    if md5 == baseline_md5:
        _append_block(row, "exact MD5 duplicate of baseline", "BASELINE_DUPLICATE")

    if md5_index:
        other_name = md5_index.get(md5)
        if other_name and other_name != candidate_path.name:
            _append_block(
                row,
                f"exact MD5 duplicate of {other_name}",
                "MD5_DUPLICATE",
            )

    if md5 != baseline_md5 and is_near_duplicate_predictions(
        labels, baseline_labels, max_shift_pct=max_near_dup_shift_pct
    ):
        _append_block(
            row,
            f"near-duplicate of baseline (Hamming shift {shift.percent:.2f}% < {max_near_dup_shift_pct:.2f}%)",
            "NEAR_DUPLICATE",
        )

    if shift.percent < min_shift_pct:
        _append_block(
            row,
            f"shift {shift.percent:.2f}% below minimum {min_shift_pct:.2f}%",
            "LOW_SHIFT",
        )

    if shift.percent > max_shift_pct:
        _append_block(
            row,
            f"shift {shift.percent:.2f}% above maximum {max_shift_pct:.2f}%",
            "HIGH_SHIFT",
        )

    if class2_range is not None:
        low, high = class2_range
        if row.class2_count < low or row.class2_count > high:
            row.flags.append("CLASS2_OUT_OF_RANGE")
            _append_block(
                row,
                f"class-2 count {row.class2_count} outside good-public range [{low}, {high}]",
                "CLASS2_DRIFT",
            )

    if tracker_score is not None and tracker_score < best_public_score:
        _append_block(
            row,
            f"known public {tracker_score:.4f} < best {best_public_score:.4f}",
            "KNOWN_LOSER",
        )

    collision_names = [
        other_name
        for other_name, other_entry in tracker.items()
        if other_name != candidate_path.name
        and other_entry.md5_prefix
        and other_entry.md5_prefix == row.md5_prefix
    ]
    if collision_names:
        row.flags.append("TRACKER_MD5_PREFIX_COLLISION")

    if md5 != baseline_md5:
        for other_name, other_labels in all_labels.items():
            if other_name == candidate_path.name:
                continue
            if np.array_equal(labels, other_labels):
                _append_block(
                    row,
                    f"identical predictions to {other_name}",
                    "PREDICTION_DUPLICATE",
                )
                break

    if tracker_score is None:
        row.flags.append("UNSCORED")

    return row


def audit_all_submissions(
    submission_paths: list[Path],
    baseline_path: Path,
    tracker_path: Path,
    *,
    best_public_score: float = 0.7830,
    min_good_score: float = 0.7780,
    min_shift_pct: float = 1.0,
    max_shift_pct: float = 10.0,
    max_near_dup_shift_pct: float = 0.3,
) -> tuple[list[SubmissionAuditRow], dict, tuple[int, int] | None]:
    tracker = parse_submissions_tracker(tracker_path)
    baseline_labels = load_labels(baseline_path)
    baseline_md5 = md5_file(baseline_path)
    baseline_dist = prediction_distribution(baseline_labels)

    expected_len = len(baseline_labels)
    all_labels = load_labels_index(submission_paths, expected_len=expected_len)

    class2_range = class2_range_from_good_submissions(
        submission_paths, min_good_score=min_good_score, tracker=tracker
    )
    if class2_range is None:
        class2_range = (class2_count(baseline_labels), class2_count(baseline_labels))

    md5_index = build_md5_index(submission_paths, expected_len=expected_len)

    rows: list[SubmissionAuditRow] = []
    for path in submission_paths:
        if path.resolve() == baseline_path.resolve():
            continue
        row = audit_submission(
            path,
            baseline_labels,
            baseline_md5,
            baseline_dist,
            tracker,
            best_public_score=best_public_score,
            class2_range=class2_range,
            min_shift_pct=min_shift_pct,
            all_labels=all_labels,
            max_near_dup_shift_pct=max_near_dup_shift_pct,
            max_shift_pct=max_shift_pct,
            md5_index=md5_index,
        )
        if row is not None:
            rows.append(row)

    meta = {
        "baseline": str(baseline_path),
        "baseline_md5": baseline_md5[:8],
        "baseline_dist": baseline_dist,
        "class2_range": class2_range,
        "best_public_score": best_public_score,
        "min_shift_pct": min_shift_pct,
        "max_shift_pct": max_shift_pct,
    }
    return rows, meta, class2_range


def gate_submission(
    candidate_path: Path,
    baseline_path: Path,
    tracker_path: Path,
    output_dir: Path,
    *,
    best_public_score: float = 0.7830,
    min_good_score: float = 0.7780,
    min_shift_pct: float = 1.0,
    max_shift_pct: float = 10.0,
    max_near_dup_shift_pct: float = 0.3,
) -> SubmissionAuditRow:
    """Run Phase-0 gates on one candidate CSV. Raises if row count mismatches baseline."""
    submission_paths = sorted(output_dir.glob("submission_*.csv"))
    if candidate_path not in submission_paths:
        submission_paths = sorted(set(submission_paths + [candidate_path]))
    tracker = parse_submissions_tracker(tracker_path)
    baseline_labels = load_labels(baseline_path)
    expected_len = len(baseline_labels)
    all_labels = load_labels_index(submission_paths, expected_len=expected_len)
    all_labels[candidate_path.name] = load_labels(candidate_path)
    row = audit_submission(
        candidate_path,
        baseline_labels,
        md5_file(baseline_path),
        prediction_distribution(baseline_labels),
        tracker,
        best_public_score=best_public_score,
        class2_range=resolve_class2_range(
            submission_paths, tracker_path, baseline_labels, min_good_score=min_good_score
        ),
        min_shift_pct=min_shift_pct,
        all_labels=all_labels,
        max_near_dup_shift_pct=max_near_dup_shift_pct,
        max_shift_pct=max_shift_pct,
        md5_index=build_md5_index(submission_paths, expected_len=expected_len),
    )
    if row is None:
        raise ValueError(f"{candidate_path} row count does not match baseline")
    return row


def shift_vs_reference(candidate_labels: np.ndarray, reference_labels: np.ndarray) -> float:
    return prediction_shift(candidate_labels, reference_labels).percent


def resolve_class2_range(
    submission_paths: list[Path],
    tracker_path: Path,
    baseline_labels: np.ndarray,
    *,
    min_good_score: float = 0.7780,
) -> tuple[int, int]:
    tracker = parse_submissions_tracker(tracker_path)
    class2_range = class2_range_from_good_submissions(
        submission_paths, min_good_score=min_good_score, tracker=tracker
    )
    if class2_range is not None:
        return class2_range
    c2 = class2_count(baseline_labels)
    return c2, c2
