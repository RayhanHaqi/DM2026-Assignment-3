import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from model.submission_audit import (
    audit_submission,
    class2_count,
    matches_denylist,
    parse_kaggle_score,
    parse_submissions_tracker,
)
from model.validation import evaluate_submit_gates


def _write_submission(path: Path, labels) -> None:
    frame = pd.DataFrame({"Id": np.arange(len(labels)), "Label": labels})
    frame.to_csv(path, index=False)


def test_parse_kaggle_score():
    assert parse_kaggle_score("**0.7830**") == 0.7830
    assert parse_kaggle_score("0.7778") == 0.7778
    assert parse_kaggle_score("?") is None
    assert parse_kaggle_score("-") is None


def test_parse_submissions_tracker_reads_rows(tmp_path):
    tracker = tmp_path / "SUBMISSIONS.md"
    tracker.write_text(
        "# Tracker\n\n"
        "| File | Date | MD5 (first 8) | Kaggle Score | Model | Features | Notes |\n"
        "| submission_a.csv | 20260601 | abcdef12 | **0.7830** | TabPFN | 91 | best |\n"
        "| submission_b.csv | 20260602 | deadbeef | 0.7700 | TabPFN | 91 | loser |\n"
    )
    entries = parse_submissions_tracker(tracker)
    assert entries["submission_a.csv"].kaggle_score == 0.7830
    assert entries["submission_b.csv"].kaggle_score == 0.77
    assert entries["submission_a.csv"].md5_prefix == "abcdef12"


def test_audit_submission_blocks_known_loser_and_baseline_duplicate(tmp_path):
    labels = np.array([0, 1, 2, 3, 4, 5] * 10)
    baseline_path = tmp_path / "baseline.csv"
    loser_path = tmp_path / "submission_loser.csv"
    dup_path = tmp_path / "submission_dup.csv"
    _write_submission(baseline_path, labels)
    shifted = labels.copy()
    shifted[:30] = (shifted[:30] + 1) % 6
    _write_submission(loser_path, shifted)
    _write_submission(dup_path, labels)

    tracker_path = tmp_path / "SUBMISSIONS.md"
    loser_md5 = hashlib.md5(loser_path.read_bytes()).hexdigest()[:8]
    tracker_path.write_text(
        "| File | Date | MD5 (first 8) | Kaggle Score | Model | Features | Notes |\n"
        f"| {loser_path.name} | 20260601 | {loser_md5} | 0.7700 | TabPFN | 91 | loser |\n"
    )
    tracker = parse_submissions_tracker(tracker_path)
    baseline_labels = labels
    baseline_md5 = hashlib.md5(baseline_path.read_bytes()).hexdigest()
    baseline_dist = {i: int((labels == i).sum()) for i in range(6)}

    loser_row = audit_submission(
        loser_path,
        baseline_labels,
        baseline_md5,
        baseline_dist,
        tracker,
        best_public_score=0.7830,
        class2_range=(class2_count(labels), class2_count(labels)),
        min_shift_pct=2.0,
        all_labels={
            baseline_path.name: baseline_labels,
            loser_path.name: shifted,
            dup_path.name: labels,
        },
    )
    assert loser_row.block_submit
    assert "KNOWN_LOSER" in loser_row.flags

    dup_row = audit_submission(
        dup_path,
        baseline_labels,
        baseline_md5,
        baseline_dist,
        tracker,
        best_public_score=0.7830,
        class2_range=(class2_count(labels), class2_count(labels)),
        min_shift_pct=2.0,
        all_labels={
            baseline_path.name: baseline_labels,
            loser_path.name: shifted,
            dup_path.name: labels,
        },
    )
    assert dup_row.block_submit
    assert "BASELINE_DUPLICATE" in dup_row.flags


def test_audit_submission_blocks_near_duplicate(tmp_path):
    labels = np.array([0, 1, 2, 3, 4, 5] * 200)
    baseline_path = tmp_path / "baseline.csv"
    near_path = tmp_path / "submission_near.csv"
    _write_submission(baseline_path, labels)
    near = labels.copy()
    near[0] = (near[0] + 1) % 6
    _write_submission(near_path, near)

    tracker = {}
    baseline_labels = labels
    baseline_md5 = hashlib.md5(baseline_path.read_bytes()).hexdigest()
    baseline_dist = {i: int((labels == i).sum()) for i in range(6)}

    row = audit_submission(
        near_path,
        baseline_labels,
        baseline_md5,
        baseline_dist,
        tracker,
        best_public_score=0.7830,
        class2_range=(class2_count(labels), class2_count(labels)),
        min_shift_pct=2.0,
        all_labels={baseline_path.name: labels, near_path.name: near},
        max_near_dup_shift_pct=0.3,
    )
    assert row.block_submit
    assert "NEAR_DUPLICATE" in row.flags


def test_audit_submission_blocks_low_and_high_shift(tmp_path):
    labels = np.array([0, 1, 2, 3, 4, 5] * 200)
    baseline_path = tmp_path / "baseline.csv"
    low_path = tmp_path / "submission_low.csv"
    high_path = tmp_path / "submission_high.csv"
    _write_submission(baseline_path, labels)
    low = labels.copy()
    low[0] = (low[0] + 1) % 6
    _write_submission(low_path, low)
    high = labels.copy()
    high[:150] = (high[:150] + 1) % 6
    _write_submission(high_path, high)

    tracker = {}
    baseline_labels = labels
    baseline_md5 = hashlib.md5(baseline_path.read_bytes()).hexdigest()
    baseline_dist = {i: int((labels == i).sum()) for i in range(6)}

    low_row = audit_submission(
        low_path,
        baseline_labels,
        baseline_md5,
        baseline_dist,
        tracker,
        best_public_score=0.7830,
        class2_range=(class2_count(labels), class2_count(labels)),
        min_shift_pct=1.0,
        all_labels={baseline_path.name: labels, low_path.name: low, high_path.name: high},
        max_near_dup_shift_pct=0.3,
        max_shift_pct=10.0,
    )
    assert low_row.block_submit
    assert "LOW_SHIFT" in low_row.flags

    high_row = audit_submission(
        high_path,
        baseline_labels,
        baseline_md5,
        baseline_dist,
        tracker,
        best_public_score=0.7830,
        class2_range=(class2_count(labels), class2_count(labels)),
        min_shift_pct=1.0,
        all_labels={baseline_path.name: labels, low_path.name: low, high_path.name: high},
        max_near_dup_shift_pct=0.3,
        max_shift_pct=10.0,
    )
    assert high_row.block_submit
    assert "HIGH_SHIFT" in high_row.flags


def test_audit_submission_blocks_denylist_filename(tmp_path):
    labels = np.array([0, 1, 2, 3, 4, 5] * 50)
    baseline_path = tmp_path / "baseline.csv"
    deny_path = tmp_path / "submission_tabpfn_v3_user_norm_20260605.csv"
    _write_submission(baseline_path, labels)
    shifted = labels.copy()
    shifted[:40] = (shifted[:40] + 1) % 6
    _write_submission(deny_path, shifted)

    baseline_md5 = hashlib.md5(baseline_path.read_bytes()).hexdigest()
    baseline_dist = {i: int((labels == i).sum()) for i in range(6)}
    row = audit_submission(
        deny_path,
        labels,
        baseline_md5,
        baseline_dist,
        {},
        best_public_score=0.7830,
        class2_range=(class2_count(labels), class2_count(labels)),
        min_shift_pct=1.0,
        all_labels={baseline_path.name: labels, deny_path.name: shifted},
        max_shift_pct=10.0,
    )
    assert matches_denylist(deny_path.name)
    assert row.block_submit
    assert "DENYLIST" in row.flags


def test_audit_submission_confidence_gate_blocks_high_confidence_bulk_shift(tmp_path):
    labels = np.array([0, 1, 2, 3, 4, 5] * 50)
    baseline_path = tmp_path / "baseline.csv"
    cand_path = tmp_path / "candidate.csv"
    _write_submission(baseline_path, labels)
    shifted = labels.copy()
    shifted[10:40] = (shifted[10:40] + 1) % 6
    _write_submission(cand_path, shifted)

    proba = np.full((len(labels), 6), 1.0 / 6.0)
    proba[10:40] = [0.90, 0.02, 0.02, 0.02, 0.02, 0.02]

    baseline_md5 = hashlib.md5(baseline_path.read_bytes()).hexdigest()
    baseline_dist = {i: int((labels == i).sum()) for i in range(6)}
    row = audit_submission(
        cand_path,
        labels,
        baseline_md5,
        baseline_dist,
        {},
        best_public_score=0.7830,
        class2_range=(class2_count(labels), class2_count(labels)),
        min_shift_pct=1.0,
        all_labels={baseline_path.name: labels, cand_path.name: shifted},
        max_shift_pct=10.0,
        baseline_proba=proba,
        min_changed_low_conf_frac=0.70,
    )
    assert row.block_submit
    assert "LOW_CONFIDENCE_SHIFT" in row.flags


def test_evaluate_submit_gates_requires_oof_margin_and_shift():
    baseline_preds = np.array([0, 1, 2, 3, 4, 5] * 20)
    candidate_preds = baseline_preds.copy()
    candidate_preds[:40] = (candidate_preds[:40] + 1) % 6

    fail = evaluate_submit_gates(
        candidate_preds,
        baseline_preds,
        candidate_oof=0.891,
        baseline_oof=0.89,
        candidate_fold_accs=[0.88, 0.90, 0.91],
        baseline_fold_accs=[0.87, 0.89, 0.90],
        class2_range=(15, 25),
        min_oof_margin=0.002,
        min_shift_pct=2.0,
    )
    assert fail.accepted is False

    ok = evaluate_submit_gates(
        candidate_preds,
        baseline_preds,
        candidate_oof=0.895,
        baseline_oof=0.89,
        candidate_fold_accs=[0.88, 0.90, 0.91],
        baseline_fold_accs=[0.87, 0.89, 0.90],
        class2_range=(15, 25),
        min_oof_margin=0.002,
        min_shift_pct=2.0,
    )
    assert ok.accepted is True
