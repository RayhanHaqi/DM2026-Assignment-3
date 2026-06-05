from pathlib import Path

import numpy as np
import pandas as pd

from model.hard_label_consensus import (
    hard_label_majority,
    select_public_good_submissions,
    v3_disagreement_override,
)


def _write_submission(path: Path, labels) -> None:
    frame = pd.DataFrame({"Id": np.arange(len(labels)), "Label": labels})
    frame.to_csv(path, index=False)


def test_select_public_good_dedupes_md5(tmp_path):
    labels = np.array([0, 1, 2, 3, 4, 5] * 5)
    baseline = tmp_path / "submission_tabpfn_v3_20260601_180045_01.csv"
    dup = tmp_path / "submission_tabpfn_standard_20260602_043209_01.csv"
    good = tmp_path / "submission_tabpfn_mi_top80_20260601_180550_01.csv"
    shifted = labels.copy()
    shifted[0] = 1
    _write_submission(baseline, labels)
    _write_submission(dup, labels)
    _write_submission(good, shifted)

    tracker = tmp_path / "SUBMISSIONS.md"
    tracker.write_text(
        "| File | Date | MD5 | Score | Model | Features | Notes |\n"
        f"| {baseline.name} | 20260601 | abc | **0.7830** | TabPFN | 91 | best |\n"
        f"| {dup.name} | 20260602 | abc | 0.7830 | TabPFN | 91 | dup |\n"
        f"| {good.name} | 20260601 | def | 0.7782 | TabPFN | 80 | mi |\n"
    )

    selected = select_public_good_submissions(
        tmp_path, tracker, min_good_score=0.7780, baseline_name=baseline.name
    )
    assert len(selected) == 2
    assert {item.filename for item in selected} == {baseline.name, good.name}


def test_v3_disagreement_override_changes_only_agreed_rows(tmp_path):
    v3 = np.array([0, 0, 1, 1, 2, 2])
    other_a = np.array([1, 0, 1, 2, 2, 2])
    other_b = np.array([1, 0, 1, 2, 2, 3])
    baseline = tmp_path / "submission_tabpfn_v3_20260601_180045_01.csv"
    a_path = tmp_path / "submission_tabpfn_fitmode_low_memory_20260601_113838_01.csv"
    b_path = tmp_path / "submission_tabpfn_optuna_best_20260601_173045_01.csv"
    _write_submission(baseline, v3)
    _write_submission(a_path, other_a)
    _write_submission(b_path, other_b)

    tracker = tmp_path / "SUBMISSIONS.md"
    tracker.write_text(
        "| File | Date | MD5 | Score | Model | Features | Notes |\n"
        f"| {baseline.name} | 20260601 | a1 | **0.7830** | TabPFN | 91 | |\n"
        f"| {a_path.name} | 20260601 | a2 | 0.7818 | TabPFN | 91 | |\n"
        f"| {b_path.name} | 20260601 | a3 | 0.7798 | TabPFN | 91 | |\n"
    )
    good = select_public_good_submissions(
        tmp_path, tracker, min_good_score=0.7780, baseline_name=baseline.name
    )
    preds = v3_disagreement_override(good, baseline_name=baseline.name, min_agree=2)
    assert preds[0] == 1
    assert preds[1] == 0
    assert preds[2] == 1


def test_majority_prefers_anchor_on_tie(tmp_path):
    v3 = np.array([0, 1, 2])
    other = np.array([1, 1, 2])
    baseline = tmp_path / "submission_tabpfn_v3_20260601_180045_01.csv"
    other_path = tmp_path / "submission_tabpfn_mi_top80_20260601_180550_01.csv"
    _write_submission(baseline, v3)
    _write_submission(other_path, other)

    tracker = tmp_path / "SUBMISSIONS.md"
    tracker.write_text(
        "| File | Date | MD5 | Score | Model | Features | Notes |\n"
        f"| {baseline.name} | 20260601 | a1 | **0.7830** | TabPFN | 91 | |\n"
        f"| {other_path.name} | 20260601 | a2 | 0.7782 | TabPFN | 80 | |\n"
    )
    good = select_public_good_submissions(
        tmp_path, tracker, min_good_score=0.7780, baseline_name=baseline.name
    )
    preds = hard_label_majority(good, baseline_name=baseline.name)
    assert preds[0] == 0
    assert preds[1] == 1
