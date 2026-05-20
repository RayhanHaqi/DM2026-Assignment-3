import pandas as pd
import pytest
import subprocess
import sys

from scripts.run_balanced_candidates import validate_submission_frame

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def test_validate_submission_frame_accepts_valid_labels():
    frame = validate_submission_frame([10, 11], [0, 5], expected_rows=2)

    assert frame.to_dict("list") == {"Id": [10, 11], "Label": [0, 5]}


def test_validate_submission_frame_rejects_bad_labels():
    with pytest.raises(ValueError, match="labels"):
        validate_submission_frame([10], [9], expected_rows=1)


def test_validate_submission_frame_rejects_wrong_row_count():
    with pytest.raises(ValueError, match="Expected 2 rows"):
        validate_submission_frame([10], [0], expected_rows=2)


def test_runner_script_help_is_directly_executable():
    result = subprocess.run(
        [sys.executable, "scripts/run_balanced_candidates.py", "--help"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Run balanced ASG3 candidates" in result.stdout


def test_daily_tree_candidate_names_are_new_macro_f1_smote():
    from scripts.run_balanced_candidates import daily_tree_candidate_names

    assert daily_tree_candidate_names() == [
        "lgb_macro_smote_refresh",
        "xgb_macro_smote_refresh",
    ]


def test_runner_script_help_includes_daily_20260520_mode():
    result = subprocess.run(
        [sys.executable, "scripts/run_balanced_candidates.py", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "--daily-20260520" in result.stdout
