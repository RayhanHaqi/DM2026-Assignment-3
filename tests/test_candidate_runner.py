import pandas as pd
import pytest
import subprocess
import sys

from scripts.run_balanced_candidates import validate_submission_frame


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
