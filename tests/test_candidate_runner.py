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


def test_daily_mode_dispatches_new_tree_candidates(monkeypatch):
    import scripts.run_balanced_candidates as runner

    calls = []

    class Args:
        tree_trials = 1
        output_dir = "output"
        no_submit = True

    def fake_lgb(name, X_train, y_train, users, X_test, test_ids, args, use_smote, metric):
        calls.append(("lgb", name, use_smote, metric))
        return {"name": name, "accuracy": 0.1, "accuracy_std": 0.0, "f1_macro": 0.2, "file": None, "scores": [0.1]}

    def fake_xgb(name, X_train, y_train, users, X_test, test_ids, args, use_smote, metric):
        calls.append(("xgb", name, use_smote, metric))
        return {"name": name, "accuracy": 0.1, "accuracy_std": 0.0, "f1_macro": 0.2, "file": None, "scores": [0.1]}

    monkeypatch.setattr(runner, "_run_lgb_candidate", fake_lgb)
    monkeypatch.setattr(runner, "_run_xgb_candidate", fake_xgb)

    results = runner._run_daily_tree_candidates(
        X_train=None,
        y_train=None,
        users=None,
        X_test=None,
        test_ids=None,
        args=Args(),
    )

    assert [row[1:] for row in calls] == [
        ("lgb_macro_smote_refresh", True, "f1_macro"),
        ("xgb_macro_smote_refresh", True, "f1_macro"),
    ]
    assert [row["name"] for row in results] == runner.daily_tree_candidate_names()
