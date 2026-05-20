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


def test_daily_mode_smoke_is_directly_executable():
    result = subprocess.run(
        [sys.executable, "scripts/run_balanced_candidates.py", "--smoke", "--daily-20260520"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "lgb_macro_smote_refresh" in result.stdout
    assert "xgb_macro_smote_refresh" in result.stdout
    assert "cnn_improved_sequence" in result.stdout


def test_summarize_scores_includes_worst_fold():
    from scripts.run_balanced_candidates import _summarize_scores

    summary = _summarize_scores([0.8, 0.6, 0.7])

    assert summary["worst"] == 0.6
    assert summary["mean"] == pytest.approx(0.7)
    assert summary["std"] == pytest.approx(0.081649658, rel=1e-6)


def test_prediction_distribution_counts_labels():
    from scripts.run_balanced_candidates import _prediction_distribution

    assert _prediction_distribution([0, 1, 1, 5]) == {0: 1, 1: 2, 2: 0, 3: 0, 4: 0, 5: 1}


def test_run_xgb_candidate_uses_separate_final_fit_smote(monkeypatch):
    import numpy as np
    import pandas as pd
    import scripts.run_balanced_candidates as runner

    calls = []

    class FakeXGB:
        def __init__(self, **params):
            self.params = params

    class Args:
        tree_trials = 1
        output_dir = "output"
        no_submit = True

    def fake_tune(X_train, y_train, users, n_trials, metric, use_smote):
        calls.append(("tune", use_smote, metric))
        return {"random_state": 42}, object()

    def fake_cv(model, X_train, y_train, users, metric, use_smote):
        calls.append(("cv", use_smote, metric))
        return [0.5, 0.6], 0.55, 0.05

    def fake_fit(model_cls, params, X_train, y_train, X_test, use_smote):
        calls.append(("fit", use_smote))
        return np.array([0, 1])

    monkeypatch.setattr(runner, "XGBClassifier", FakeXGB)
    monkeypatch.setattr(runner, "tune_xgboost", fake_tune)
    monkeypatch.setattr(runner, "cv_evaluate", fake_cv)
    monkeypatch.setattr(runner, "_fit_tree_model", fake_fit)

    result = runner._run_xgb_candidate(
        "xgb_final_fit_audit",
        pd.DataFrame({"a": [0.0, 1.0]}),
        pd.Series([0, 1]),
        pd.Series(["u1", "u2"]),
        pd.DataFrame({"a": [0.5, 1.5]}),
        pd.Series([10, 11]),
        Args(),
        use_smote=True,
        metric="f1_macro",
        final_fit_smote=False,
    )

    assert ("tune", True, "f1_macro") in calls
    assert ("cv", True, "accuracy") in calls
    assert ("cv", True, "f1_macro") in calls
    assert ("fit", False) in calls
    assert result["name"] == "xgb_final_fit_audit"


def test_plateau_candidate_names_match_spec():
    from scripts.run_balanced_candidates import plateau_candidate_names

    assert plateau_candidate_names() == [
        "xgb_final_fit_audit",
        "xgb_targeted_temporal",
        "rocket_sequence",
    ]


def test_runner_script_help_includes_plateau_mode():
    result = subprocess.run(
        [sys.executable, "scripts/run_balanced_candidates.py", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "--plateau-20260520" in result.stdout
    assert "--rocket-kernels" in result.stdout


def test_plateau_mode_smoke_is_directly_executable():
    result = subprocess.run(
        [sys.executable, "scripts/run_balanced_candidates.py", "--smoke", "--plateau-20260520"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "xgb_final_fit_audit" in result.stdout
    assert "xgb_targeted_temporal" in result.stdout
    assert "rocket_sequence" in result.stdout
