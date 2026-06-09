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
        n_jobs = -1

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
        n_jobs = -1

    def fake_tune(X_train, y_train, users, n_trials, metric, use_smote, n_jobs=-1):
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


def test_targeted_candidate_names_match_plan():
    from scripts.run_balanced_candidates import targeted_candidate_names

    assert targeted_candidate_names() == [
        "lgb_targeted_temporal_tuned",
        "xgb_targeted_temporal_seed_ensemble",
        "xgb_targeted_temporal_calibrated",
    ]


def test_runner_script_help_includes_targeted_mode():
    result = subprocess.run(
        [sys.executable, "scripts/run_balanced_candidates.py", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "--targeted-20260522" in result.stdout


def test_targeted_mode_smoke_is_directly_executable():
    result = subprocess.run(
        [sys.executable, "scripts/run_balanced_candidates.py", "--smoke", "--targeted-20260522"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert result.returncode == 0
    assert "lgb_targeted_temporal_tuned" in result.stdout
    assert "xgb_targeted_temporal_seed_ensemble" in result.stdout
    assert "xgb_targeted_temporal_calibrated" in result.stdout


def test_class_multiplier_search_improves_over_baseline():
    import numpy as np
    from sklearn.metrics import f1_score
    from scripts.run_balanced_candidates import _search_class_multipliers

    rng = np.random.RandomState(99)
    y_true = rng.choice([0, 1, 2], size=60)
    base_proba = np.ones((60, 3), dtype=float) * 0.1
    for i, label in enumerate(y_true):
        base_proba[i, label] = 0.5
        base_proba[i, (label + 1) % 3] = 0.3
        base_proba[i, (label + 2) % 3] = 0.1
    base_proba = base_proba / base_proba.sum(axis=1, keepdims=True)

    baseline_f1 = f1_score(y_true, base_proba.argmax(axis=1), average="macro")
    mult, calib_f1 = _search_class_multipliers(base_proba, y_true, np.array([0, 1, 2]), n_iters=2000)

    assert calib_f1 >= baseline_f1
    assert len(mult) == 3
    assert (mult > 0).all()


def test_targeted_mode_dispatches_correct_candidates(monkeypatch):
    import scripts.run_balanced_candidates as runner
    import numpy as np

    calls = []

    class Args:
        tree_trials = 1
        output_dir = "output"
        no_submit = True
        n_jobs = -1

    def fake_lgb(name, X_train, y_train, users, X_test, test_ids, args, use_smote, metric, final_fit_smote=None, features="42 base"):
        calls.append(("lgb", name))
        return {"name": name, "accuracy": 0.1, "accuracy_std": 0.0, "f1_macro": 0.2, "file": None, "scores": [0.1]}

    def fake_ensemble(name, X_train, y_train, users, X_test, test_ids, args, use_smote, metric, final_fit_smote=None, seeds=None, features="42 base"):
        calls.append(("ensemble", name))
        return {"name": name, "accuracy": 0.1, "accuracy_std": 0.0, "f1_macro": 0.2, "file": None, "scores": [0.1]}

    def fake_combine(X_base, X_seq):
        return X_base

    def fake_tune(X, y, groups, n_trials=50, metric="f1_macro", use_smote=True, n_jobs=-1):
        return {"random_state": 42}, object()

    def fake_eval(model, X, y, groups, X_test, n_splits=5, use_smote=False, name="model"):
        from model.oof import OOFResult
        return OOFResult(
            name=name, accuracy=0.5, accuracy_std=0.0, worst_accuracy=0.5,
            macro_f1=0.5, fold_accuracy=[0.5], fold_macro_f1=[0.5],
            oof_proba=np.array([[0.9, 0.1], [0.1, 0.9]]),
            test_proba=np.array([[0.9, 0.1], [0.1, 0.9]]),
            confusion=np.zeros((2, 2)), classes=np.array([0, 1]),
        )

    def fake_mult_search(oof_proba, y_true, classes, n_iters=2000):
        return np.ones(len(classes)), 0.5

    monkeypatch.setattr(runner, "_run_lgb_candidate", fake_lgb)
    monkeypatch.setattr(runner, "_run_xgb_seed_ensemble", fake_ensemble)
    monkeypatch.setattr(runner, "combine_base_and_temporal_features", fake_combine)
    monkeypatch.setattr(runner, "tune_xgboost", fake_tune)
    monkeypatch.setattr(runner, "evaluate_oof_model", fake_eval)
    monkeypatch.setattr(runner, "_search_class_multipliers", fake_mult_search)

    results = runner._run_targeted_candidates(
        X_train=pd.DataFrame({"a": [0.0, 1.0]}),
        y_train=pd.Series([0, 1]),
        users=pd.Series(["u1", "u2"]),
        X_test=pd.DataFrame({"a": [0.5, 1.5]}),
        test_ids=pd.Series([10, 11]),
        X_seq=None,
        X_test_seq=None,
        args=Args(),
    )

    assert ("lgb", "lgb_targeted_temporal_tuned") in calls
    assert ("ensemble", "xgb_targeted_temporal_seed_ensemble") in calls
    assert len(results) >= 2


def test_next_candidate_names_match_plan():
    from scripts.run_balanced_candidates import next_candidate_names

    assert next_candidate_names() == [
        "catboost_targeted_temporal_no_smote",
        "xgb_dart_targeted_temporal_no_smote",
        "sequence_feature_candidate",
    ]


def test_runner_script_help_includes_next_mode():
    result = subprocess.run(
        [sys.executable, "scripts/run_balanced_candidates.py", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "--next-20260524" in result.stdout


def test_next_mode_smoke_is_directly_executable():
    result = subprocess.run(
        [sys.executable, "scripts/run_balanced_candidates.py", "--smoke", "--next-20260524"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert result.returncode == 0
    assert "catboost_targeted_temporal_no_smote" in result.stdout
    assert "xgb_dart_targeted_temporal_no_smote" in result.stdout


def test_today_candidate_names_match_plan():
    from scripts.run_balanced_candidates import today_candidate_names

    assert today_candidate_names() == [
        "xgb_targeted_temporal_accuracy_refit_no_smote",
        "xgb_targeted_temporal_seed_ensemble_no_smote",
        "xgb_catboost_conservative_blend",
    ]


def test_runner_script_help_includes_today_mode():
    result = subprocess.run(
        [sys.executable, "scripts/run_balanced_candidates.py", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "--today-20260525" in result.stdout


def test_today_mode_dispatches_correct_candidates(monkeypatch):
    import scripts.run_balanced_candidates as runner

    calls = []

    class Args:
        tree_trials = 1
        output_dir = "output"
        no_submit = True
        n_jobs = -1
        seed = 42
        smoke = True

    def fake_combine(X_base, X_seq):
        return X_base

    def fake_xgb(name, X_train, y_train, users, X_test, test_ids, args, use_smote, metric="accuracy", final_fit_smote=None, features="42 base"):
        calls.append(("xgb", name, use_smote, metric, final_fit_smote, features))
        return {"name": name, "accuracy": 0.1, "accuracy_std": 0.0, "f1_macro": 0.2, "file": None, "scores": [0.1]}

    def fake_seed(name, X_train, y_train, users, X_test, test_ids, args, use_smote=False, metric="f1_macro", final_fit_smote=None, seeds=None, features="42 base"):
        calls.append(("seed", name, use_smote, metric, final_fit_smote, tuple(seeds), features))
        return {"name": name, "accuracy": 0.2, "accuracy_std": 0.0, "f1_macro": 0.3, "file": None, "scores": [0.2]}

    def fake_blend(name, X_train, y_train, users, X_test, test_ids, args, features="42 base + targeted temporal"):
        calls.append(("blend", name, features))
        return {"name": name, "accuracy": 0.3, "accuracy_std": 0.0, "f1_macro": 0.4, "file": None, "scores": [0.3]}

    monkeypatch.setattr(runner, "combine_base_and_temporal_features", fake_combine)
    monkeypatch.setattr(runner, "_run_xgb_candidate", fake_xgb)
    monkeypatch.setattr(runner, "_run_xgb_seed_ensemble", fake_seed)
    monkeypatch.setattr(runner, "_run_xgb_catboost_blend", fake_blend)

    results = runner._run_today_candidates(
        X_train=pd.DataFrame({"a": [0.0, 1.0]}),
        y_train=pd.Series([0, 1]),
        users=pd.Series(["u1", "u2"]),
        X_test=pd.DataFrame({"a": [0.5, 1.5]}),
        test_ids=pd.Series([10, 11]),
        X_seq=None,
        X_test_seq=None,
        args=Args(),
    )

    assert calls == [
        ("xgb", "xgb_targeted_temporal_accuracy_refit_no_smote", True, "accuracy", False, "42 base + targeted temporal"),
        ("seed", "xgb_targeted_temporal_seed_ensemble_no_smote", True, "f1_macro", False, (11, 23, 42, 71, 101), "42 base + targeted temporal"),
        ("blend", "xgb_catboost_conservative_blend", "42 base + targeted temporal"),
    ]
    assert [row["name"] for row in results] == runner.today_candidate_names()


def test_align_proba_to_classes_reorders_columns():
    import numpy as np
    from scripts.run_balanced_candidates import _align_proba_to_classes

    source_proba = np.array([[0.1, 0.7, 0.2]])
    aligned = _align_proba_to_classes(source_proba, np.array([2, 0, 1]), np.array([0, 1, 2]))

    assert aligned.tolist() == [[0.7, 0.2, 0.1]]


def test_today_mode_smoke_is_directly_executable():
    result = subprocess.run(
        [sys.executable, "scripts/run_balanced_candidates.py", "--smoke", "--today-20260525"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert result.returncode == 0
    assert "xgb_targeted_temporal_accuracy_refit_no_smote" in result.stdout
    assert "xgb_targeted_temporal_seed_ensemble_no_smote" in result.stdout
    assert "xgb_catboost_conservative_blend" in result.stdout
