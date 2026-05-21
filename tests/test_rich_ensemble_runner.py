import subprocess
import sys

from scripts.run_rich_tabular_ensemble import rich_candidate_names, parse_group_list

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def test_rich_candidate_names_match_submission_plan():
    assert rich_candidate_names(include_pseudolabel=False) == [
        "xgb_rich_temporal_selected",
        "extratrees_rich_temporal",
        "oof_weighted_tabular_ensemble",
    ]
    assert rich_candidate_names(include_pseudolabel=True)[-1] == "xgb_rich_temporal_pseudolabel"


def test_parse_group_list_splits_space_separated_groups():
    assert parse_group_list("segments trend diff") == ["segments", "trend", "diff"]


def test_rich_tabular_ensemble_help_is_directly_executable():
    result = subprocess.run(
        [sys.executable, "scripts/run_rich_tabular_ensemble.py", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "Run rich tabular ensemble candidates" in result.stdout
    assert "--selected-groups" in result.stdout
    assert "--include-pseudolabel" in result.stdout
