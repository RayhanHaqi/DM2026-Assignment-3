import subprocess
import sys

from scripts.run_rich_feature_ablation import ablation_feature_sets

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def test_ablation_feature_sets_include_expected_candidates():
    names = [name for name, _ in ablation_feature_sets()]
    assert names == [
        "base_42",
        "targeted_temporal",
        "segments",
        "segments_trend",
        "magnitude_diff",
        "fft",
        "rolling_autocorr",
        "all_rich",
    ]


def test_rich_feature_ablation_help_is_directly_executable():
    result = subprocess.run(
        [sys.executable, "scripts/run_rich_feature_ablation.py", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "Run rich temporal feature ablations" in result.stdout
    assert "--smoke" in result.stdout
