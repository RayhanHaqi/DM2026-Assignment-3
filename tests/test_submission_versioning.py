import hashlib

import pandas as pd

from model.utils import generate_submission


def test_generate_submission_uses_datetime_sequence_without_v_prefix(tmp_path):
    output_dir = tmp_path / "output"
    first = generate_submission([10, 11], [1, 2], output_dir / "submission_lgb.csv", timestamp="20260518_143000")
    second = generate_submission([10, 11], [2, 3], output_dir / "submission_ensemble.csv", timestamp="20260518_143000")

    assert first.name == "submission_lgb_20260518_143000_01.csv"
    assert second.name == "submission_ensemble_20260518_143000_02.csv"
    assert pd.read_csv(first).to_dict("list") == {"Id": [10, 11], "Label": [1, 2]}
    assert pd.read_csv(second).to_dict("list") == {"Id": [10, 11], "Label": [2, 3]}


def test_generate_submission_tracks_md5_and_metadata(tmp_path):
    output_dir = tmp_path / "output"
    path = generate_submission(
        [20],
        [4],
        output_dir / "submission_lgb.csv",
        timestamp="20260518_143000",
        model="LightGBM",
        features="42 base",
        notes="datetime test",
    )

    md5_prefix = hashlib.md5(path.read_bytes()).hexdigest()[:8]
    tracker = (output_dir / "SUBMISSIONS.md").read_text()

    assert "| File | Date | MD5 (first 8) | Kaggle Score | Model | Features | Notes |" in tracker
    assert f"| {path.name} | 20260518_143000 | {md5_prefix} | ? | LightGBM | 42 base | datetime test |" in tracker
