import numpy as np
import pandas as pd

from model.sequence import load_test_sequences, load_train_sequences


FEATURE_COLS = ["mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z"]


def _write_sequence_csv(path, file_id, label=None, rows=4):
    data = {col: np.arange(rows, dtype=float) + i for i, col in enumerate(FEATURE_COLS)}
    data["file_id"] = [file_id] * rows
    if label is not None:
        data["label"] = [label] * rows
    pd.DataFrame(data).to_csv(path, index=False)


def test_load_train_sequences_returns_arrays_and_metadata(tmp_path):
    user_dir = tmp_path / "User_001"
    user_dir.mkdir()
    _write_sequence_csv(user_dir / "a.csv", file_id=10, label=2)
    _write_sequence_csv(user_dir / "b.csv", file_id=11, label=3)

    X, y, ids, users = load_train_sequences(tmp_path)

    assert X.shape == (2, 4, 6)
    assert y.tolist() == [2, 3]
    assert ids.tolist() == [10, 11]
    assert users.tolist() == ["User_001", "User_001"]
    assert X[0, :, 0].tolist() == [0.0, 1.0, 2.0, 3.0]


def test_load_test_sequences_returns_arrays_and_metadata(tmp_path):
    user_dir = tmp_path / "User_002"
    user_dir.mkdir()
    _write_sequence_csv(user_dir / "c.csv", file_id=20)

    X, ids, users = load_test_sequences(tmp_path)

    assert X.shape == (1, 4, 6)
    assert ids.tolist() == [20]
    assert users.tolist() == ["User_002"]
