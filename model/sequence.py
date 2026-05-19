from glob import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLS = ["mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z"]


def _resolve_base_path(base_path):
    base_path = Path(base_path)
    if list(base_path.glob("User_*")):
        return base_path

    nested = base_path / base_path.name
    if nested.exists() and list(nested.glob("User_*")):
        return nested

    return base_path


def _read_sequence(fpath):
    df = pd.read_csv(fpath)
    return df[FEATURE_COLS].to_numpy(dtype=np.float32), df


def load_train_sequences(base_path):
    x_list, y_list, id_list, user_list = [], [], [], []
    base_path = _resolve_base_path(base_path)
    user_dirs = sorted(glob(os.path.join(str(base_path), "User_*")))

    for user_dir in user_dirs:
        user_name = os.path.basename(user_dir)
        csv_files = sorted(glob(os.path.join(user_dir, "*.csv")))
        for fpath in csv_files:
            sequence, df = _read_sequence(fpath)
            x_list.append(sequence)
            y_list.append(int(df["label"].iloc[0]))
            id_list.append(int(df["file_id"].iloc[0]))
            user_list.append(user_name)

    return (
        np.stack(x_list),
        pd.Series(y_list, name="label"),
        pd.Series(id_list, name="file_id"),
        pd.Series(user_list, name="user"),
    )


def load_test_sequences(base_path):
    x_list, id_list, user_list = [], [], []
    base_path = _resolve_base_path(base_path)
    user_dirs = sorted(glob(os.path.join(str(base_path), "User_*")))

    for user_dir in user_dirs:
        user_name = os.path.basename(user_dir)
        csv_files = sorted(glob(os.path.join(user_dir, "*.csv")))
        for fpath in csv_files:
            sequence, df = _read_sequence(fpath)
            x_list.append(sequence)
            id_list.append(int(df["file_id"].iloc[0]))
            user_list.append(user_name)

    return (
        np.stack(x_list),
        pd.Series(id_list, name="file_id"),
        pd.Series(user_list, name="user"),
    )
