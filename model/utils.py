import numpy as np
import pandas as pd
import os
from glob import glob

FEATURE_COLS = ["mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z"]


def _aggregate_file(df):
    rows = []
    for col in FEATURE_COLS:
        series = df[col]
        rows.append({
            f"{col}__mean": series.mean(),
            f"{col}__std": series.std(),
            f"{col}__min": series.min(),
            f"{col}__max": series.max(),
            f"{col}__q25": series.quantile(0.25),
            f"{col}__q50": series.quantile(0.50),
            f"{col}__q75": series.quantile(0.75),
            f"{col}__skew": series.skew(),
            f"{col}__kurt": series.kurtosis(),
        })
        jerk = series.diff().dropna()
        if len(jerk) > 1:
            rows.append({
                f"{col}__jerk_mean": jerk.mean(),
                f"{col}__jerk_std": jerk.std(),
                f"{col}__jerk_max": jerk.max(),
            })

    sma = np.sqrt(df["mean_x"] ** 2 + df["mean_y"] ** 2 + df["mean_z"] ** 2)
    rows.append({
        "sma__mean": sma.mean(),
        "sma__std": sma.std(),
        "sma__min": sma.min(),
        "sma__max": sma.max(),
    })

    for (a1, a2) in [("mean_x", "mean_y"), ("mean_x", "mean_z"), ("mean_y", "mean_z"),
                       ("std_x", "std_y"), ("std_x", "std_z"), ("std_y", "std_z")]:
        rows.append({f"{a1}_{a2}_corr": df[a1].corr(df[a2])})

    return pd.concat([pd.Series(r) for r in rows])


def load_train_data(base_path):
    x_list, y_list, id_list, user_list = [], [], [], []
    user_dirs = sorted(glob(os.path.join(base_path, "User_*")))
    for user_dir in user_dirs:
        user_name = os.path.basename(user_dir)
        csv_files = sorted(glob(os.path.join(user_dir, "*.csv")))
        for fpath in csv_files:
            df = pd.read_csv(fpath)
            feats = _aggregate_file(df)
            label = int(df["label"].iloc[0])
            file_id = int(df["file_id"].iloc[0])
            x_list.append(feats)
            y_list.append(label)
            id_list.append(file_id)
            user_list.append(user_name)
    X = pd.DataFrame(x_list).reset_index(drop=True)
    y = pd.Series(y_list, name="label")
    ids = pd.Series(id_list, name="file_id")
    users = pd.Series(user_list, name="user")
    return X, y, ids, users


def load_test_data(base_path):
    x_list, id_list, user_list = [], [], []
    user_dirs = sorted(glob(os.path.join(base_path, "User_*")))
    for user_dir in user_dirs:
        user_name = os.path.basename(user_dir)
        csv_files = sorted(glob(os.path.join(user_dir, "*.csv")))
        for fpath in csv_files:
            df = pd.read_csv(fpath)
            feats = _aggregate_file(df)
            file_id = int(df["file_id"].iloc[0])
            x_list.append(feats)
            id_list.append(file_id)
            user_list.append(user_name)
    X = pd.DataFrame(x_list).reset_index(drop=True)
    ids = pd.Series(id_list, name="file_id")
    users = pd.Series(user_list, name="user")
    return X, ids, users


def generate_submission(file_ids, preds, output_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    base, ext = os.path.splitext(output_path)
    version = 1
    while os.path.exists(f"{base}_v{version}{ext}"):
        version += 1
    versioned_path = f"{base}_v{version}{ext}"
    sub = pd.DataFrame({"Id": file_ids, "Label": preds})
    sub.to_csv(versioned_path, index=False)
    print(f"Saved {len(sub)} rows -> {versioned_path}")
