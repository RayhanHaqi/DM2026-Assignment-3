import numpy as np
import pandas as pd
import os
import hashlib
from glob import glob
from pathlib import Path

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
        })
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


def generate_submission(
    file_ids,
    preds,
    output_path,
    timestamp=None,
    model="?",
    features="?",
    notes="auto-generated",
):
    output_path = Path(output_path)
    output_dir = output_path.parent if str(output_path.parent) != "" else Path(".")
    output_dir.mkdir(parents=True, exist_ok=True)

    if timestamp is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    sequence = 1
    while True:
        versioned_path = output_dir / f"{output_path.stem}_{timestamp}_{sequence:02d}{output_path.suffix}"
        if not any(output_dir.glob(f"*_{timestamp}_{sequence:02d}{output_path.suffix}")):
            break
        sequence += 1

    sub = pd.DataFrame({"Id": file_ids, "Label": preds})
    sub.to_csv(versioned_path, index=False)
    print(f"Saved {len(sub)} rows -> {versioned_path}")

    tracker_path = output_dir / "SUBMISSIONS.md"
    md5_prefix = hashlib.md5(versioned_path.read_bytes()).hexdigest()[:8]
    entry = f"| {versioned_path.name} | {timestamp} | {md5_prefix} | ? | {model} | {features} | {notes} |\n"
    if not tracker_path.exists():
        tracker_path.write_text(
            "# ASG3 Submission Tracker\n\n"
            "| File | Date | MD5 (first 8) | Kaggle Score | Model | Features | Notes |\n"
            "|------|------|---------------|-------------|-------|----------|-------|\n"
        )
    with tracker_path.open("a") as f:
        f.write(entry)

    return versioned_path
