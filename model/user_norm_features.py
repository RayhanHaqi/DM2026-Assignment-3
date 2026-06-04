"""Per-user z-score features from raw sequences (group-safe when fit on train slice only)."""

import numpy as np
import pandas as pd

from model.sequence import FEATURE_COLS

NORM_FEATURE_COLS = [f"unorm_{col}__mean" for col in FEATURE_COLS] + [
    f"unorm_{col}__std" for col in FEATURE_COLS
]


def fit_user_norm_stats(X_seq, users):
    """Per-user channel mean/std on training sequences only."""
    X_seq = np.asarray(X_seq, dtype=np.float64)
    users_arr = np.asarray(users)
    global_stats = _channel_stats(X_seq.reshape(-1, X_seq.shape[-1]))
    per_user: dict[str, dict] = {}

    for user in np.unique(users_arr):
        mask = users_arr == user
        samples = X_seq[mask]
        flat = samples.reshape(-1, samples.shape[-1])
        per_user[str(user)] = _channel_stats(flat)

    return {"global": global_stats, "per_user": per_user}


def _channel_stats(flat_2d):
    means = flat_2d.mean(axis=0)
    stds = flat_2d.std(axis=0)
    stds = np.where(stds < 1e-8, 1.0, stds)
    return {"mean": means, "std": stds}


def _stats_for_user(user, stats):
    user_key = str(user)
    if user_key in stats["per_user"]:
        return stats["per_user"][user_key]
    return stats["global"]


def build_user_norm_features(X_seq, users, stats):
    """12 features: mean and std of per-user z-scored channels per file."""
    X_seq = np.asarray(X_seq, dtype=np.float64)
    users_arr = np.asarray(users)
    rows = []

    for sample, user in zip(X_seq, users_arr):
        ch_stats = _stats_for_user(user, stats)
        z = (sample - ch_stats["mean"]) / ch_stats["std"]
        row = {}
        for col_i, col in enumerate(FEATURE_COLS):
            row[f"unorm_{col}__mean"] = float(z[:, col_i].mean())
            row[f"unorm_{col}__std"] = float(z[:, col_i].std())
        rows.append(row)

    return pd.DataFrame(rows, columns=NORM_FEATURE_COLS).reset_index(drop=True)
