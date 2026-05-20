import numpy as np
import pandas as pd


FEATURE_COLS = ["mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z"]


def _safe_corr(a, b):
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def build_targeted_temporal_features(X_seq):
    """Build compact temporal features from sequences shaped (n_samples, n_steps, 6)."""
    X_seq = np.asarray(X_seq, dtype=np.float32)
    rows = []
    for sample in X_seq:
        first, middle, last = np.array_split(sample, 3)
        row = {}

        for col_i, col in enumerate(FEATURE_COLS):
            first_mean = float(first[:, col_i].mean())
            middle_mean = float(middle[:, col_i].mean())
            last_mean = float(last[:, col_i].mean())
            row[f"{col}__first_mean"] = first_mean
            row[f"{col}__middle_mean"] = middle_mean
            row[f"{col}__last_mean"] = last_mean
            row[f"{col}__last_minus_first_mean"] = last_mean - first_mean
            row[f"energy_{col}"] = float(np.mean(sample[:, col_i] ** 2))

            diffs = np.diff(sample[:, col_i])
            row[f"jerk_{col}__mean"] = float(diffs.mean()) if len(diffs) else 0.0
            row[f"jerk_{col}__std"] = float(diffs.std()) if len(diffs) else 0.0

        mean_axes = sample[:, :3]
        magnitude = np.sqrt(np.sum(mean_axes ** 2, axis=1))
        row["mag__mean"] = float(magnitude.mean())
        row["mag__std"] = float(magnitude.std())
        row["mag__min"] = float(magnitude.min())
        row["mag__max"] = float(magnitude.max())

        row["corr_mean_x_mean_y"] = _safe_corr(sample[:, 0], sample[:, 1])
        row["corr_mean_x_mean_z"] = _safe_corr(sample[:, 0], sample[:, 2])
        row["corr_mean_y_mean_z"] = _safe_corr(sample[:, 1], sample[:, 2])
        rows.append(row)

    return pd.DataFrame(rows).reset_index(drop=True)


def combine_base_and_temporal_features(X_base, X_seq):
    temporal = build_targeted_temporal_features(X_seq)
    return pd.concat([X_base.reset_index(drop=True), temporal], axis=1)
