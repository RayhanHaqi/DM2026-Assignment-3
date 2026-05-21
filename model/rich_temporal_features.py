import numpy as np
import pandas as pd


FEATURE_COLS = ["mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z"]
DEFAULT_GROUPS = ("segments", "trend", "diff", "magnitude", "fft")


def _stats(prefix, values):
    values = np.asarray(values, dtype=np.float32)
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": float(values.std()),
        f"{prefix}_min": float(values.min()),
        f"{prefix}_max": float(values.max()),
        f"{prefix}_q25": float(np.quantile(values, 0.25)),
        f"{prefix}_q50": float(np.quantile(values, 0.50)),
        f"{prefix}_q75": float(np.quantile(values, 0.75)),
    }


def _safe_slope(values):
    x = np.arange(len(values), dtype=np.float32)
    if len(values) < 2 or np.std(values) == 0.0:
        return 0.0
    return float(np.polyfit(x, values, 1)[0])


def _spectral_entropy(power):
    total = float(power.sum())
    if total <= 0.0:
        return 0.0
    probs = power / total
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())


def _add_segments(row, sample):
    for n_windows in (3, 5, 10):
        for win_i, window in enumerate(np.array_split(sample, n_windows)):
            for col_i, col in enumerate(FEATURE_COLS):
                row[f"segments_{n_windows}_{col}_win{win_i}_mean"] = float(window[:, col_i].mean())
                row[f"segments_{n_windows}_{col}_win{win_i}_std"] = float(window[:, col_i].std())


def _add_trend(row, sample):
    first, _, last = np.array_split(sample, 3)
    for col_i, col in enumerate(FEATURE_COLS):
        values = sample[:, col_i]
        row[f"trend_{col}_slope"] = _safe_slope(values)
        row[f"trend_{col}_last_minus_first"] = float(values[-1] - values[0])
        row[f"trend_{col}_lastwin_minus_firstwin_mean"] = float(last[:, col_i].mean() - first[:, col_i].mean())


def _add_diff(row, sample):
    for col_i, col in enumerate(FEATURE_COLS):
        diffs = np.diff(sample[:, col_i])
        if len(diffs) == 0:
            diffs = np.array([0.0], dtype=np.float32)
        row[f"diff_{col}_mean"] = float(diffs.mean())
        row[f"diff_{col}_std"] = float(diffs.std())
        row[f"diff_{col}_abs_mean"] = float(np.abs(diffs).mean())
        row[f"diff_{col}_abs_max"] = float(np.abs(diffs).max())
        row[f"diff_{col}_q25"] = float(np.quantile(diffs, 0.25))
        row[f"diff_{col}_q75"] = float(np.quantile(diffs, 0.75))


def _add_magnitude(row, sample):
    mean_mag = np.sqrt(np.sum(sample[:, :3] ** 2, axis=1))
    std_mag = np.sqrt(np.sum(sample[:, 3:] ** 2, axis=1))
    row.update(_stats("mag_mean_axes", mean_mag))
    row.update(_stats("mag_std_axes", std_mag))


def _add_fft(row, sample):
    for col_i, col in enumerate(FEATURE_COLS):
        values = sample[:, col_i] - sample[:, col_i].mean()
        power = np.abs(np.fft.rfft(values)) ** 2
        if len(power) == 0:
            power = np.array([0.0], dtype=np.float32)
        bands = np.array_split(power, 3)
        row[f"fft_{col}_low_energy"] = float(bands[0].sum())
        row[f"fft_{col}_mid_energy"] = float(bands[1].sum())
        row[f"fft_{col}_high_energy"] = float(bands[2].sum())
        row[f"fft_{col}_dominant_idx"] = int(np.argmax(power))
        row[f"fft_{col}_entropy"] = _spectral_entropy(power)


def _add_rolling(row, sample, window_size=15):
    frame = pd.DataFrame(sample, columns=FEATURE_COLS)
    rolling = frame.rolling(window=window_size, min_periods=1)
    for col in FEATURE_COLS:
        row.update(_stats(f"rolling_{col}_mean", rolling[col].mean().to_numpy()))
        row.update(_stats(f"rolling_{col}_std", rolling[col].std().fillna(0.0).to_numpy()))


def _autocorr(values, lag):
    if len(values) <= lag:
        return 0.0
    left = values[:-lag]
    right = values[lag:]
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _add_autocorr(row, sample):
    mean_mag = np.sqrt(np.sum(sample[:, :3] ** 2, axis=1))
    series = {"mean_x": sample[:, 0], "mean_y": sample[:, 1], "mean_z": sample[:, 2], "mean_mag": mean_mag}
    for name, values in series.items():
        for lag in (1, 5, 10):
            row[f"autocorr_{name}_lag{lag}"] = _autocorr(values, lag)


def build_rich_temporal_features(X_seq, groups=None):
    X_seq = np.asarray(X_seq, dtype=np.float32)
    groups = tuple(DEFAULT_GROUPS if groups is None else groups)
    rows = []
    for sample in X_seq:
        row = {}
        if "segments" in groups:
            _add_segments(row, sample)
        if "trend" in groups:
            _add_trend(row, sample)
        if "diff" in groups:
            _add_diff(row, sample)
        if "magnitude" in groups:
            _add_magnitude(row, sample)
        if "fft" in groups:
            _add_fft(row, sample)
        if "rolling" in groups:
            _add_rolling(row, sample)
        if "autocorr" in groups:
            _add_autocorr(row, sample)
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def combine_base_and_rich_features(X_base, X_seq, groups=None):
    rich = build_rich_temporal_features(X_seq, groups=groups)
    return pd.concat([X_base.reset_index(drop=True), rich], axis=1)
