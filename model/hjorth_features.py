import numpy as np
import pandas as pd

FEATURE_COLS = ["mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z"]


def compute_hjorth(signal_2d):
    out = {}
    for col_i, col in enumerate(FEATURE_COLS):
        sig = signal_2d[:, col_i]
        var = float(np.var(sig))
        diff = np.diff(sig)
        var_diff = float(np.var(diff))

        if var > 1e-12:
            mobility = np.sqrt(var_diff / var)
        else:
            mobility = 0.0
        out[f"mobility_{col}"] = mobility

        diff2 = np.diff(diff)
        var_diff2 = float(np.var(diff2))
        if var_diff > 1e-12 and var > 1e-12:
            mobility_diff = np.sqrt(var_diff2 / var_diff)
            complexity = mobility_diff / mobility
        else:
            complexity = 0.0
        out[f"complexity_{col}"] = complexity

    return out


def compute_spectral_features(signal_2d, fs=50.0):
    n = len(signal_2d)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    out = {}

    for col_i, col in enumerate(FEATURE_COLS):
        sig = signal_2d[:, col_i]
        mag = np.abs(np.fft.rfft(sig))
        mag_sum = mag.sum()

        if mag_sum > 1e-12:
            centroid = float(np.sum(freqs * mag) / mag_sum)
            geo_mean = float(np.exp(np.sum(np.log(mag + 1e-12)) / len(mag)))
            arith_mean = float(mag.mean())
            flatness = geo_mean / arith_mean if arith_mean > 1e-12 else 0.0

            cumsum = np.cumsum(mag)
            rolloff_idx = np.searchsorted(cumsum, 0.85 * cumsum[-1])
            rolloff = float(freqs[min(rolloff_idx, len(freqs) - 1)])

            bandwidth = float(np.sqrt(
                np.sum(((freqs - centroid) ** 2) * mag) / mag_sum
            ))
        else:
            centroid = 0.0
            flatness = 0.0
            rolloff = 0.0
            bandwidth = 0.0

        out[f"spectral_centroid_{col}"] = centroid
        out[f"spectral_flatness_{col}"] = flatness
        out[f"spectral_rolloff_{col}"] = rolloff
        out[f"spectral_bandwidth_{col}"] = bandwidth

    return out


def build_hjorth_spectral_features(X_seq):
    X_seq = np.asarray(X_seq, dtype=np.float64)
    rows = []
    for sample in X_seq:
        row = {}
        row.update(compute_hjorth(sample))
        row.update(compute_spectral_features(sample))
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)
