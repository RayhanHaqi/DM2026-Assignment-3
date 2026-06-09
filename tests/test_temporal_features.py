import numpy as np
import pandas as pd

from model.temporal_features import (
    build_spectral_window_features,
    build_targeted_temporal_features,
    combine_base_and_temporal_features,
    combine_base_temporal_spectral_features,
)


def test_build_targeted_temporal_features_has_expected_columns():
    X_seq = np.arange(2 * 6 * 6, dtype=np.float32).reshape(2, 6, 6)

    features = build_targeted_temporal_features(X_seq)

    expected_columns = {
        "mean_x__first_mean",
        "mean_x__middle_mean",
        "mean_x__last_mean",
        "mean_x__last_minus_first_mean",
        "mag__mean",
        "mag__std",
        "jerk_mean_x__mean",
        "jerk_mean_x__std",
        "corr_mean_x_mean_y",
        "energy_mean_x",
    }
    assert expected_columns <= set(features.columns)
    assert len(features) == 2
    assert features.notna().all().all()


def test_build_targeted_temporal_features_values_are_deterministic():
    X_seq = np.ones((1, 6, 6), dtype=np.float32)
    X_seq[0, :, 0] = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)

    features = build_targeted_temporal_features(X_seq)

    assert features.loc[0, "mean_x__first_mean"] == 1.5
    assert features.loc[0, "mean_x__middle_mean"] == 3.5
    assert features.loc[0, "mean_x__last_mean"] == 5.5
    assert features.loc[0, "mean_x__last_minus_first_mean"] == 4.0
    assert features.loc[0, "jerk_mean_x__mean"] == 1.0


def test_combine_base_and_temporal_features_preserves_base_columns_first():
    base = pd.DataFrame({"base_a": [1.0, 2.0], "base_b": [3.0, 4.0]})
    X_seq = np.ones((2, 6, 6), dtype=np.float32)

    combined = combine_base_and_temporal_features(base, X_seq)

    assert list(combined.columns[:2]) == ["base_a", "base_b"]
    assert len(combined) == 2
    assert combined.shape[1] > base.shape[1]


def test_build_spectral_window_features_shape_and_finite():
    X_seq = np.sin(np.linspace(0, 8 * np.pi, 2 * 60 * 6, dtype=np.float32)).reshape(2, 60, 6)

    features = build_spectral_window_features(X_seq)

    assert features.shape[0] == 2
    assert features.shape[1] == 6 * 7
    assert features.columns.tolist()[0].startswith("fft_mean_x__")
    assert np.isfinite(features.to_numpy()).all()


def test_combine_base_temporal_spectral_extends_feature_count():
    base = pd.DataFrame({"base_a": [1.0, 2.0]})
    X_seq = np.ones((2, 12, 6), dtype=np.float32)

    temporal_only = combine_base_and_temporal_features(base, X_seq)
    full = combine_base_temporal_spectral_features(base, X_seq)

    assert full.shape[0] == 2
    assert full.shape[1] > temporal_only.shape[1]
    assert list(full.columns[:1]) == ["base_a"]
