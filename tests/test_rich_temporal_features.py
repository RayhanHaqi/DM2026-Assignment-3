import numpy as np
import pandas as pd

from model.rich_temporal_features import DEFAULT_GROUPS, build_rich_temporal_features, combine_base_and_rich_features


def test_build_rich_temporal_features_has_default_groups():
    X_seq = np.arange(2 * 30 * 6, dtype=np.float32).reshape(2, 30, 6)

    features = build_rich_temporal_features(X_seq)

    columns = set(features.columns)

    assert set(DEFAULT_GROUPS) == {"segments", "trend", "diff", "magnitude", "fft"}
    assert "segments_3_mean_x_win0_mean" in columns
    assert "trend_mean_x_slope" in columns
    assert "diff_mean_x_abs_mean" in columns
    assert "mag_mean_axes_mean" in columns
    assert "fft_mean_x_low_energy" in columns
    assert len(features) == 2
    assert features.notna().all().all()


def test_build_rich_temporal_features_respects_group_selection():
    X_seq = np.ones((1, 30, 6), dtype=np.float32)

    features = build_rich_temporal_features(X_seq, groups=["magnitude"])

    assert "mag_mean_axes_mean" in features.columns
    assert all(column.startswith("mag_") for column in features.columns)


def test_rich_temporal_features_are_deterministic():
    X_seq = np.ones((1, 30, 6), dtype=np.float32)
    X_seq[0, :, 0] = np.arange(30, dtype=np.float32)

    first = build_rich_temporal_features(X_seq)
    second = build_rich_temporal_features(X_seq)

    pd.testing.assert_frame_equal(first, second)
    assert first.loc[0, "trend_mean_x_last_minus_first"] == 29.0


def test_combine_base_and_rich_features_preserves_base_columns_first():
    base = pd.DataFrame({"base_a": [1.0, 2.0], "base_b": [3.0, 4.0]})
    X_seq = np.ones((2, 30, 6), dtype=np.float32)

    combined = combine_base_and_rich_features(base, X_seq, groups=["magnitude"])

    assert list(combined.columns[:2]) == ["base_a", "base_b"]
    assert len(combined) == 2
    assert combined.shape[1] > base.shape[1]
