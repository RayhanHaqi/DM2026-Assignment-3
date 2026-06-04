import numpy as np
import pandas as pd

from model.user_norm_features import (
    NORM_FEATURE_COLS,
    build_user_norm_features,
    fit_user_norm_stats,
)


def test_user_norm_features_shape_and_no_leakage_between_users():
    rng = np.random.RandomState(0)
    X_seq = rng.normal(size=(8, 20, 6)).astype(np.float32)
    users = pd.Series(["u1", "u1", "u1", "u1", "u2", "u2", "u2", "u2"])

    train_stats = fit_user_norm_stats(X_seq[:4], users.iloc[:4])
    feats = build_user_norm_features(X_seq, users, train_stats)

    assert list(feats.columns) == NORM_FEATURE_COLS
    assert feats.shape == (8, 12)
    assert np.isfinite(feats.to_numpy()).all()
