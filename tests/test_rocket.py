import numpy as np
import pandas as pd

from model.rocket import RocketTransformer, fit_rocket_full, predict_rocket, train_rocket_candidate


def test_rocket_transformer_is_deterministic_and_has_two_features_per_kernel():
    X = np.arange(3 * 12 * 2, dtype=np.float32).reshape(3, 12, 2)
    transformer_a = RocketTransformer(n_kernels=4, kernel_size=3, random_state=7)
    transformer_b = RocketTransformer(n_kernels=4, kernel_size=3, random_state=7)

    Xa = transformer_a.fit_transform(X)
    Xb = transformer_b.fit_transform(X)

    assert Xa.shape == (3, 8)
    np.testing.assert_allclose(Xa, Xb)


def test_train_rocket_candidate_returns_grouped_validation_metrics():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(12, 20, 3)).astype(np.float32)
    y = pd.Series([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    groups = pd.Series(["u1", "u1", "u2", "u2", "u3", "u3", "u4", "u4", "u5", "u5", "u6", "u6"])

    result = train_rocket_candidate(X, y, groups, n_kernels=6, n_splits=3, random_state=42)

    assert result.name == "rocket_sequence"
    assert len(result.accuracy_scores) == 3
    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.f1_macro <= 1.0


def test_fit_and_predict_rocket_returns_valid_labels():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(12, 20, 3)).astype(np.float32)
    y = pd.Series([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    X_test = rng.normal(size=(4, 20, 3)).astype(np.float32)

    model = fit_rocket_full(X, y, n_kernels=6, random_state=42)
    preds = predict_rocket(model, X_test)

    assert preds.shape == (4,)
    assert set(preds.tolist()) <= {0, 1, 2}
