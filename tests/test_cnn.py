import numpy as np
import torch

from model.cnn import HAR1DCNN, fit_cnn_full, predict_cnn, train_cnn_candidate


def test_har_1d_cnn_forward_shape():
    model = HAR1DCNN(n_features=6, n_classes=6)
    X = torch.randn(4, 20, 6)

    logits = model(X)

    assert logits.shape == (4, 6)


def test_train_cnn_candidate_returns_validation_metrics():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(18, 20, 6)).astype(np.float32)
    y = np.array([0, 1, 2, 3, 4, 5] * 3)
    groups = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5])

    result = train_cnn_candidate(
        X,
        y,
        groups,
        epochs=1,
        batch_size=6,
        patience=1,
        device="cpu",
        seed=42,
    )
    preds = predict_cnn(result.model, X[:3], device="cpu")

    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.f1_macro <= 1.0
    assert preds.shape == (3,)
    assert set(preds.tolist()) <= {0, 1, 2, 3, 4, 5}


def test_fit_cnn_full_returns_model_that_predicts():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(12, 20, 6)).astype(np.float32)
    y = np.array([0, 1, 2, 3, 4, 5] * 2)

    model = fit_cnn_full(X, y, epochs=1, batch_size=6, device="cpu", seed=7)
    preds = predict_cnn(model, X[:2], device="cpu")

    assert preds.shape == (2,)
    assert set(preds.tolist()) <= {0, 1, 2, 3, 4, 5}
