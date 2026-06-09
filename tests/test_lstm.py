import numpy as np
import torch

from model.cnn import SequenceNormalizer
from model.lstm import (
    HARDeepConvLSTM,
    fit_deepconv_lstm_full,
    predict_deepconv_lstm,
    predict_deepconv_lstm_proba,
    train_deepconv_lstm_candidate,
)


def test_har_deepconv_lstm_forward_shape():
    model = HARDeepConvLSTM(n_features=6, n_classes=6)
    batch = torch.randn(4, 300, 6)

    logits = model(batch)

    assert logits.shape == (4, 6)


def test_predict_deepconv_lstm_proba_is_normalized():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, 40, 6)).astype(np.float32)
    model = fit_deepconv_lstm_full(X, np.array([0, 1, 2, 3, 4]), epochs=1, batch_size=5, device="cpu", seed=0)

    proba, classes = predict_deepconv_lstm_proba(model, X[:3], device="cpu", normalize=True)

    assert proba.shape == (3, 6)
    assert classes.tolist() == [0, 1, 2, 3, 4, 5]
    assert np.isfinite(proba).all()
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(3), atol=1e-5)


def test_train_deepconv_lstm_candidate_grouped_validation():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(18, 40, 6)).astype(np.float32)
    y = np.array([0, 1, 2, 3, 4, 5] * 3)
    groups = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5])

    result = train_deepconv_lstm_candidate(
        X,
        y,
        groups,
        epochs=1,
        batch_size=6,
        patience=1,
        device="cpu",
        seed=42,
        normalize=True,
    )
    preds = predict_deepconv_lstm(result.model, X[:3], device="cpu", normalize=True)

    assert 0.0 <= result.accuracy <= 1.0
    assert preds.shape == (3,)
    assert set(preds.tolist()) <= {0, 1, 2, 3, 4, 5}


def test_sequence_normalizer_does_not_leak_test_statistics():
    train = np.array([[[1.0, 2.0]], [[3.0, 4.0]]], dtype=np.float32)
    test = np.array([[[100.0, 200.0]]], dtype=np.float32)
    normalizer = SequenceNormalizer.fit(train)
    normalized_test = normalizer.transform(test)

    assert normalized_test.shape == test.shape
    assert not np.allclose(normalized_test, test)
