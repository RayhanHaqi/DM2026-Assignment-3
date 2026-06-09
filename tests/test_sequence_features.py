import importlib.util

import numpy as np
import pytest

from model.sequence_features import build_extra_channels, optional_import_error_message


def test_build_extra_channels_adds_magnitudes():
    X = np.ones((2, 5, 6), dtype=np.float32)
    features = build_extra_channels(X)

    assert features.shape == (2, 5, 8)
    assert np.allclose(features[:, :, 6], np.sqrt(3.0))
    assert np.allclose(features[:, :, 7], np.sqrt(3.0))


def test_optional_import_error_message_mentions_package():
    message = optional_import_error_message("aeon")

    assert "pip install aeon" in message


@pytest.mark.skipif(importlib.util.find_spec("pycatch22") is None, reason="pycatch22 not installed")
def test_build_catch22_features_shape():
    from model.sequence_features import build_catch22_features

    X = np.random.RandomState(42).normal(size=(3, 20, 6)).astype(np.float32)
    features = build_catch22_features(X)

    assert features.shape[0] == 3
    assert features.shape[1] > 0
    assert np.isfinite(features.to_numpy()).all()
