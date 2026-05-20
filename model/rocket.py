from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class RocketResult:
    name: str
    accuracy: float
    accuracy_std: float
    f1_macro: float
    accuracy_scores: list
    f1_scores: list


class RocketTransformer:
    def __init__(self, n_kernels=512, kernel_size=9, random_state=42):
        self.n_kernels = n_kernels
        self.kernel_size = kernel_size
        self.random_state = random_state

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float32)
        rng = np.random.default_rng(self.random_state)
        n_channels = X.shape[2]
        self.channels_ = rng.integers(0, n_channels, size=self.n_kernels)
        self.weights_ = rng.normal(size=(self.n_kernels, self.kernel_size)).astype(np.float32)
        self.weights_ -= self.weights_.mean(axis=1, keepdims=True)
        self.biases_ = rng.uniform(-1.0, 1.0, size=self.n_kernels).astype(np.float32)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        features = np.empty((X.shape[0], self.n_kernels * 2), dtype=np.float32)
        for kernel_i in range(self.n_kernels):
            channel = self.channels_[kernel_i]
            weights = self.weights_[kernel_i]
            bias = self.biases_[kernel_i]
            conv = np.stack([
                np.convolve(sample[:, channel], weights, mode="valid") + bias
                for sample in X
            ])
            features[:, kernel_i * 2] = conv.max(axis=1)
            features[:, kernel_i * 2 + 1] = (conv > 0).mean(axis=1)
        return features

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


def _build_rocket_pipeline(n_kernels, random_state):
    transformer = RocketTransformer(n_kernels=n_kernels, random_state=random_state)
    classifier = make_pipeline(StandardScaler(), RidgeClassifier(class_weight="balanced"))
    return transformer, classifier


def train_rocket_candidate(X, y, groups, n_kernels=512, n_splits=5, random_state=42):
    X = np.asarray(X, dtype=np.float32)
    y_array = np.asarray(y)
    groups_array = np.asarray(groups)
    kf = GroupKFold(n_splits=n_splits)
    accuracy_scores = []
    f1_scores = []

    for train_idx, val_idx in kf.split(X, y_array, groups_array):
        transformer, classifier = _build_rocket_pipeline(n_kernels, random_state)
        X_train = transformer.fit_transform(X[train_idx])
        X_val = transformer.transform(X[val_idx])
        classifier.fit(X_train, y_array[train_idx])
        preds = classifier.predict(X_val)
        accuracy_scores.append(float(accuracy_score(y_array[val_idx], preds)))
        f1_scores.append(float(f1_score(y_array[val_idx], preds, average="macro")))

    return RocketResult(
        name="rocket_sequence",
        accuracy=float(np.mean(accuracy_scores)),
        accuracy_std=float(np.std(accuracy_scores)),
        f1_macro=float(np.mean(f1_scores)),
        accuracy_scores=accuracy_scores,
        f1_scores=f1_scores,
    )


def fit_rocket_full(X, y, n_kernels=512, random_state=42):
    transformer, classifier = _build_rocket_pipeline(n_kernels, random_state)
    X_features = transformer.fit_transform(X)
    classifier.fit(X_features, np.asarray(y))
    return transformer, classifier


def predict_rocket(model, X_test):
    transformer, classifier = model
    X_features = transformer.transform(X_test)
    return classifier.predict(X_features)
