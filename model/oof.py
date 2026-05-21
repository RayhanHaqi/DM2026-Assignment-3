from dataclasses import dataclass
from itertools import product

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from model.train import _apply_smote


@dataclass
class OOFResult:
    name: str
    accuracy: float
    accuracy_std: float
    worst_accuracy: float
    macro_f1: float
    fold_accuracy: list
    fold_macro_f1: list
    oof_proba: np.ndarray
    test_proba: np.ndarray
    confusion: np.ndarray
    classes: np.ndarray


def prediction_distribution(preds, labels=(0, 1, 2, 3, 4, 5)):
    values, counts = np.unique(np.asarray(preds, dtype=int), return_counts=True)
    distribution = {int(label): 0 for label in labels}
    distribution.update({int(label): int(count) for label, count in zip(values, counts)})
    return distribution


def _splitter(n_splits):
    try:
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    except TypeError:
        return GroupKFold(n_splits=n_splits)


def evaluate_oof_model(model, X, y, groups, X_test, n_splits=5, use_smote=False, name="model"):
    y_array = np.asarray(y)
    groups_array = np.asarray(groups)
    classes = np.unique(y_array)
    oof_proba = np.zeros((len(y_array), len(classes)), dtype=float)
    test_proba_folds = []
    fold_accuracy = []
    fold_macro_f1 = []
    splitter = _splitter(n_splits)

    for train_idx, val_idx in splitter.split(X, y_array, groups_array):
        X_train = X.iloc[train_idx] if hasattr(X, "iloc") else X[train_idx]
        X_val = X.iloc[val_idx] if hasattr(X, "iloc") else X[val_idx]
        y_train = y.iloc[train_idx] if hasattr(y, "iloc") else y_array[train_idx]
        y_val = y_array[val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        if use_smote:
            X_train_scaled, y_train = _apply_smote(X_train_scaled, y_train)

        fold_model = clone(model)
        fold_model.fit(X_train_scaled, y_train)
        val_proba_raw = fold_model.predict_proba(X_val_scaled)
        test_proba_raw = fold_model.predict_proba(X_test_scaled)
        fold_classes = fold_model.classes_
        val_proba = np.zeros((len(val_proba_raw), len(classes)), dtype=float)
        test_proba = np.zeros((len(test_proba_raw), len(classes)), dtype=float)
        for slot, c in enumerate(fold_classes):
            if c in classes:
                idx = np.where(classes == c)[0][0]
                val_proba[:, idx] = val_proba_raw[:, slot]
                test_proba[:, idx] = test_proba_raw[:, slot]
        oof_proba[val_idx] = val_proba
        test_proba_folds.append(test_proba)

        preds = classes[np.argmax(val_proba, axis=1)]
        fold_accuracy.append(float(accuracy_score(y_val, preds)))
        fold_macro_f1.append(float(f1_score(y_val, preds, average="macro")))

    oof_preds = classes[np.argmax(oof_proba, axis=1)]
    return OOFResult(
        name=name,
        accuracy=float(accuracy_score(y_array, oof_preds)),
        accuracy_std=float(np.std(fold_accuracy)),
        worst_accuracy=float(np.min(fold_accuracy)),
        macro_f1=float(f1_score(y_array, oof_preds, average="macro")),
        fold_accuracy=fold_accuracy,
        fold_macro_f1=fold_macro_f1,
        oof_proba=oof_proba,
        test_proba=np.mean(test_proba_folds, axis=0),
        confusion=confusion_matrix(y_array, oof_preds, labels=classes),
        classes=classes,
    )


def search_weighted_ensemble(oof_probas, y_true, step=0.1):
    y_true = np.asarray(y_true)
    n_models = len(oof_probas)
    grid = np.arange(0.0, 1.0 + step / 2, step)
    best_weights = None
    best_score = -1.0
    best_blend = None

    for raw_weights in product(grid, repeat=n_models):
        total = sum(raw_weights)
        if not np.isclose(total, 1.0):
            continue
        weights = np.asarray(raw_weights, dtype=float)
        blend = sum(weight * proba for weight, proba in zip(weights, oof_probas))
        preds = np.argmax(blend, axis=1)
        score = accuracy_score(y_true, preds)
        if score > best_score:
            best_score = float(score)
            best_weights = weights
            best_blend = blend

    return best_weights, best_score, best_blend
