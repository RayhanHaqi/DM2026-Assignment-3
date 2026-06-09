from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold


VALID_LABELS = (0, 1, 2, 3, 4, 5)


@dataclass
class SequenceOOFResult:
    name: str
    accuracy: float
    accuracy_std: float
    worst_accuracy: float
    macro_f1: float
    fold_accuracy: list
    fold_macro_f1: list
    oof_proba: np.ndarray
    test_proba: np.ndarray
    classes: np.ndarray
    prediction_distribution: dict


def prediction_distribution(preds, labels=VALID_LABELS):
    values, counts = np.unique(np.asarray(preds, dtype=int), return_counts=True)
    distribution = {int(label): 0 for label in labels}
    distribution.update({int(label): int(count) for label, count in zip(values, counts)})
    return distribution


def align_proba_to_classes(proba, source_classes, target_classes):
    aligned = np.zeros((len(proba), len(target_classes)), dtype=float)
    source_index = {int(label): i for i, label in enumerate(source_classes)}
    for target_i, label in enumerate(target_classes):
        if int(label) in source_index:
            aligned[:, target_i] = proba[:, source_index[int(label)]]
    return aligned


def blend_probabilities(base_proba, sequence_proba, sequence_weight):
    return (1.0 - sequence_weight) * base_proba + sequence_weight * sequence_proba


def search_blend_weight(base_proba, sequence_proba, y_true, classes, weights=(0.0, 0.02, 0.05, 0.08, 0.10)):
    y_true = np.asarray(y_true)
    best_weight = 0.0
    best_accuracy = -1.0
    best_preds = None
    for weight in weights:
        blended = blend_probabilities(base_proba, sequence_proba, weight)
        preds = classes[np.argmax(blended, axis=1)]
        accuracy = float(accuracy_score(y_true, preds))
        if accuracy > best_accuracy:
            best_weight = float(weight)
            best_accuracy = accuracy
            best_preds = preds
    return best_weight, best_accuracy, best_preds


def _splitter(n_splits, random_state):
    try:
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    except TypeError:
        return GroupKFold(n_splits=n_splits)


def evaluate_sequence_oof_model(
    fit_fn,
    predict_proba_fn,
    X_seq,
    y,
    groups,
    X_test_seq,
    n_splits=5,
    random_state=42,
    name="sequence_model",
):
    y_array = np.asarray(y)
    groups_array = np.asarray(groups)
    classes = np.unique(y_array).astype(int)
    oof_proba = np.zeros((len(y_array), len(classes)), dtype=float)
    test_proba_folds = []
    fold_accuracy = []
    fold_macro_f1 = []
    splitter = _splitter(n_splits=n_splits, random_state=random_state)

    for train_idx, val_idx in splitter.split(X_seq, y_array, groups_array):
        model = fit_fn(X_seq[train_idx], y_array[train_idx])
        val_raw, val_classes = predict_proba_fn(model, X_seq[val_idx])
        test_raw, test_classes = predict_proba_fn(model, X_test_seq)
        val_proba = align_proba_to_classes(val_raw, val_classes, classes)
        test_proba = align_proba_to_classes(test_raw, test_classes, classes)

        oof_proba[val_idx] = val_proba
        test_proba_folds.append(test_proba)
        preds = classes[np.argmax(val_proba, axis=1)]
        fold_accuracy.append(float(accuracy_score(y_array[val_idx], preds)))
        fold_macro_f1.append(float(f1_score(y_array[val_idx], preds, average="macro", zero_division=0)))

    oof_preds = classes[np.argmax(oof_proba, axis=1)]
    return SequenceOOFResult(
        name=name,
        accuracy=float(accuracy_score(y_array, oof_preds)),
        accuracy_std=float(np.std(fold_accuracy)),
        worst_accuracy=float(np.min(fold_accuracy)),
        macro_f1=float(f1_score(y_array, oof_preds, average="macro", zero_division=0)),
        fold_accuracy=fold_accuracy,
        fold_macro_f1=fold_macro_f1,
        oof_proba=oof_proba,
        test_proba=np.mean(test_proba_folds, axis=0),
        classes=classes,
        prediction_distribution=prediction_distribution(oof_preds),
    )
