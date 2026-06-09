"""Save/load TabPFN V3 probability caches for calibration sweeps."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_v3_prob_cache(
    path: Path,
    *,
    test_proba: np.ndarray,
    test_preds: np.ndarray,
    classes: np.ndarray,
    test_ids: np.ndarray,
    oof_accuracy: float,
    oof_macro_f1: float,
    fold_accuracies: np.ndarray,
    oof_proba: np.ndarray | None = None,
    config: str = "v3_seed42_n16_f1_91temporal",
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "test_proba": np.asarray(test_proba, dtype=np.float64),
        "test_preds": np.asarray(test_preds, dtype=int),
        "classes": np.asarray(classes, dtype=int),
        "test_ids": np.asarray(test_ids),
        "oof_accuracy": float(oof_accuracy),
        "oof_macro_f1": float(oof_macro_f1),
        "fold_accuracies": np.asarray(fold_accuracies, dtype=float),
        "config": config,
    }
    if oof_proba is not None:
        payload["oof_proba"] = np.asarray(oof_proba, dtype=np.float64)
    np.savez_compressed(path, **payload)
    return path


def load_v3_prob_cache(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Probability cache not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}
