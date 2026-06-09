#!/usr/bin/env python
"""TabPFN v3 — memory-optimized finetuning + seed search + calibration."""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.tabpfn_model import _require_tabpfn
from model.temporal_features import combine_base_and_temporal_features
from model.utils import generate_submission, load_test_data, load_train_data
from model.sequence import load_test_sequences, load_train_sequences


VALID_LABELS = {0, 1, 2, 3, 4, 5}


def _validate_frame(test_ids, preds):
    frame = pd.DataFrame({"Id": test_ids, "Label": preds})
    assert len(frame) == 6849
    assert set(frame["Label"].astype(int).tolist()) <= VALID_LABELS


def _write_sub(name, test_ids, preds, output_dir, model, features, notes):
    _validate_frame(test_ids, preds)
    return generate_submission(
        test_ids, preds,
        Path(output_dir) / f"submission_{name}.csv",
        model=model, features=features, notes=notes,
    )


def _tabpfn_oof(X_train, y_train, users, device, seed, n_estimators=16,
                eval_metric="f1", n_splits=5, **clf_kwargs):
    """Standard TabPFN OOF evaluation."""
    TabPFNClassifier = _require_tabpfn()
    y_arr = np.asarray(y_train)
    groups_arr = np.asarray(users)
    kf = GroupKFold(n_splits=n_splits)
    oof_preds = np.zeros(len(y_arr), dtype=int)

    for tr_idx, val_idx in kf.split(X_train, y_arr, groups_arr):
        X_tr = X_train.iloc[tr_idx]
        X_val = X_train.iloc[val_idx]
        y_tr = y_arr[tr_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        clf = TabPFNClassifier(
            n_estimators=n_estimators, device=device,
            random_state=seed, eval_metric=eval_metric, **clf_kwargs,
        )
        clf.fit(X_tr_s, y_tr)
        oof_preds[val_idx] = clf.predict(X_val_s)

    return float(accuracy_score(y_arr, oof_preds))


# ── Experiment 1: Seed search WITHOUT tuning_config ─────────────────────────

def experiment_seed_search(X_train, y_train, X_test, test_ids, users, args):
    print("\n=== Experiment 1: Seed search (no tuning_config, eval_metric=f1) ===")

    seeds = sorted(set(list(range(0, 100, 5)) + [42]))  # include seed 42
    best_oof = -1.0
    best_seed = 42
    results = []

    for seed in seeds:
        oof = _tabpfn_oof(
            X_train, y_train, users, args.device, seed,
            n_estimators=16, eval_metric="f1", n_splits=5,
        )
        results.append((seed, oof))
        if oof > best_oof:
            best_oof = oof
            best_seed = seed
        marker = " ← BEST" if seed == best_seed else ""
        print(f"  seed={seed:3d}  OOF={oof:.4f}{marker}")

    print(f"\n  Best seed: {best_seed} (OOF={best_oof:.4f})")
    results_dict = dict(results)
    print(f"  Seed 42 OOF: {results_dict.get(42, 'N/A'):.4f}" if 42 in results_dict else f"  Seed 42: not tested")

    # Fit with best seed
    TabPFNClassifier = _require_tabpfn()
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    clf = TabPFNClassifier(n_estimators=16, device=args.device, random_state=best_seed, eval_metric="f1")
    clf.fit(X_tr, np.asarray(y_train))
    preds = clf.predict(X_te)

    path = _write_sub(
        f"tabpfn_best_seed{best_seed}", test_ids, preds, args.output_dir,
        model="TabPFN", features="42 base + targeted temporal",
        notes=f"seed={best_seed}; OOF={best_oof:.4f}; no tuning_config; eval_metric=f1",
    )
    print(f"  File: {path}")
    return best_seed, best_oof


# ── Experiment 2: Finetuning with memory optimization ───────────────────────

def experiment_finetuning_subsampled(X_train, y_train, X_test, test_ids, users, args):
    print("\n=== Experiment 2: Finetuning (subsampled for memory) ===")

    try:
        import torch
        if not torch.cuda.is_available():
            print("  CUDA not available — skipping")
            return
        from tabpfn.finetuning.finetuned_classifier import FinetunedTabPFNClassifier
    except (ImportError, AttributeError) as e:
        print(f"  FinetunedTabPFNClassifier not available: {e}")
        return

    y_arr = np.asarray(y_train)
    groups_arr = np.asarray(users)

    # Subsample: stratified by user, keep ~3K samples
    subsample_frac = 0.3
    rng = np.random.RandomState(42)
    keep_mask = rng.random(len(y_arr)) < subsample_frac
    # Ensure all classes are represented
    for c in np.unique(y_arr):
        class_mask = y_arr == c
        if (class_mask & keep_mask).sum() < 2:
            # Force at least 2 samples per class
            class_indices = np.where(class_mask)[0]
            keep_indices = rng.choice(class_indices, size=max(2, int(class_mask.sum() * subsample_frac)), replace=False)
            keep_mask[keep_indices] = True

    X_sub = X_train.iloc[keep_mask] if hasattr(X_train, "iloc") else X_train[keep_mask]
    y_sub = y_arr[keep_mask]
    print(f"  Subsampled: {keep_mask.sum()}/{len(y_arr)} samples ({keep_mask.sum()/len(y_arr)*100:.0f}%)")

    scaler = StandardScaler()
    X_sub_s = scaler.fit_transform(X_sub)
    X_test_s = scaler.transform(X_test)

    configs = [
        {"name": "tabpfn_ft_sub_20e", "epochs": 20, "lr": 2e-5, "patience": 6},
        {"name": "tabpfn_ft_sub_30e", "epochs": 30, "lr": 2e-5, "patience": 8},
    ]

    for cfg in configs:
        name = cfg["name"]
        print(f"\n  {name} (epochs={cfg['epochs']}, lr={cfg['lr']})...")
        try:
            clf = FinetunedTabPFNClassifier(
                device="cuda",
                epochs=cfg["epochs"],
                learning_rate=cfg["lr"],
                weight_decay=0.01,
                n_estimators_finetune=1,
                n_estimators_validation=1,
                n_estimators_final_inference=4,
                early_stopping=True,
                early_stopping_patience=cfg["patience"],
                eval_metric="log_loss",
                random_state=42,
            )
            clf.fit(X_sub_s, y_sub)
            preds = clf.predict(X_test_s)

            path = _write_sub(
                name, test_ids, preds, args.output_dir,
                model="TabPFN Finetuned (subsampled)",
                features="42 base + targeted temporal",
                notes=f"subsample={subsample_frac}; epochs={cfg['epochs']}; lr={cfg['lr']}; "
                      f"n_est_ft=1/val=1/final=4; eval_metric=log_loss",
            )
            dist = {i: int((preds == i).sum()) for i in range(6)}
            print(f"    Dist: {dist}")
            print(f"    File: {path}")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"    OOM — skipping")
                import torch
                torch.cuda.empty_cache()
            else:
                print(f"    FAILED: {e}")


# ── Experiment 3: TabPFN with CalibratedClassifierCV ────────────────────────

def experiment_calibrated(X_train, y_train, X_test, test_ids, users, args):
    print("\n=== Experiment 3: TabPFN + CalibratedClassifierCV ===")

    TabPFNClassifier = _require_tabpfn()
    y_arr = np.asarray(y_train)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    base = TabPFNClassifier(
        n_estimators=16, device=args.device, random_state=42, eval_metric="f1",
    )

    # CalibratedClassifierCV with isotonic regression
    print("  Fitting CalibratedClassifierCV (isotonic, cv=5)...")
    calibrated = CalibratedClassifierCV(base, method="isotonic", cv=5)
    calibrated.fit(X_tr, y_arr)
    preds = calibrated.predict(X_te)

    path = _write_sub(
        "tabpfn_calibrated_isotonic", test_ids, preds, args.output_dir,
        model="TabPFN + CalibratedClassifierCV",
        features="42 base + targeted temporal",
        notes="isotonic calibration, cv=5, base: n_est=16, f1",
    )
    dist = {i: int((preds == i).sum()) for i in range(6)}
    print(f"  Dist: {dist}")
    print(f"  File: {path}")


# ── Experiment 4: TabPFN fit_mode variants ───────────────────────────────────

def experiment_fit_modes(X_train, y_train, X_test, test_ids, users, args):
    print("\n=== Experiment 4: fit_mode variants ===")

    TabPFNClassifier = _require_tabpfn()
    y_arr = np.asarray(y_train)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    modes = ["low_memory", "fit_preprocessors"]
    for mode in modes:
        print(f"  fit_mode={mode}...")
        clf = TabPFNClassifier(
            n_estimators=16, device=args.device, random_state=42,
            eval_metric="f1", fit_mode=mode,
        )
        clf.fit(X_tr, y_arr)
        preds = clf.predict(X_te)

        path = _write_sub(
            f"tabpfn_fitmode_{mode}", test_ids, preds, args.output_dir,
            model="TabPFN", features="42 base + targeted temporal",
            notes=f"fit_mode={mode}; n_est=16; eval_metric=f1",
        )
        dist = {i: int((preds == i).sum()) for i in range(6)}
        print(f"    Dist: {dist}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    train_path = Path(args.data_dir) / "train" / "train"
    test_path = Path(args.data_dir) / "test" / "test"

    print(f"Loading data from {train_path} and {test_path}...")
    X_train, y_train, _, users = load_train_data(str(train_path))
    X_test, test_ids, _ = load_test_data(str(test_path))
    X_seq, _, _, _ = load_train_sequences(str(train_path))
    X_test_seq, _, _ = load_test_sequences(str(test_path))

    if args.smoke:
        X_train = X_train.iloc[:60]
        y_train = y_train.iloc[:60]
        users = users.iloc[:60]
        X_test = X_test.iloc[:20]
        test_ids = test_ids.iloc[:20]
        X_seq = X_seq[:60]
        X_test_seq = X_test_seq[:20]

    X_train_t = combine_base_and_temporal_features(X_train, X_seq)
    X_test_t = combine_base_and_temporal_features(X_test, X_test_seq)
    print(f"Train: {X_train_t.shape}, Test: {X_test_t.shape}")

    # Run experiments
    experiment_seed_search(X_train_t, y_train, X_test_t, test_ids, users, args)
    experiment_finetuning_subsampled(X_train_t, y_train, X_test_t, test_ids, users, args)
    experiment_calibrated(X_train_t, y_train, X_test_t, test_ids, users, args)
    experiment_fit_modes(X_train_t, y_train, X_test_t, test_ids, users, args)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
