#!/usr/bin/env python
"""TabPFN post-breakthrough stacking pipeline (0.7823 baseline)."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.tabpfn_model import fit_tabpfn_full, fit_tabpfn_oof, predict_tabpfn, predict_tabpfn_proba
from model.temporal_features import combine_base_and_temporal_features
from model.train import _apply_smote, tune_lightgbm, tune_xgboost
from model.catboost_train import predict_catboost_proba, tune_catboost
from model.utils import generate_submission, load_test_data, load_train_data
from model.sequence import load_test_sequences, load_train_sequences


VALID_LABELS = {0, 1, 2, 3, 4, 5}
N_CLASSES = 6


def _validate_frame(file_ids, preds):
    frame = pd.DataFrame({"Id": file_ids, "Label": preds})
    assert len(frame) == 6849, f"Expected 6849 rows, got {len(frame)}"
    assert not frame.isna().any().any(), "Nulls in submission"
    assert set(frame["Label"].astype(int).tolist()) <= VALID_LABELS, "Invalid labels"
    return frame


def _write_sub(name, test_ids, preds, output_dir, model, features, notes):
    _validate_frame(test_ids, preds)
    return generate_submission(
        test_ids, preds,
        Path(output_dir) / f"submission_{name}.csv",
        model=model, features=features, notes=notes,
    )


def _fit_tree_proba(model_cls, params, X_train, y_train, X_test, random_state):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    m = model_cls(**{**params, "random_state": random_state})
    m.fit(X_tr, y_train)
    return m.predict_proba(X_te), m.classes_


def _align_proba(proba, model_classes, expected_classes):
    aligned = np.zeros((len(proba), len(expected_classes)), dtype=float)
    src_idx = {int(c): i for i, c in enumerate(model_classes)}
    for j, c in enumerate(expected_classes):
        if int(c) in src_idx:
            aligned[:, j] = proba[:, src_idx[int(c)]]
    return aligned


def tabpfn_seed_ensemble(X_train, y_train, X_test, test_ids, args):
    print("\n=== TabPFN Seed Ensemble ===")
    seeds = [0, 11, 23, 42, 71]
    all_probas = []
    classes = None

    for seed in seeds:
        print(f"  Seed {seed}...")
        model, scaler = fit_tabpfn_full(
            X_train, y_train, n_estimators=16, device=args.device, random_state=seed,
        )
        proba, clz = predict_tabpfn_proba((model, scaler), X_test)
        if classes is None:
            classes = clz
        all_probas.append(proba)

    mean_proba = np.mean(all_probas, axis=0)
    preds = classes[np.argmax(mean_proba, axis=1)]

    path = _write_sub(
        "tabpfn_seed_ensemble", test_ids, preds, args.output_dir,
        model="TabPFN Seed Ensemble",
        features="42 base + targeted temporal",
        notes=f"n_estimators=16; seeds={seeds}; avg of {len(seeds)} runs",
    )
    print(f"  File: {path}")
    print(f"  Distribution: {pd.Series(preds).value_counts().sort_index().to_dict()}")
    return path


def tabpfn_fast_variants(X_train, y_train, X_test, test_ids, args):
    print("\n=== TabPFN Fast Variants ===")
    results = []

    configs = [
        ("tabpfn_n32", dict(n_estimators=32, device=args.device, random_state=args.seed),
         "n_estimators=32"),
        ("tabpfn_n8", dict(n_estimators=8, device=args.device, random_state=args.seed),
         "n_estimators=8"),
        ("tabpfn_acc", dict(n_estimators=16, device=args.device, random_state=args.seed, eval_metric="accuracy"),
         "n_estimators=16; eval_metric=accuracy"),
    ]

    for name, kwargs, notes in configs:
        print(f"  {name}...")
        try:
            model, scaler = fit_tabpfn_full(X_train, y_train, **kwargs)
        except Exception as e:
            if "eval_metric" in str(e).lower():
                print(f"    Skipping: invalid eval_metric")
                continue
            raise

        preds = predict_tabpfn((model, scaler), X_test)
        path = _write_sub(
            name, test_ids, preds, args.output_dir,
            model="TabPFN", features="42 base + targeted temporal", notes=notes,
        )
        results.append((name, path))
        print(f"    Dist: {pd.Series(preds).value_counts().sort_index().to_dict()}")

    return results


def oof_stacking(X_train, y_train, users, X_test, test_ids, args):
    print("\n=== OOF Stacking (TabPFN + XGB + CBT + LGB) ===")
    y_arr = np.asarray(y_train)
    users_arr = np.asarray(users)
    kf = GroupKFold(n_splits=5)
    folds = list(kf.split(X_train, y_arr, users_arr))
    classes_all = np.arange(N_CLASSES)

    oof_tabpfn = np.zeros((len(y_arr), N_CLASSES), dtype=float)
    oof_xgb = np.zeros((len(y_arr), N_CLASSES), dtype=float)
    oof_cbt = np.zeros((len(y_arr), N_CLASSES), dtype=float)
    oof_lgb = np.zeros((len(y_arr), N_CLASSES), dtype=float)

    test_tabpfn = np.zeros((len(X_test), N_CLASSES), dtype=float)
    test_xgb = np.zeros((len(X_test), N_CLASSES), dtype=float)
    test_cbt = np.zeros((len(X_test), N_CLASSES), dtype=float)
    test_lgb = np.zeros((len(X_test), N_CLASSES), dtype=float)

    # Tune base models first (quick)
    print("  Tuning XGBoost...")
    xgb_params, _ = tune_xgboost(
        X_train, y_train, users, n_trials=30, metric="f1_macro",
        use_smote=True, n_jobs=args.n_jobs,
    )
    print("  Tuning CatBoost...")
    cbt_params, _ = tune_catboost(
        X_train, y_train, users, n_trials=30, metric="f1_macro",
        use_smote=True, n_jobs=args.n_jobs,
    )
    print("  Tuning LightGBM...")
    lgb_params, _ = tune_lightgbm(
        X_train, y_train, users, n_trials=30, metric="f1_macro",
        use_smote=True, n_jobs=args.n_jobs,
    )

    for fold_i, (tr_idx, val_idx) in enumerate(folds):
        print(f"  Fold {fold_i + 1}/5...")
        X_tr = X_train.iloc[tr_idx]
        X_val = X_train.iloc[val_idx]
        y_tr = y_arr[tr_idx]
        y_val = y_arr[val_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)
        X_te_s = scaler.transform(X_test)

        # TabPFN
        try:
            model, _ = fit_tabpfn_full(
                pd.DataFrame(X_tr_s), y_tr,
                n_estimators=16, device=args.device, random_state=args.seed,
            )
            t_val, _ = predict_tabpfn_proba((model, scaler), X_val_s)
            t_test, _ = predict_tabpfn_proba((model, scaler), X_te_s)
            oof_tabpfn[val_idx] = t_val
            test_tabpfn += t_test / len(folds)
        except Exception as e:
            print(f"    TabPFN fold {fold_i} failed: {e}")

        # XGBoost
        X_tr_s_smt, y_tr_smt = _apply_smote(X_tr_s.copy(), y_tr) if True else (X_tr_s, y_tr)
        m = XGBClassifier(**{**xgb_params, "random_state": args.seed})
        m.fit(X_tr_s_smt, y_tr_smt)
        oof_xgb[val_idx] = m.predict_proba(X_val_s)
        test_xgb += m.predict_proba(X_te_s) / len(folds)

        # CatBoost
        from catboost import CatBoostClassifier
        cbt_kw = {k: v for k, v in cbt_params.items()
                  if k not in ("loss_function", "eval_metric", "random_seed", "thread_count", "verbose", "allow_writing_files", "random_strength", "bagging_temperature")}
        cbt_m = CatBoostClassifier(
            loss_function="MultiClass", eval_metric="Accuracy",
            iterations=cbt_kw.pop("iterations", 500),
            depth=cbt_kw.pop("depth", 6),
            learning_rate=cbt_kw.pop("learning_rate", 0.05),
            l2_leaf_reg=cbt_kw.pop("l2_leaf_reg", 3.0),
            random_seed=args.seed, verbose=False, allow_writing_files=False,
        )
        try:
            cbt_m.fit(X_tr_s_smt, y_tr_smt)
            cbt_val_raw = cbt_m.predict_proba(X_val_s)
            cbt_test_raw = cbt_m.predict_proba(X_te_s)
            cbt_cls = cbt_m.classes_
            oof_cbt[val_idx] = _align_proba(cbt_val_raw, cbt_cls, classes_all)
            test_cbt += _align_proba(cbt_test_raw, cbt_cls, classes_all) / len(folds)
        except Exception as e:
            print(f"    CatBoost fold {fold_i} failed: {e}")

        # LightGBM
        lgb_kw = {k: v for k, v in lgb_params.items()
                  if k not in ("class_weight", "verbose")}
        lgb_m = LGBMClassifier(**{**lgb_kw, "random_state": args.seed, "verbose": -1})
        lgb_m.fit(X_tr_s_smt, y_tr_smt)
        lgb_val_raw = lgb_m.predict_proba(X_val_s)
        lgb_test_raw = lgb_m.predict_proba(X_te_s)
        lgb_cls = lgb_m.classes_
        oof_lgb[val_idx] = _align_proba(lgb_val_raw, lgb_cls, classes_all)
        test_lgb += _align_proba(lgb_test_raw, lgb_cls, classes_all) / len(folds)

    oof_preds = classes_all[np.argmax(oof_tabpfn, axis=1)]
    oof_acc = float(accuracy_score(y_arr, oof_preds))
    oof_f1 = float(f1_score(y_arr, oof_preds, average="macro"))
    print(f"  TabPFN OOF accuracy: {oof_acc:.4f}, OOF F1: {oof_f1:.4f}")

    # Build stacking features
    X_stack_tr = np.hstack([
        np.asarray(X_train),
        oof_tabpfn, oof_xgb, oof_cbt, oof_lgb,
    ])
    X_stack_te = np.hstack([
        np.asarray(X_test),
        test_tabpfn, test_xgb, test_cbt, test_lgb,
    ])
    print(f"  Stacking features: {X_stack_tr.shape[1]} dims (91 + {4 * N_CLASSES} OOF)")

    # Meta-learner
    print("  Tuning meta-learner XGBoost...")
    import optuna

    def meta_obj(trial):
        p = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            "random_state": args.seed, "n_jobs": args.n_jobs,
        }
        scores = []
        for tr_idx, val_idx in folds:
            X_tr = X_stack_tr[tr_idx]
            X_val = X_stack_tr[val_idx]
            y_tr = y_arr[tr_idx]
            y_val = y_arr[val_idx]
            scaler_m = StandardScaler()
            X_tr_s = scaler_m.fit_transform(X_tr)
            X_val_s = scaler_m.transform(X_val)
            X_tr_s, y_tr = _apply_smote(X_tr_s, y_tr)
            m = XGBClassifier(**p)
            m.fit(X_tr_s, y_tr)
            preds = m.predict(X_val_s)
            scores.append(f1_score(y_val, preds, average="macro"))
        return np.mean(scores)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(meta_obj, n_trials=30, n_jobs=1)
    meta_params = study.best_params
    meta_params.update({"random_state": args.seed, "n_jobs": args.n_jobs})
    print(f"  Meta-learner best params: n_est={meta_params['n_estimators']}, depth={meta_params['max_depth']}, lr={meta_params['learning_rate']:.4f}")

    # Final meta-learner fit (no SMOTE)
    scaler_final = StandardScaler()
    X_stack_tr_s = scaler_final.fit_transform(X_stack_tr)
    X_stack_te_s = scaler_final.transform(X_stack_te)
    meta = XGBClassifier(**meta_params)
    meta.fit(X_stack_tr_s, y_arr)
    preds_stack = meta.predict(X_stack_te_s)

    path = _write_sub(
        "tabpfn_oof_stacking", test_ids, preds_stack, args.output_dir,
        model="Stacking (TabPFN+XGB+CBT+LGB OOF + XGB meta)",
        features="42 base + targeted temporal + 24 OOF meta-features",
        notes=f"meta-learner: 30 trials macro-F1; base models 30 trials each",
    )
    print(f"  Stacking file: {path}")
    print(f"  Distribution: {pd.Series(preds_stack).value_counts().sort_index().to_dict()}")

    # Also generate simple blend for comparison (TabPFN + XGB only — reliable class alignment)
    blend_test = (test_tabpfn * 0.90 + test_xgb * 0.10)
    preds_blend = classes_all[np.argmax(blend_test, axis=1)]
    path_blend = _write_sub(
        "tabpfn_xgb_blend", test_ids, preds_blend, args.output_dir,
        model="TabPFN 0.90 XGB 0.10 blend",
        features="42 base + targeted temporal",
        notes="Simple avg: 0.90 TabPFN + 0.10 XGBoost",
    )
    print(f"  Blend file: {path_blend}")
    print(f"  Blend distribution: {pd.Series(preds_blend).value_counts().sort_index().to_dict()}")

    return path, path_blend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--no-submit", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.no_submit = True
        args.n_jobs = 1

    train_path = Path(args.data_dir) / "train" / "train"
    test_path = Path(args.data_dir) / "test" / "test"
    if not train_path.exists():
        train_path = Path(args.data_dir) / "train"
    if not test_path.exists():
        test_path = Path(args.data_dir) / "test"

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

    tabpfn_seed_ensemble(X_train_t, y_train, X_test_t, test_ids, args)
    tabpfn_fast_variants(X_train_t, y_train, X_test_t, test_ids, args)
    oof_stacking(X_train_t, y_train, users, X_test_t, test_ids, args)


if __name__ == "__main__":
    main()
