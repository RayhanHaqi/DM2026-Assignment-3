import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.oof import evaluate_oof_model
from model.rocket import fit_rocket_full, predict_rocket, train_rocket_candidate
from model.sequence import load_test_sequences, load_train_sequences
from model.temporal_features import combine_base_and_temporal_features
from model.train import _apply_smote, cv_evaluate, tune_lightgbm, tune_xgboost
from model.utils import generate_submission, load_test_data, load_train_data
from model.catboost_train import fit_catboost_full, predict_catboost, predict_catboost_proba, tune_catboost
from model.dart_train import fit_dart_full, predict_dart
from model.sequence_features import (
    build_catch22_features,
    fit_aeon_rocket_ridge,
    fit_minirocket_ridge,
    predict_aeon_rocket,
    predict_aeon_rocket_proba,
    predict_minirocket,
)
from model.sequence_oof import blend_probabilities, evaluate_sequence_oof_model, search_blend_weight
from model.hjorth_features import build_hjorth_spectral_features
from model.tabpfn_model import fit_tabpfn_full, fit_tabpfn_oof, predict_tabpfn, predict_tabpfn_proba


VALID_LABELS = {0, 1, 2, 3, 4, 5}


def _summarize_scores(scores):
    scores = np.asarray(scores, dtype=float)
    return {
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "worst": float(scores.min()),
    }


def _prediction_distribution(preds):
    values, counts = np.unique(np.asarray(preds, dtype=int), return_counts=True)
    distribution = {label: 0 for label in sorted(VALID_LABELS)}
    distribution.update({int(label): int(count) for label, count in zip(values, counts)})
    return distribution


def daily_tree_candidate_names():
    return ["lgb_macro_smote_refresh", "xgb_macro_smote_refresh"]


def plateau_candidate_names():
    return ["xgb_final_fit_audit", "xgb_targeted_temporal", "rocket_sequence"]


def targeted_candidate_names():
    return [
        "lgb_targeted_temporal_tuned",
        "xgb_targeted_temporal_seed_ensemble",
        "xgb_targeted_temporal_calibrated",
    ]


def next_candidate_names():
    return [
        "catboost_targeted_temporal_no_smote",
        "xgb_dart_targeted_temporal_no_smote",
        "sequence_feature_candidate",
    ]


def today_candidate_names():
    return [
        "xgb_targeted_temporal_accuracy_refit_no_smote",
        "xgb_targeted_temporal_seed_ensemble_no_smote",
        "xgb_catboost_conservative_blend",
    ]


def research_candidate_names():
    return [
        "aeon_minirocket_sequence",
        "aeon_multirocket_sequence",
        "catch22_targeted_xgb",
        "xgb_sequence_conservative_blend",
    ]


def improve_candidate_names():
    return [
        "xgb_catboost_tuned_blend",
        "xgb_pseudolabel_90",
    ]


def break080_candidate_names():
    return [
        "xgb_hjorth_spectral_selected",
        "xgb_multiobj_ensemble",
        "tabpfn_standalone",
        "stacking_ensemble",
    ]


def validate_submission_frame(file_ids, preds, expected_rows=6849):
    frame = pd.DataFrame({"Id": file_ids, "Label": preds})
    if len(frame) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, got {len(frame)}")
    if frame.isna().any().any():
        raise ValueError("Submission contains null values")
    labels = set(frame["Label"].astype(int).tolist())
    if not labels <= VALID_LABELS:
        raise ValueError(f"Submission contains invalid labels: {sorted(labels - VALID_LABELS)}")
    return frame


def _split_path(data_dir, split):
    path = Path(data_dir) / split
    nested = path / split
    if nested.exists():
        return nested
    return path


def _limit_by_user(X, ids, users, y=None, per_user_limit=None):
    if per_user_limit is None:
        if y is None:
            return X, ids, users
        return X, y, ids, users

    keep = users.groupby(users).cumcount() < per_user_limit
    if hasattr(X, "iloc"):
        X_limited = X.loc[keep].reset_index(drop=True)
    else:
        X_limited = X[np.asarray(keep)]
    ids_limited = ids.loc[keep].reset_index(drop=True)
    users_limited = users.loc[keep].reset_index(drop=True)
    if y is None:
        return X_limited, ids_limited, users_limited
    return X_limited, y.loc[keep].reset_index(drop=True), ids_limited, users_limited


def _fit_tree_model(model_cls, params, X_train, y_train, X_test, use_smote):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    y_fit = y_train
    if use_smote:
        X_train_scaled, y_fit = _apply_smote(X_train_scaled, y_train)
    model = model_cls(**params)
    model.fit(X_train_scaled, y_fit)
    return model.predict(X_test_scaled)


def _fit_tree_model_proba(model_cls, params, X_train, y_train, X_test, use_smote, random_state):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    y_fit = y_train
    if use_smote:
        X_train_scaled, y_fit = _apply_smote(X_train_scaled, y_train)
    model_params = {**params, "random_state": random_state}
    model = model_cls(**model_params)
    model.fit(X_train_scaled, y_fit)
    return model.predict_proba(X_test_scaled), model.classes_


def _align_proba_to_classes(proba, source_classes, target_classes):
    aligned = np.zeros((len(proba), len(target_classes)), dtype=float)
    source_index = {int(label): i for i, label in enumerate(source_classes)}
    for target_i, label in enumerate(target_classes):
        aligned[:, target_i] = proba[:, source_index[int(label)]]
    return aligned


def _write_submission(name, test_ids, preds, output_dir, no_submit, model, features, notes):
    validate_submission_frame(test_ids, preds, expected_rows=len(test_ids))
    if no_submit:
        return None
    return generate_submission(
        test_ids,
        preds,
        Path(output_dir) / f"submission_{name}.csv",
        model=model,
        features=features,
        notes=notes,
    )


def _run_lgb_candidate(name, X_train, y_train, users, X_test, test_ids, args, use_smote, metric="accuracy", final_fit_smote=None, features="42 base"):
    if final_fit_smote is None:
        final_fit_smote = use_smote
    print(f"\nTuning {name}...")
    params, _ = tune_lightgbm(
        X_train,
        y_train,
        users,
        n_trials=args.tree_trials,
        metric=metric,
        use_smote=use_smote,
        n_jobs=args.n_jobs,
    )
    scores, acc, acc_std = cv_evaluate(
        LGBMClassifier(**params),
        X_train,
        y_train,
        users,
        metric="accuracy",
        use_smote=use_smote,
    )
    _, f1, _ = cv_evaluate(
        LGBMClassifier(**params),
        X_train,
        y_train,
        users,
        metric="f1_macro",
        use_smote=use_smote,
    )
    preds = _fit_tree_model(LGBMClassifier, params, X_train, y_train, X_test, use_smote=final_fit_smote)
    path = _write_submission(
        name,
        test_ids,
        preds,
        args.output_dir,
        args.no_submit,
        model="LightGBM",
        features=features,
        notes=f"{metric}-tuned; {args.tree_trials} trials; cv_smote={use_smote}; final_fit_smote={final_fit_smote}",
    )
    return {"name": name, "accuracy": acc, "accuracy_std": acc_std, "f1_macro": f1, "file": path, "scores": scores}


def _run_xgb_candidate(name, X_train, y_train, users, X_test, test_ids, args, use_smote=False, metric="accuracy", final_fit_smote=None, features="42 base"):
    if final_fit_smote is None:
        final_fit_smote = use_smote
    print(f"\nTuning {name}...")
    params, _ = tune_xgboost(
        X_train,
        y_train,
        users,
        n_trials=args.tree_trials,
        metric=metric,
        use_smote=use_smote,
        n_jobs=args.n_jobs,
    )
    scores, acc, acc_std = cv_evaluate(
        XGBClassifier(**params),
        X_train,
        y_train,
        users,
        metric="accuracy",
        use_smote=use_smote,
    )
    _, f1, _ = cv_evaluate(
        XGBClassifier(**params),
        X_train,
        y_train,
        users,
        metric="f1_macro",
        use_smote=use_smote,
    )
    preds = _fit_tree_model(XGBClassifier, params, X_train, y_train, X_test, use_smote=final_fit_smote)
    path = _write_submission(
        name,
        test_ids,
        preds,
        args.output_dir,
        args.no_submit,
        model="XGBoost",
        features=features,
        notes=f"{metric}-tuned; {args.tree_trials} trials; cv_smote={use_smote}; final_fit_smote={final_fit_smote}",
    )
    return {"name": name, "accuracy": acc, "accuracy_std": acc_std, "f1_macro": f1, "file": path, "scores": scores}


def _run_daily_tree_candidates(X_train, y_train, users, X_test, test_ids, args):
    lgb_name, xgb_name = daily_tree_candidate_names()
    return [
        _run_lgb_candidate(lgb_name, X_train, y_train, users, X_test, test_ids, args, use_smote=True, metric="f1_macro"),
        _run_xgb_candidate(xgb_name, X_train, y_train, users, X_test, test_ids, args, use_smote=True, metric="f1_macro"),
    ]


def _run_plateau_tree_candidates(X_train, y_train, users, X_test, test_ids, X_seq, X_test_seq, args):
    audit_name, temporal_name, _ = plateau_candidate_names()
    X_train_temporal = combine_base_and_temporal_features(X_train, X_seq)
    X_test_temporal = combine_base_and_temporal_features(X_test, X_test_seq)
    return [
        _run_xgb_candidate(
            audit_name,
            X_train,
            y_train,
            users,
            X_test,
            test_ids,
            args,
            use_smote=True,
            metric="f1_macro",
            final_fit_smote=False,
            features="42 base",
        ),
        _run_xgb_candidate(
            temporal_name,
            X_train_temporal,
            y_train,
            users,
            X_test_temporal,
            test_ids,
            args,
            use_smote=True,
            metric="f1_macro",
            final_fit_smote=False,
            features="42 base + targeted temporal",
        ),
    ]


def _run_rocket_candidate(X_seq, y, users, X_test_seq, test_ids, args):
    name = plateau_candidate_names()[2]
    print(f"\nTraining {name}...")
    result = train_rocket_candidate(
        X_seq,
        y,
        users,
        n_kernels=args.rocket_kernels,
        random_state=args.seed,
    )
    model = fit_rocket_full(X_seq, y, n_kernels=args.rocket_kernels, random_state=args.seed)
    preds = predict_rocket(model, X_test_seq)
    path = _write_submission(
        name,
        test_ids,
        preds,
        args.output_dir,
        args.no_submit,
        model="ROCKET Ridge",
        features="raw 300x6 sequence",
        notes=f"n_kernels={args.rocket_kernels}; grouped validation",
    )
    return {
        "name": name,
        "accuracy": result.accuracy,
        "accuracy_std": result.accuracy_std,
        "f1_macro": result.f1_macro,
        "file": path,
        "scores": result.accuracy_scores,
    }


def _run_cnn_candidate(X_seq, y, users, X_test_seq, test_ids, args, name="cnn_raw_sequence", variant="small", normalize=False):
    from model.cnn import fit_cnn_full, predict_cnn, train_cnn_candidate

    print(f"\nTraining {name}...")
    result = train_cnn_candidate(
        X_seq,
        y,
        users,
        epochs=args.cnn_epochs,
        batch_size=args.cnn_batch_size,
        patience=args.cnn_patience,
        device=args.device,
        seed=args.seed,
        variant=variant,
        normalize=normalize,
    )
    full_epochs = max(1, result.best_epoch)
    model = fit_cnn_full(
        X_seq,
        y,
        epochs=full_epochs,
        batch_size=args.cnn_batch_size,
        device=args.device,
        seed=args.seed,
        variant=variant,
        normalize=normalize,
    )
    preds = predict_cnn(model, X_test_seq, device=args.device, normalize=normalize)
    path = _write_submission(
        name,
        test_ids,
        preds,
        args.output_dir,
        args.no_submit,
        model="1D CNN",
        features="raw 300x6 sequence",
        notes=f"variant={variant}; normalize={normalize}; epochs={full_epochs}; validation from grouped split",
    )
    return {
        "name": name,
        "accuracy": result.accuracy,
        "accuracy_std": 0.0,
        "f1_macro": result.f1_macro,
        "file": path,
        "scores": [result.accuracy],
    }


def _search_class_multipliers(oof_proba, y_true, classes, n_iters=2000):
    best_mult = np.ones(len(classes))
    best_score = f1_score(y_true, oof_proba.argmax(axis=1), average="macro")
    rng = np.random.RandomState(42)
    options = [0.85, 1.0, 1.15, 1.3, 1.5]
    for _ in range(n_iters):
        mult = np.ones(len(classes))
        for i in range(len(classes)):
            mult[i] = float(rng.choice(options))
        adjusted = oof_proba * mult[np.newaxis, :]
        preds = classes[adjusted.argmax(axis=1)]
        score = f1_score(y_true, preds, average="macro")
        if score > best_score:
            best_score = score
            best_mult = mult.copy()
    return best_mult, best_score


def _run_xgb_seed_ensemble(name, X_train, y_train, users, X_test, test_ids, args,
                           use_smote=False, metric="f1_macro", final_fit_smote=None,
                           seeds=None, features="42 base + targeted temporal"):
    if final_fit_smote is None:
        final_fit_smote = use_smote
    if seeds is None:
        seeds = [11, 23, 42, 71, 101]

    print(f"\nTuning {name} (base model)...")
    params, _ = tune_xgboost(
        X_train, y_train, users,
        n_trials=args.tree_trials, metric=metric, use_smote=use_smote,
        n_jobs=args.n_jobs,
    )
    scores, acc, acc_std = cv_evaluate(
        XGBClassifier(**params), X_train, y_train, users,
        metric="accuracy", use_smote=use_smote,
    )
    _, f1, _ = cv_evaluate(
        XGBClassifier(**params), X_train, y_train, users,
        metric="f1_macro", use_smote=use_smote,
    )

    print(f"Training ensemble with seeds: {seeds}")
    all_probas = []
    classes = None
    for seed in seeds:
        proba, clz = _fit_tree_model_proba(
            XGBClassifier, params, X_train, y_train, X_test,
            use_smote=final_fit_smote, random_state=seed,
        )
        if classes is None:
            classes = clz
        all_probas.append(proba)

    mean_proba = np.mean(all_probas, axis=0)
    preds = classes[np.argmax(mean_proba, axis=1)]

    path = _write_submission(
        name, test_ids, preds, args.output_dir, args.no_submit,
        model="XGBoost Seed Ensemble",
        features=features,
        notes=f"{metric}-tuned; {args.tree_trials} trials; seeds={seeds}; cv_smote={use_smote}; final_fit_smote={final_fit_smote}",
    )
    return {"name": name, "accuracy": acc, "accuracy_std": acc_std, "f1_macro": f1, "file": path, "scores": scores}


def _run_targeted_candidates(X_train, y_train, users, X_test, test_ids, X_seq, X_test_seq, args):
    lgb_name, ensemble_name, calib_name = targeted_candidate_names()

    X_train_temporal = combine_base_and_temporal_features(X_train, X_seq)
    X_test_temporal = combine_base_and_temporal_features(X_test, X_test_seq)

    results = []

    lgb_result = _run_lgb_candidate(
        lgb_name, X_train_temporal, y_train, users, X_test_temporal, test_ids, args,
        use_smote=True, metric="f1_macro", final_fit_smote=True,
        features="42 base + targeted temporal",
    )
    results.append(lgb_result)

    ensemble_result = _run_xgb_seed_ensemble(
        ensemble_name, X_train_temporal, y_train, users, X_test_temporal, test_ids, args,
        use_smote=True, metric="f1_macro", final_fit_smote=True,
        features="42 base + targeted temporal",
    )
    results.append(ensemble_result)

    print(f"\nRunning OOF evaluation for calibration...")
    params, _ = tune_xgboost(
        X_train_temporal, y_train, users,
        n_trials=args.tree_trials, metric="f1_macro", use_smote=True,
        n_jobs=args.n_jobs,
    )
    oof_result = evaluate_oof_model(
        XGBClassifier(**params), X_train_temporal, y_train, users,
        X_test_temporal, n_splits=5, use_smote=True, name="xgb_calib_oof",
    )
    base_f1 = oof_result.macro_f1
    mult, calib_f1 = _search_class_multipliers(oof_result.oof_proba, y_train, oof_result.classes)

    print(f"Calibration: base OOF macro-F1={base_f1:.4f}, calibrated OOF macro-F1={calib_f1:.4f}")
    print(f"Multipliers: {dict(zip(oof_result.classes.tolist(), mult.tolist()))}")

    if calib_f1 > base_f1:
        adjusted_test = oof_result.test_proba * mult[np.newaxis, :]
        calib_preds = oof_result.classes[adjusted_test.argmax(axis=1)]
        path = _write_submission(
            calib_name, test_ids, calib_preds, args.output_dir, args.no_submit,
            model="XGBoost Calibrated",
            features="42 base + targeted temporal",
            notes=f"multipliers={dict(zip(oof_result.classes.tolist(), [round(m, 3) for m in mult.tolist()]))}; base_OOF_f1={base_f1:.4f}; calib_OOF_f1={calib_f1:.4f}",
        )
        results.append({
            "name": calib_name, "accuracy": oof_result.accuracy,
            "accuracy_std": oof_result.accuracy_std, "f1_macro": calib_f1,
            "file": path, "scores": oof_result.fold_accuracy,
        })
    else:
        print(f"Skipping calibration submission: calibrated F1 ({calib_f1:.4f}) <= base ({base_f1:.4f})")
        results.append({
            "name": calib_name, "accuracy": oof_result.accuracy,
            "accuracy_std": oof_result.accuracy_std, "f1_macro": base_f1,
            "file": None, "scores": oof_result.fold_accuracy,
        })

    return results


def _run_next_candidates(X_train, y_train, users, X_test, test_ids, X_seq, X_test_seq, args):
    cat_name, dart_name, sequence_name = next_candidate_names()
    X_train_temporal = combine_base_and_temporal_features(X_train, X_seq)
    X_test_temporal = combine_base_and_temporal_features(X_test, X_test_seq)
    results = []

    print(f"\nTraining {cat_name}...")
    cat_iterations = 20 if getattr(args, "smoke", False) else 500
    try:
        cat_model = fit_catboost_full(
            X_train_temporal,
            y_train,
            n_jobs=args.n_jobs,
            random_state=args.seed,
            iterations=cat_iterations,
        )
        cat_preds = predict_catboost(cat_model, X_test_temporal)
    except Exception as exc:
        print(f"CatBoost failed ({exc}); falling back to DART for this slot")
        dart_estimators = 10 if getattr(args, "smoke", False) else 500
        cat_model = fit_dart_full(
            X_train_temporal,
            y_train,
            n_jobs=args.n_jobs,
            random_state=args.seed,
            n_estimators=dart_estimators,
        )
        cat_preds = predict_dart(cat_model, X_test_temporal)
    cat_path = _write_submission(
        cat_name,
        test_ids,
        cat_preds,
        args.output_dir,
        args.no_submit,
        model="CatBoost",
        features="42 base + targeted temporal",
        notes=f"no SMOTE; iterations={cat_iterations}; n_jobs={args.n_jobs}",
    )
    results.append({"name": cat_name, "accuracy": 0.0, "accuracy_std": 0.0, "f1_macro": 0.0, "file": cat_path, "scores": []})

    print(f"\nTraining {dart_name}...")
    dart_estimators = 10 if getattr(args, "smoke", False) else 500
    dart_model = fit_dart_full(
        X_train_temporal,
        y_train,
        n_jobs=args.n_jobs,
        random_state=args.seed,
        n_estimators=dart_estimators,
    )
    dart_preds = predict_dart(dart_model, X_test_temporal)
    dart_path = _write_submission(
        dart_name,
        test_ids,
        dart_preds,
        args.output_dir,
        args.no_submit,
        model="XGBoost DART",
        features="42 base + targeted temporal",
        notes=f"booster=dart; final_fit_smote=False; n_estimators={dart_estimators}; n_jobs={args.n_jobs}",
    )
    results.append({"name": dart_name, "accuracy": 0.0, "accuracy_std": 0.0, "f1_macro": 0.0, "file": dart_path, "scores": []})

    print(f"\nTraining {sequence_name}...")
    try:
        seq_model = fit_minirocket_ridge(X_seq, y_train)
        seq_preds = predict_minirocket(seq_model, X_test_seq)
        seq_model_name = "MiniRocket Ridge"
        seq_features = "raw 300x6 sequence"
        seq_notes = "aeon MiniRocketMultivariate"
    except ImportError as exc:
        print(f"MiniRocket unavailable: {exc}")
        catch_train = build_catch22_features(X_seq)
        catch_test = build_catch22_features(X_test_seq)
        catch_train = pd.concat([X_train_temporal.reset_index(drop=True), catch_train.reset_index(drop=True)], axis=1)
        catch_test = pd.concat([X_test_temporal.reset_index(drop=True), catch_test.reset_index(drop=True)], axis=1)
        dart_model = fit_dart_full(catch_train, y_train, n_jobs=args.n_jobs, random_state=args.seed, n_estimators=dart_estimators)
        seq_preds = predict_dart(dart_model, catch_test)
        seq_model_name = "XGBoost DART"
        seq_features = "42 base + targeted temporal + catch22"
        seq_notes = "catch22 fallback"

    seq_path = _write_submission(
        sequence_name,
        test_ids,
        seq_preds,
        args.output_dir,
        args.no_submit,
        model=seq_model_name,
        features=seq_features,
        notes=seq_notes,
    )
    results.append({"name": sequence_name, "accuracy": 0.0, "accuracy_std": 0.0, "f1_macro": 0.0, "file": seq_path, "scores": []})
    return results


def _run_xgb_catboost_blend(name, X_train, y_train, users, X_test, test_ids, args, features="42 base + targeted temporal"):
    print(f"\nTraining {name}...")
    params, _ = tune_xgboost(
        X_train,
        y_train,
        users,
        n_trials=args.tree_trials,
        metric="f1_macro",
        use_smote=True,
        n_jobs=args.n_jobs,
    )
    scores, acc, acc_std = cv_evaluate(
        XGBClassifier(**params),
        X_train,
        y_train,
        users,
        metric="accuracy",
        use_smote=True,
    )
    _, f1, _ = cv_evaluate(
        XGBClassifier(**params),
        X_train,
        y_train,
        users,
        metric="f1_macro",
        use_smote=True,
    )

    xgb_proba, xgb_classes = _fit_tree_model_proba(
        XGBClassifier,
        params,
        X_train,
        y_train,
        X_test,
        use_smote=False,
        random_state=args.seed,
    )

    print(f"  Tuning CatBoost...")
    cat_params, cat_model = tune_catboost(
        X_train,
        y_train,
        users,
        n_trials=args.tree_trials,
        metric="f1_macro",
        use_smote=True,
        n_jobs=args.n_jobs,
    )
    cat_proba = predict_catboost_proba(cat_model, X_test)
    cat_classes = getattr(cat_model, "classes_", xgb_classes)
    cat_aligned = _align_proba_to_classes(cat_proba, cat_classes, xgb_classes)

    print(f"  Computing OOF for blend ratio search...")
    xgb_oof = evaluate_oof_model(
        XGBClassifier(**params),
        X_train, y_train, users,
        X_test,
        n_splits=3 if getattr(args, "smoke", False) else 5,
        use_smote=True,
        name="xgb_oof",
    )
    cat_oof = evaluate_oof_model(
        cat_model,
        X_train, y_train, users,
        X_test,
        n_splits=3 if getattr(args, "smoke", False) else 5,
        use_smote=True,
        name="cat_oof",
    )
    cat_oof_aligned = _align_proba_to_classes(cat_oof.oof_proba, cat_oof.classes, xgb_oof.classes)

    best_weight = 0.0
    best_oof_acc = -1.0
    for w in [0.0, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25]:
        blend = (1.0 - w) * xgb_oof.oof_proba + w * cat_oof_aligned
        preds = xgb_oof.classes[np.argmax(blend, axis=1)]
        oof_acc = float(accuracy_score(np.asarray(y_train), preds))
        print(f"    cat_weight={w:.2f} oof_acc={oof_acc:.4f}")
        if oof_acc > best_oof_acc:
            best_oof_acc = oof_acc
            best_weight = w

    print(f"  Best blend ratio: cat_weight={best_weight:.2f} (OOF acc={best_oof_acc:.4f})")

    blend_proba = (1.0 - best_weight) * xgb_proba + best_weight * cat_aligned
    preds = xgb_classes[np.argmax(blend_proba, axis=1)]
    path = _write_submission(
        name,
        test_ids,
        preds,
        args.output_dir,
        args.no_submit,
        model="XGBoost + CatBoost Blend",
        features=features,
        notes=f"xgb_metric=f1_macro; cat_tuned; cv_smote=True; final_fit_smote=False; blend={(1.0 - best_weight):.2f}xgb/{best_weight:.2f}cat; best_oof_acc={best_oof_acc:.4f}",
    )
    return {"name": name, "accuracy": acc, "accuracy_std": acc_std, "f1_macro": f1, "file": path, "scores": scores}


def _run_today_candidates(X_train, y_train, users, X_test, test_ids, X_seq, X_test_seq, args):
    accuracy_name, ensemble_name, blend_name = today_candidate_names()
    X_train_temporal = combine_base_and_temporal_features(X_train, X_seq)
    X_test_temporal = combine_base_and_temporal_features(X_test, X_test_seq)

    return [
        _run_xgb_candidate(
            accuracy_name,
            X_train_temporal,
            y_train,
            users,
            X_test_temporal,
            test_ids,
            args,
            use_smote=True,
            metric="accuracy",
            final_fit_smote=False,
            features="42 base + targeted temporal",
        ),
        _run_xgb_seed_ensemble(
            ensemble_name,
            X_train_temporal,
            y_train,
            users,
            X_test_temporal,
            test_ids,
            args,
            use_smote=True,
            metric="f1_macro",
            final_fit_smote=False,
            seeds=[11, 23, 42, 71, 101],
            features="42 base + targeted temporal",
        ),
        _run_xgb_catboost_blend(
            blend_name,
            X_train_temporal,
            y_train,
            users,
            X_test_temporal,
            test_ids,
            args,
            features="42 base + targeted temporal",
        ),
    ]


def _run_aeon_sequence_candidate(name, kind, X_seq, y_train, users, X_test_seq, test_ids, args):
    print(f"\nTraining {name}...")
    if kind == "multirocket":
        n_kernels = 84 if getattr(args, "smoke", False) else min(args.rocket_kernels, 5000)
    else:
        n_kernels = 84 if getattr(args, "smoke", False) else max(84, args.rocket_kernels)

    def fit_fn(X_fold, y_fold):
        return fit_aeon_rocket_ridge(
            X_fold,
            y_fold,
            kind=kind,
            n_kernels=n_kernels,
            n_jobs=args.n_jobs,
            random_state=args.seed,
        )

    result = evaluate_sequence_oof_model(
        fit_fn,
        predict_aeon_rocket_proba,
        np.asarray(X_seq),
        np.asarray(y_train),
        np.asarray(users),
        np.asarray(X_test_seq),
        n_splits=3 if getattr(args, "smoke", False) else 5,
        random_state=args.seed,
        name=name,
    )
    full_model = fit_fn(np.asarray(X_seq), np.asarray(y_train))
    preds = predict_aeon_rocket(full_model, np.asarray(X_test_seq))
    path = _write_submission(
        name,
        test_ids,
        preds,
        args.output_dir,
        args.no_submit,
        model=f"aeon {kind} Ridge",
        features="raw 300x6 sequence",
        notes=f"n_kernels={n_kernels}; grouped OOF validation; standalone sequence research",
    )
    return {
        "name": name,
        "accuracy": result.accuracy,
        "accuracy_std": result.accuracy_std,
        "f1_macro": result.macro_f1,
        "file": path,
        "scores": result.fold_accuracy,
    }


def _run_catch22_targeted_xgb(name, X_train_temporal, y_train, users, X_test_temporal, test_ids, X_seq, X_test_seq, args):
    print(f"\nTraining {name}...")
    catch_train = build_catch22_features(X_seq)
    catch_test = build_catch22_features(X_test_seq)
    X_train_full = pd.concat([X_train_temporal.reset_index(drop=True), catch_train.reset_index(drop=True)], axis=1)
    X_test_full = pd.concat([X_test_temporal.reset_index(drop=True), catch_test.reset_index(drop=True)], axis=1)
    return _run_xgb_candidate(
        name,
        X_train_full,
        y_train,
        users,
        X_test_full,
        test_ids,
        args,
        use_smote=True,
        metric="f1_macro",
        final_fit_smote=False,
        features="42 base + targeted temporal + catch22",
    )


def _run_xgb_sequence_blend(name, X_train_temporal, y_train, users, X_test_temporal, test_ids, X_seq, X_test_seq, args):
    print(f"\nTraining {name}...")
    params, _ = tune_xgboost(
        X_train_temporal,
        y_train,
        users,
        n_trials=args.tree_trials,
        metric="f1_macro",
        use_smote=True,
        n_jobs=args.n_jobs,
    )
    from model.oof import evaluate_oof_model

    base_oof = evaluate_oof_model(
        XGBClassifier(**params),
        X_train_temporal,
        y_train,
        users,
        X_test_temporal,
        n_splits=3 if getattr(args, "smoke", False) else 5,
        use_smote=True,
        name="xgb_research_oof",
    )
    n_kernels = 84 if getattr(args, "smoke", False) else max(84, args.rocket_kernels)

    def fit_fn(X_fold, y_fold):
        return fit_aeon_rocket_ridge(
            X_fold,
            y_fold,
            kind="minirocket",
            n_kernels=n_kernels,
            n_jobs=args.n_jobs,
            random_state=args.seed,
        )

    seq_oof = evaluate_sequence_oof_model(
        fit_fn,
        predict_aeon_rocket_proba,
        np.asarray(X_seq),
        np.asarray(y_train),
        np.asarray(users),
        np.asarray(X_test_seq),
        n_splits=3 if getattr(args, "smoke", False) else 5,
        random_state=args.seed,
        name="minirocket_research_oof",
    )
    weight, blend_acc, blend_oof_preds = search_blend_weight(
        base_oof.oof_proba,
        seq_oof.oof_proba,
        np.asarray(y_train),
        base_oof.classes,
        weights=(0.0, 0.02, 0.05, 0.08, 0.10),
    )
    test_blend = blend_probabilities(base_oof.test_proba, seq_oof.test_proba, weight)
    preds = base_oof.classes[np.argmax(test_blend, axis=1)]
    path = _write_submission(
        name,
        test_ids,
        preds,
        args.output_dir,
        args.no_submit,
        model="XGBoost + aeon MiniRocket Blend",
        features="42 base + targeted temporal + raw sequence blend",
        notes=f"xgb_metric=f1_macro; cv_smote=True; final_fit_smote=False; sequence_weight={weight:.2f}; blend_oof_acc={blend_acc:.4f}",
    )
    return {
        "name": name,
        "accuracy": blend_acc,
        "accuracy_std": base_oof.accuracy_std,
        "f1_macro": float(f1_score(np.asarray(y_train), blend_oof_preds, average="macro", zero_division=0)),
        "file": path,
        "scores": base_oof.fold_accuracy,
    }


def _run_xgb_pseudolabel(name, X_train, y_train, users, X_test, test_ids, args,
                          threshold=0.90, features="42 base + targeted temporal"):
    print(f"\nTraining {name} (threshold={threshold})...")
    params, _ = tune_xgboost(
        X_train, y_train, users,
        n_trials=args.tree_trials,
        metric="f1_macro",
        use_smote=True,
        n_jobs=args.n_jobs,
    )
    scores, acc, acc_std = cv_evaluate(
        XGBClassifier(**params), X_train, y_train, users,
        metric="accuracy", use_smote=True,
    )
    _, f1, _ = cv_evaluate(
        XGBClassifier(**params), X_train, y_train, users,
        metric="f1_macro", use_smote=True,
    )

    test_proba, classes = _fit_tree_model_proba(
        XGBClassifier, params, X_train, y_train, X_test,
        use_smote=False, random_state=args.seed,
    )
    confidence = test_proba.max(axis=1)
    pseudo_mask = confidence >= threshold
    n_pseudo = int(pseudo_mask.sum())
    print(f"  Confidence >= {threshold}: {n_pseudo} test samples selected")

    if n_pseudo < 500:
        print(f"  WARNING: too few pseudo-labels ({n_pseudo} < 500); falling back to standard XGBoost")
        preds = classes[np.argmax(test_proba, axis=1)]
    else:
        pseudo_labels = classes[np.argmax(test_proba[pseudo_mask], axis=1)]
        X_aug = np.vstack([np.asarray(X_train), X_test[pseudo_mask]])
        y_aug = np.concatenate([np.asarray(y_train), pseudo_labels])
        scaler = StandardScaler()
        X_aug_s = scaler.fit_transform(X_aug)
        X_test_s = scaler.transform(X_test)
        model = XGBClassifier(**{**params, "random_state": args.seed})
        model.fit(X_aug_s, y_aug)
        test_proba_aug = model.predict_proba(X_test_s)
        preds = model.classes_[np.argmax(test_proba_aug, axis=1)]

    path = _write_submission(
        name, test_ids, preds, args.output_dir, args.no_submit,
        model="XGBoost Pseudo-Label",
        features=features,
        notes=f"f1_macro-tuned; cv_smote=True; final_fit_smote=False; pseudo_threshold={threshold}; n_pseudo={n_pseudo}",
    )
    return {"name": name, "accuracy": acc, "accuracy_std": acc_std, "f1_macro": f1, "file": path, "scores": scores}


def _run_improve_candidates(X_train, y_train, users, X_test, test_ids, X_seq, X_test_seq, args):
    blend_name, pseudo_name = improve_candidate_names()
    X_train_temporal = combine_base_and_temporal_features(X_train, X_seq)
    X_test_temporal = combine_base_and_temporal_features(X_test, X_test_seq)

    return [
        _run_xgb_catboost_blend(
            blend_name, X_train_temporal, y_train, users, X_test_temporal, test_ids, args,
            features="42 base + targeted temporal",
        ),
        _run_xgb_pseudolabel(
            pseudo_name, X_train_temporal, y_train, users, X_test_temporal, test_ids, args,
            threshold=0.90, features="42 base + targeted temporal",
        ),
    ]


def _run_hjorth_candidate(name, X_train, y_train, users, X_test, test_ids, X_seq, X_test_seq, args):
    print(f"\nTraining {name}...")
    hjorth_train = build_hjorth_spectral_features(X_seq)
    hjorth_test = build_hjorth_spectral_features(X_test_seq)

    X_train_full = pd.concat([X_train.reset_index(drop=True), hjorth_train.reset_index(drop=True)], axis=1)
    X_test_full = pd.concat([X_test.reset_index(drop=True), hjorth_test.reset_index(drop=True)], axis=1)

    from sklearn.feature_selection import mutual_info_classif
    mi = mutual_info_classif(np.asarray(X_train_full), np.asarray(y_train), random_state=args.seed)
    mi_series = pd.Series(mi, index=X_train_full.columns).sort_values(ascending=False)
    n_keep = min(100, len(mi_series))
    selected_cols = mi_series.head(n_keep).index.tolist()
    print(f"  MI selection: keeping {len(selected_cols)}/{len(mi_series)} features")

    X_train_sel = X_train_full[selected_cols]
    X_test_sel = X_test_full[selected_cols]

    return _run_xgb_candidate(
        name,
        X_train_sel, y_train, users, X_test_sel, test_ids, args,
        use_smote=True, metric="f1_macro", final_fit_smote=False,
        features=f"42 base + targeted temporal + HJ+Spectral (MI selected, {len(selected_cols)} features)",
    )


def _run_multiobj_ensemble(name, X_train, y_train, users, X_test, test_ids, args):
    print(f"\nTraining {name}...")
    objectives = ["f1_macro", "accuracy", "neg_log_loss"]
    all_probas = []
    classes = None

    for metric in objectives:
        print(f"  Tuning XGB with metric={metric} (repeated CV)...")
        params, _ = tune_xgboost(
            X_train, y_train, users,
            n_trials=args.tree_trials,
            metric=metric,
            use_smote=True,
            n_jobs=args.n_jobs,
            n_repeats=3,
        )
        scores, acc, acc_std = cv_evaluate(
            XGBClassifier(**params), X_train, y_train, users,
            metric="accuracy", use_smote=True,
        )
        print(f"    CV accuracy: {acc:.4f} +/- {acc_std:.4f}")

        proba, clz = _fit_tree_model_proba(
            XGBClassifier, params, X_train, y_train, X_test,
            use_smote=False, random_state=args.seed,
        )
        if classes is None:
            classes = clz
        all_probas.append(proba)

    mean_proba = np.mean(all_probas, axis=0)
    preds = classes[np.argmax(mean_proba, axis=1)]

    path = _write_submission(
        name, test_ids, preds, args.output_dir, args.no_submit,
        model="XGBoost Multi-Objective Ensemble",
        features="42 base + targeted temporal",
        notes=f"repeated_cv=3; metrics={objectives}; avg of 3 models",
    )
    return {"name": name, "accuracy": 0.0, "accuracy_std": 0.0, "f1_macro": 0.0, "file": path, "scores": []}


def _run_tabpfn_standalone(name, X_train, y_train, users, X_test, test_ids, args):
    print(f"\nTraining {name}...")
    try:
        model, scaler = fit_tabpfn_full(
            X_train, np.asarray(y_train),
            n_estimators=16, device=args.device or "cuda", random_state=args.seed,
        )
    except ImportError as exc:
        print(f"  TabPFN unavailable: {exc}")
        print(f"  Skipping {name} — set TABPFN_TOKEN env var to enable")
        return {"name": name, "accuracy": 0.0, "accuracy_std": 0.0, "f1_macro": 0.0, "file": None, "scores": []}

    oof_proba, oof_preds = fit_tabpfn_oof(
        X_train, np.asarray(y_train), np.asarray(users),
        n_splits=3 if getattr(args, "smoke", False) else 5,
        n_estimators=16, device=args.device or "cuda", random_state=args.seed,
    )
    oof_acc = float(accuracy_score(np.asarray(y_train), oof_preds))
    oof_f1 = float(f1_score(np.asarray(y_train), oof_preds, average="macro"))
    print(f"  TabPFN OOF accuracy: {oof_acc:.4f}, OOF macro-F1: {oof_f1:.4f}")

    preds = predict_tabpfn((model, scaler), X_test)

    path = _write_submission(
        name, test_ids, preds, args.output_dir, args.no_submit,
        model="TabPFN",
        features="42 base + targeted temporal",
        notes=f"n_estimators=16; eval_metric=f1; oof_acc={oof_acc:.4f}; oof_f1={oof_f1:.4f}",
    )
    return {"name": name, "accuracy": oof_acc, "accuracy_std": 0.0, "f1_macro": oof_f1, "file": path, "scores": []}


def _run_stacking_ensemble(name, X_train, y_train, users, X_test, test_ids,
                           X_seq, X_test_seq, args):
    print(f"\nTraining {name}...")

    print("  Training XGBoost (base)...")
    xgb_params, _ = tune_xgboost(
        X_train, y_train, users,
        n_trials=args.tree_trials, metric="f1_macro", use_smote=True,
        n_jobs=args.n_jobs, n_repeats=3,
    )
    xgb_oof = evaluate_oof_model(
        XGBClassifier(**xgb_params), X_train, y_train, users, X_test,
        n_splits=3 if getattr(args, "smoke", False) else 5, use_smote=True, name="xgb_stack",
    )

    print("  Training CatBoost...")
    cat_params, cat_model = tune_catboost(
        X_train, y_train, users,
        n_trials=args.tree_trials, metric="f1_macro", use_smote=True, n_jobs=args.n_jobs,
    )
    cat_oof = evaluate_oof_model(
        cat_model, X_train, y_train, users, X_test,
        n_splits=3 if getattr(args, "smoke", False) else 5, use_smote=True, name="cat_stack",
    )

    print("  Training LightGBM...")
    lgb_params, _ = tune_lightgbm(
        X_train, y_train, users,
        n_trials=args.tree_trials, metric="f1_macro", use_smote=True, n_jobs=args.n_jobs,
    )
    lgb_oof = evaluate_oof_model(
        LGBMClassifier(**lgb_params), X_train, y_train, users, X_test,
        n_splits=3 if getattr(args, "smoke", False) else 5, use_smote=True, name="lgb_stack",
    )

    tabpfn_available = False
    try:
        print("  Training TabPFN...")
        tabpfn_oof_proba, _ = fit_tabpfn_oof(
            X_train, np.asarray(y_train), np.asarray(users),
            n_splits=3 if getattr(args, "smoke", False) else 5,
            n_estimators=16, device=args.device or "cuda", random_state=args.seed,
        )
        tabpfn_available = True
    except ImportError as exc:
        print(f"  TabPFN unavailable: {exc}")
        print(f"  Stacking without TabPFN — set TABPFN_TOKEN to enable")

    classes_list = xgb_oof.classes.tolist()
    n_classes = len(classes_list)

    stack_feat_train = [xgb_oof.oof_proba, cat_oof.oof_proba, lgb_oof.oof_proba]
    stack_feat_test = [xgb_oof.test_proba, cat_oof.test_proba, lgb_oof.test_proba]

    if tabpfn_available:
        stack_feat_train.append(tabpfn_oof_proba)
        try:
            tabpfn_model_full, tabpfn_scaler_f = fit_tabpfn_full(
                X_train, np.asarray(y_train),
                n_estimators=16, device=args.device or "cuda", random_state=args.seed,
            )
            tabpfn_test_proba, _ = predict_tabpfn_proba((tabpfn_model_full, tabpfn_scaler_f), X_test)
            stack_feat_test.append(tabpfn_test_proba)
        except Exception as exc:
            print(f"  TabPFN full predict failed: {exc}")
            tabpfn_available = False

    X_stack_train = np.hstack([np.asarray(X_train)] + stack_feat_train)
    X_stack_test = np.hstack([np.asarray(X_test)] + stack_feat_test)

    print(f"  Training meta-learner XGBoost on {X_stack_train.shape[1]} features...")
    scaler = StandardScaler()
    X_stack_train_s = scaler.fit_transform(X_stack_train)
    X_stack_test_s = scaler.transform(X_stack_test)

    meta = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                         random_state=args.seed, n_jobs=args.n_jobs)
    meta.fit(X_stack_train_s, np.asarray(y_train))
    meta_preds = meta.predict(X_stack_test_s)

    model_desc = "XGB/CBT/LGB"
    if tabpfn_available:
        model_desc += "/TabPFN"
    path = _write_submission(
        name, test_ids, meta_preds, args.output_dir, args.no_submit,
        model=f"Stacking Ensemble ({model_desc})",
        features="42 base + targeted temporal + OOF meta-features",
        notes=f"meta-learner: XGBoost (default); base models: {model_desc} (all tuned+OOF)",
    )
    return {"name": name, "accuracy": 0.0, "accuracy_std": 0.0, "f1_macro": 0.0, "file": path, "scores": []}


def _run_break_080_candidates(X_train, y_train, users, X_test, test_ids,
                               X_seq, X_test_seq, args):
    hjorth_name, multiobj_name, tabpfn_name, stack_name = break080_candidate_names()
    X_train_temporal = combine_base_and_temporal_features(X_train, X_seq)
    X_test_temporal = combine_base_and_temporal_features(X_test, X_test_seq)

    results = []

    results.append(
        _run_hjorth_candidate(
            hjorth_name, X_train_temporal, y_train, users, X_test_temporal,
            test_ids, X_seq, X_test_seq, args,
        )
    )

    results.append(
        _run_multiobj_ensemble(
            multiobj_name, X_train_temporal, y_train, users, X_test_temporal, test_ids, args,
        )
    )

    results.append(
        _run_tabpfn_standalone(
            tabpfn_name, X_train_temporal, y_train, users, X_test_temporal, test_ids, args,
        )
    )

    results.append(
        _run_stacking_ensemble(
            stack_name, X_train_temporal, y_train, users, X_test_temporal,
            test_ids, X_seq, X_test_seq, args,
        )
    )

    return results


def _run_research_candidates(X_train, y_train, users, X_test, test_ids, X_seq, X_test_seq, args):
    mini_name, multi_name, catch_name, blend_name = research_candidate_names()
    X_train_temporal = combine_base_and_temporal_features(X_train, X_seq)
    X_test_temporal = combine_base_and_temporal_features(X_test, X_test_seq)

    return [
        _run_aeon_sequence_candidate(mini_name, "minirocket", X_seq, y_train, users, X_test_seq, test_ids, args),
        _run_aeon_sequence_candidate(multi_name, "multirocket", X_seq, y_train, users, X_test_seq, test_ids, args),
        _run_catch22_targeted_xgb(catch_name, X_train_temporal, y_train, users, X_test_temporal, test_ids, X_seq, X_test_seq, args),
        _run_xgb_sequence_blend(blend_name, X_train_temporal, y_train, users, X_test_temporal, test_ids, X_seq, X_test_seq, args),
    ]


def _print_summary(results):
    ranked = sorted(results, key=lambda row: row["accuracy"], reverse=True)
    print("\nCandidate summary:")
    for i, row in enumerate(ranked, start=1):
        file_text = row["file"] if row["file"] is not None else "not written (--no-submit)"
        print(
            f"{i}. {row['name']:<20} "
            f"acc={row['accuracy']:.4f} (+/- {row['accuracy_std']:.4f}) "
            f"f1={row['f1_macro']:.4f} file={file_text}"
        )
    print("\nRecommended order:")
    for row in ranked[:3]:
        print(f"- {row['name']}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run balanced ASG3 candidates.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--tree-trials", type=int, default=30)
    parser.add_argument("--cnn-epochs", type=int, default=30)
    parser.add_argument("--cnn-batch-size", type=int, default=128)
    parser.add_argument("--cnn-patience", type=int, default=5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-user-limit", type=int, default=None)
    parser.add_argument("--include-xgb", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument(
        "--daily-20260520",
        action="store_true",
        help="Run today's approved candidates: LGB macro/SMOTE, XGB macro/SMOTE, improved CNN.",
    )
    parser.add_argument(
        "--plateau-20260520",
        action="store_true",
        help="Run plateau-breaker candidates: XGB final-fit audit, targeted temporal XGB, ROCKET sequence.",
    )
    parser.add_argument(
        "--targeted-20260522",
        action="store_true",
        help="Run targeted temporal candidates: LGB tuned, XGB seed ensemble, XGB calibrated multipliers.",
    )
    parser.add_argument(
        "--next-20260524",
        action="store_true",
        help="Run next improvement candidates: CatBoost, XGB DART, and sequence feature path.",
    )
    parser.add_argument(
        "--research-20260525",
        action="store_true",
        help="Run research candidates: aeon ROCKET-family sequence models, catch22, and conservative XGB/sequence blend.",
    )
    parser.add_argument(
        "--today-20260525",
        action="store_true",
        help="Run today's candidates: XGB accuracy refit, XGB no-SMOTE seed ensemble, and XGB/CatBoost blend.",
    )
    parser.add_argument(
        "--improve-20260527",
        action="store_true",
        help="Run improvement candidates: tuned XGB+CatBoost blend with grid-searched ratio, and XGB pseudo-labeling (threshold=0.90).",
    )
    parser.add_argument(
        "--break-080",
        action="store_true",
        help="Run break-0.80 candidates: Hjorth+spectral features, multi-obj ensemble, TabPFN, stacking.",
    )
    parser.add_argument("--rocket-kernels", type=int, default=512)
    parser.add_argument("--no-submit", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        args.tree_trials = 1
        args.cnn_epochs = 1
        args.cnn_patience = 1
        args.per_user_limit = 2
        args.rocket_kernels = 8
        args.no_submit = True

    train_path = _split_path(args.data_dir, "train")
    test_path = _split_path(args.data_dir, "test")

    print(f"Loading aggregate data from {train_path} and {test_path}...")
    X_train, y_train, train_ids, users = load_train_data(train_path)
    X_test, test_ids, test_users = load_test_data(test_path)
    X_train, y_train, train_ids, users = _limit_by_user(X_train, train_ids, users, y_train, args.per_user_limit)
    X_test, test_ids, test_users = _limit_by_user(X_test, test_ids, test_users, per_user_limit=args.per_user_limit)

    print(f"\nLoading sequence data from {train_path} and {test_path}...")
    X_seq, y_seq, seq_ids, seq_users = load_train_sequences(train_path)
    X_test_seq, seq_test_ids, seq_test_users = load_test_sequences(test_path)
    X_seq, y_seq, seq_ids, seq_users = _limit_by_user(X_seq, seq_ids, seq_users, y_seq, args.per_user_limit)
    X_test_seq, seq_test_ids, seq_test_users = _limit_by_user(
        X_test_seq,
        seq_test_ids,
        seq_test_users,
        per_user_limit=args.per_user_limit,
    )

    if args.research_20260525:
        results = _run_research_candidates(
            X_train, y_train, users, X_test, test_ids,
            X_seq, X_test_seq, args,
        )
    elif args.improve_20260527:
        results = _run_improve_candidates(
            X_train, y_train, users, X_test, test_ids,
            X_seq, X_test_seq, args,
        )
    elif args.break_080:
        results = _run_break_080_candidates(
            X_train, y_train, users, X_test, test_ids,
            X_seq, X_test_seq, args,
        )
    elif args.today_20260525:
        results = _run_today_candidates(
            X_train, y_train, users, X_test, test_ids,
            X_seq, X_test_seq, args,
        )
    elif args.plateau_20260520:
        results = _run_plateau_tree_candidates(
            X_train,
            y_train,
            users,
            X_test,
            test_ids,
            X_seq,
            X_test_seq,
            args,
        )
        results.append(_run_rocket_candidate(X_seq, y_seq, seq_users, X_test_seq, seq_test_ids, args))
    elif args.targeted_20260522:
        results = _run_targeted_candidates(
            X_train, y_train, users, X_test, test_ids,
            X_seq, X_test_seq, args,
        )
    elif args.next_20260524:
        results = _run_next_candidates(
            X_train, y_train, users, X_test, test_ids,
            X_seq, X_test_seq, args,
        )
    elif args.daily_20260520:
        results = _run_daily_tree_candidates(X_train, y_train, users, X_test, test_ids, args)
        results.append(
            _run_cnn_candidate(
                X_seq,
                y_seq.to_numpy(),
                seq_users.to_numpy(),
                X_test_seq,
                seq_test_ids,
                args,
                name="cnn_improved_sequence",
                variant="improved",
                normalize=True,
            )
        )
    else:
        results = [
            _run_lgb_candidate("lgb_acc_no_smote", X_train, y_train, users, X_test, test_ids, args, use_smote=False),
            _run_lgb_candidate("lgb_acc_smote", X_train, y_train, users, X_test, test_ids, args, use_smote=True),
        ]

        if args.include_xgb:
            results.append(
                _run_xgb_candidate(
                    "xgb_acc_no_smote",
                    X_train,
                    y_train,
                    users,
                    X_test,
                    test_ids,
                    args,
                    use_smote=False,
                    metric="accuracy",
                )
            )
        results.append(
            _run_cnn_candidate(
                X_seq,
                y_seq.to_numpy(),
                seq_users.to_numpy(),
                X_test_seq,
                seq_test_ids,
                args,
                name="cnn_raw_sequence",
                variant="small",
                normalize=False,
            )
        )

    _print_summary(results)


if __name__ == "__main__":
    main()
