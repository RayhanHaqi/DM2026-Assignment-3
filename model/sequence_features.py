import numpy as np
import pandas as pd


FEATURE_COLS = ["mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z"]


def optional_import_error_message(package):
    return f"{package} is required for this sequence feature path. Install with: pip install {package}"


def build_extra_channels(X_seq):
    X = np.asarray(X_seq, dtype=np.float32)
    mean_mag = np.sqrt(np.sum(X[:, :, :3] ** 2, axis=2, keepdims=True))
    std_mag = np.sqrt(np.sum(X[:, :, 3:6] ** 2, axis=2, keepdims=True))
    return np.concatenate([X, mean_mag, std_mag], axis=2)


def build_catch22_features(X_seq):
    try:
        import pycatch22
    except ImportError as exc:
        raise ImportError(optional_import_error_message("pycatch22")) from exc

    X = build_extra_channels(X_seq)
    channel_names = FEATURE_COLS + ["mean_mag", "std_mag"]
    rows = []
    for sample in X:
        row = {}
        for channel_i, channel_name in enumerate(channel_names):
            result = pycatch22.catch22_all(sample[:, channel_i])
            for feature_name, value in zip(result["names"], result["values"]):
                row[f"catch22_{channel_name}_{feature_name}"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows).replace([np.inf, -np.inf], 0.0).fillna(0.0)


def fit_minirocket_ridge(X_seq, y):
    import_error = None
    MiniRocketMultivariate = None
    for path, package in [
        ("sktime.transformations.panel.rocket", "sktime"),
        ("aeon.transformations.collection.convolution_based", "aeon"),
    ]:
        try:
            module = __import__(path, fromlist=["MiniRocketMultivariate"])
            MiniRocketMultivariate = module.MiniRocketMultivariate
            break
        except ImportError:
            continue
    if MiniRocketMultivariate is None:
        raise ImportError(optional_import_error_message("sktime or aeon"))
    from sklearn.linear_model import RidgeClassifierCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    transformer = MiniRocketMultivariate(random_state=42)
    classifier = make_pipeline(StandardScaler(with_mean=False), RidgeClassifierCV(alphas=np.logspace(-3, 3, 7)))
    X_input = np.asarray(X_seq, dtype=np.float32)
    X_input = X_input.transpose(0, 2, 1)
    X_features = transformer.fit_transform(X_input)
    classifier.fit(X_features, np.asarray(y))
    return transformer, classifier


def predict_minirocket(model, X_seq):
    transformer, classifier = model
    X_input = np.asarray(X_seq, dtype=np.float32).transpose(0, 2, 1)
    X_features = transformer.transform(X_input)
    return classifier.predict(X_features).astype(int)


def to_aeon_collection(X_seq):
    X = np.asarray(X_seq, dtype=np.float32)
    if X.ndim != 3:
        raise ValueError(f"Expected sequence array with 3 dimensions, got shape {X.shape}")
    return X.transpose(0, 2, 1)


def decision_scores_to_proba(scores):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    totals = exp_scores.sum(axis=1, keepdims=True)
    return exp_scores / totals


def make_aeon_transformer(kind="minirocket", n_kernels=10000, n_jobs=1, random_state=42):
    if kind not in {"minirocket", "multirocket"}:
        raise ValueError(f"Unknown aeon transformer kind: {kind}")
    try:
        from aeon.transformations.collection.convolution_based import MiniRocket, MultiRocket
    except ImportError as exc:
        raise ImportError(optional_import_error_message("aeon")) from exc

    if kind == "minirocket":
        return MiniRocket(n_kernels=n_kernels, n_jobs=n_jobs, random_state=random_state)
    return MultiRocket(n_kernels=n_kernels, n_jobs=n_jobs, random_state=random_state)


def fit_aeon_rocket_ridge(X_seq, y, kind="minirocket", n_kernels=10000, n_jobs=1, random_state=42):
    from sklearn.linear_model import RidgeClassifierCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    transformer = make_aeon_transformer(
        kind=kind,
        n_kernels=n_kernels,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    X_features = transformer.fit_transform(to_aeon_collection(X_seq), np.asarray(y))
    classifier = make_pipeline(
        StandardScaler(with_mean=False),
        RidgeClassifierCV(alphas=np.logspace(-3, 3, 7)),
    )
    classifier.fit(X_features, np.asarray(y))
    return {"transformer": transformer, "classifier": classifier, "kind": kind}


def _transform_aeon_rocket(model, X_seq):
    return model["transformer"].transform(to_aeon_collection(X_seq))


def predict_aeon_rocket(model, X_seq):
    X_features = _transform_aeon_rocket(model, X_seq)
    return model["classifier"].predict(X_features).astype(int)


def predict_aeon_rocket_proba(model, X_seq):
    X_features = _transform_aeon_rocket(model, X_seq)
    classifier = model["classifier"]
    if hasattr(classifier, "predict_proba"):
        proba = classifier.predict_proba(X_features)
    else:
        proba = decision_scores_to_proba(classifier.decision_function(X_features))
    return proba, classifier.classes_.astype(int)
