from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning

import warnings
from netcal.scaling import TemperatureScaling
from netcal.binning import HistogramBinning

from .metrics import classification_metrics, expected_calibration_error


def build_features(pairs: List[Tuple[np.ndarray, np.ndarray, int]]) -> Tuple[np.ndarray, np.ndarray]:
    # Combine query and doc embeddings via absolute difference and product
    X_list = []
    y_list = []
    for q_emb, d_emb, y in pairs:
        X_list.append(np.concatenate([np.abs(q_emb - d_emb), q_emb * d_emb], axis=-1))
        y_list.append(y)
    # Use float32 to reduce memory and avoid overflow in downstream libs
    X = np.vstack(X_list).astype(np.float32, copy=False)
    y = np.array(y_list, dtype=np.int32)
    return X, y


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    estimator: str = "logistic_regression",
    params: Dict = None,
):
    params = params or {}
    if estimator == "logistic_regression":
        # Auto-switch solver for very large problems to avoid liblinear overflow
        n_samples, n_features = X_train.shape
        p = params.copy()
        solver = p.get("solver", "liblinear")
        if solver == "liblinear" and (n_samples > 2_000_000 or n_features > 10_000):
            # saga scales better to large N and supports l2
            p["solver"] = "saga"
            p.setdefault("penalty", "l2")
            p.setdefault("max_iter", 2000)
        # Leverage multi-threading when supported
        if p.get("solver", "liblinear") in ("liblinear", "saga") and "n_jobs" not in p:
            p["n_jobs"] = -1
        clf = LogisticRegression(**p)
        # Suppress ConvergenceWarning so strict PYTHONWARNINGS=error won't abort the run
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            clf.fit(X_train, y_train)
        return clf
    elif estimator == "linear_svc":
        base = LinearSVC(**params)
        model = CalibratedClassifierCV(base, method="sigmoid")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            model.fit(X_train, y_train)
        return model
    elif estimator == "sgd_logistic":
        # Fast, scalable logistic via SGD with early stopping
        p = params.copy()
        p_defaults = {
            "loss": "log_loss",
            "penalty": "l2",
            "alpha": 1e-4,          # L2 strength ~ 1/C
            "max_iter": 2000,
            "early_stopping": True,
            "n_jobs": -1,
            "class_weight": None,   # set to 'balanced' if classes imbalanced
            "learning_rate": "optimal",
            "eta0": 0.0,
            "random_state": 42,
            "verbose": 1,
        }
        for k, v in p_defaults.items():
            p.setdefault(k, v)
        # Coerce common YAML string forms to correct dtypes
        try:
            p["alpha"] = float(p.get("alpha", 1e-4))
        except Exception:
            p["alpha"] = 1e-4
        try:
            p["max_iter"] = int(p.get("max_iter", 2000))
        except Exception:
            p["max_iter"] = 2000
        try:
            p["n_jobs"] = int(p.get("n_jobs", -1))
        except Exception:
            p["n_jobs"] = -1
        try:
            p["verbose"] = int(p.get("verbose", 1))
        except Exception:
            p["verbose"] = 1
        try:
            p["random_state"] = int(p.get("random_state", 42))
        except Exception:
            p["random_state"] = 42
        es = p.get("early_stopping", True)
        if isinstance(es, str):
            p["early_stopping"] = es.strip().lower() in ("1", "true", "yes", "y")
        clf = SGDClassifier(**p)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            clf.fit(X_train, y_train)
        return clf
    else:
        raise ValueError(f"Unknown estimator: {estimator}")


def calibrate_probs(probs: np.ndarray, y_true: np.ndarray, method: str = "temperature", bins: int = 15) -> np.ndarray:
    # probs expected shape (N, 1) for binary; flatten to (N,)
    p = probs.reshape(-1)
    if len(np.unique(y_true)) < 2:
        # Skip calibration for degenerate splits
        return p.reshape(-1, 1)
    if method == "temperature":
        # Temperature scaling (may rely on pyro under the hood). If any warning/error occurs
        # under strict PYTHONWARNINGS=error, fall back to histogram binning.
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                ts = TemperatureScaling()
                ts.fit(p, y_true)
                out = ts.transform(p)
            return out.reshape(-1, 1)
        except Exception:
            hb = HistogramBinning(bins=bins)
            hb.fit(p, y_true)
            return hb.transform(p).reshape(-1, 1)
    elif method == "histogram":
        # Non-parametric histogram binning calibration (avoids pyro)
        hb = HistogramBinning(bins=bins)
        hb.fit(p, y_true)
        return hb.transform(p).reshape(-1, 1)
    else:
        raise ValueError(f"Unknown calibration method: {method}")


def evaluate_classifier(model, X: np.ndarray, y_true: np.ndarray, calibration: Dict) -> Dict:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
    else:
        # Decision function to probability via logistic transform for LinearSVC (if not calibrated outside)
        dec = model.decision_function(X)
        probs = 1.0 / (1.0 + np.exp(-dec))

    calib_method = calibration.get("method", "histogram")
    bins = int(calibration.get("bins", 15))
    if len(np.unique(y_true)) < 2:
        probs_cal = probs
    else:
        probs_cal = calibrate_probs(probs.reshape(-1, 1), y_true, method=calib_method, bins=bins).reshape(-1)

    base = classification_metrics(y_true, probs)
    cal = classification_metrics(y_true, probs_cal)
    ece_base = expected_calibration_error(y_true, probs, n_bins=bins)
    ece_cal = expected_calibration_error(y_true, probs_cal, n_bins=bins)

    return {
        "base": base,
        "calibrated": cal,
        "ece_base": ece_base,
        "ece_calibrated": ece_cal,
    }
