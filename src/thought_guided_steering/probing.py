"""Frozen linear risk probes and case-disjoint out-of-fold predictions."""

from dataclasses import asdict, dataclass
import warnings

import numpy as np
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .directions import finite_array


@dataclass(frozen=True)
class LinearProbe:
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    weight: tuple[float, ...]
    bias: float
    calibration_weight: float = 1.0
    calibration_bias: float = 0.0

    def __post_init__(self):
        values = [finite_array(x, 1) for x in (self.mean, self.scale, self.weight)]
        if len({x.shape for x in values}) != 1 or (values[1] <= 0).any():
            raise ValueError("Invalid probe dimensions or scale")
        if not np.isfinite([self.bias, self.calibration_weight, self.calibration_bias]).all():
            raise ValueError("Invalid probe intercept or calibration")
        for name in ("mean", "scale", "weight"):
            object.__setattr__(self, name, tuple(float(value) for value in getattr(self, name)))

    def predict(self, features) -> np.ndarray:
        features = finite_array(features, 2)
        if features.shape[1] != len(self.weight):
            raise ValueError("Feature width does not match frozen probe")
        raw = ((features - self.mean) / self.scale) @ self.weight + self.bias
        return expit(self.calibration_weight * raw + self.calibration_bias)

    def to_dict(self) -> dict:
        return asdict(self)


def fit_probe(features, labels, *, max_iter: int = 2000, regularization_c: float | None = None) -> LinearProbe:
    """Fit preprocessing on these training rows only; no regularization by default (Eq. 2)."""
    features = finite_array(features, 2)
    labels = np.asarray(labels)
    if labels.shape != (len(features),) or set(labels.tolist()) != {0, 1}:
        raise ValueError("Training requires aligned binary labels of both classes")
    if regularization_c is not None and (not np.isfinite(regularization_c) or regularization_c <= 0):
        raise ValueError("Explicit regularization C must be finite and positive")
    scaler = StandardScaler().fit(features)
    classifier = LogisticRegression(penalty=None if regularization_c is None else "l2",
                                    C=1.0 if regularization_c is None else regularization_c,
                                    solver="lbfgs", max_iter=max_iter)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        classifier.fit(scaler.transform(features), labels)
    return LinearProbe(tuple(scaler.mean_), tuple(scaler.scale_), tuple(classifier.coef_[0]),
                       float(classifier.intercept_[0]))


def grouped_oof_predictions(features, labels, case_ids, fold_ids, **fit_options) -> np.ndarray:
    """Use a supplied, fixed partition; no layer/threshold search or calibration fit."""
    features = finite_array(features, 2)
    labels = np.asarray(labels)
    cases = np.asarray(case_ids)
    folds = np.asarray(fold_ids)
    if any(a.shape != (len(features),) for a in (labels, cases, folds)):
        raise ValueError("Features, labels, cases and folds must align")
    if len(np.unique(folds)) < 2:
        raise ValueError("At least two folds required")
    for case in np.unique(cases):
        if len(np.unique(folds[cases == case])) != 1:
            raise ValueError("Case leakage across folds")
    predictions = np.full(len(features), np.nan)
    for fold in np.unique(folds):
        heldout = folds == fold
        probe = fit_probe(features[~heldout], labels[~heldout], **fit_options)
        predictions[heldout] = probe.predict(features[heldout])
    return predictions
