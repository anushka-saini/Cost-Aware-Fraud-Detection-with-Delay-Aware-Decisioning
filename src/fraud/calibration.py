"""Probability calibration, fitted on eval_2 and never on the held-out set."""

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

EPS = 1e-12


def _logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


class IsotonicCalibrator:
    """Monotone, non-parametric map from raw score to probability."""

    method = "isotonic"

    def fit(self, scores, y):
        self._iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self._iso.fit(np.asarray(scores, dtype=np.float64), np.asarray(y, dtype=np.float64))
        return self

    def transform(self, scores):
        return self._iso.predict(np.asarray(scores, dtype=np.float64))

    @property
    def n_levels(self) -> int:
        """Number of distinct output values (a step function: few levels means coarse)."""
        return int(len(np.unique(self._iso.y_thresholds_)))


class PlattCalibrator:
    """Logistic regression on the logit of the raw score (two parameters)."""

    method = "platt"

    def fit(self, scores, y):
        self._lr = LogisticRegression(C=1e6, max_iter=1000)
        self._lr.fit(_logit(scores).reshape(-1, 1), np.asarray(y))
        return self

    def transform(self, scores):
        return self._lr.predict_proba(_logit(scores).reshape(-1, 1))[:, 1]


class IdentityCalibrator:
    method = "none"

    def fit(self, scores, y):
        return self

    def transform(self, scores):
        return np.asarray(scores, dtype=np.float64)


def expected_calibration_error(y, probs, bins: int = 15) -> float:
    """ECE over equal-mass bins (equal-width bins are almost all empty at a 0.07% base rate)."""
    y, probs = np.asarray(y, dtype=np.float64), np.asarray(probs, dtype=np.float64)
    order = np.argsort(probs, kind="stable")
    error = 0.0
    for chunk in np.array_split(order, bins):
        if len(chunk):
            error += len(chunk) / len(y) * abs(y[chunk].mean() - probs[chunk].mean())
    return float(error)


def calibration_metrics(y, probs) -> dict:
    return {"brier": float(brier_score_loss(y, np.clip(probs, 0, 1))),
            "ece": expected_calibration_error(y, probs),
            "mean_predicted": float(np.mean(probs)), "base_rate": float(np.mean(y))}


def fit_calibrator(scores, y, method: str = "isotonic"):
    cls = {"isotonic": IsotonicCalibrator, "platt": PlattCalibrator, "none": IdentityCalibrator}[method]
    return cls().fit(scores, y)
