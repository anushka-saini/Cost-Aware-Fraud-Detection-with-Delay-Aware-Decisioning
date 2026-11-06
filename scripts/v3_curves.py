"""Curves for the dashboard's Model lab page.

Everything here comes from the days 11-13 scores that `v3_phase1.py compare`
saved (models/runs/v3/validation_scores.npz). The held-out days are not
touched again.

- PR and ROC curves on days 11-13 for the main models.
- Calibration on day 13: Platt fitted on days 11-12, checked on day 13,
  the same out-of-sample check Phase 2 used.

The curves are thinned to a fixed grid so the page stays small:
precision is interpolated (the best precision at that recall or higher),
and the ROC curve is read at false-positive rates spaced on a log scale,
because at a 0.07% fraud rate everything interesting happens below 1%.

    python scripts/v3_curves.py
"""

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

from fraud.calibration import calibration_metrics, fit_calibrator
from fraud.config import get_settings
from fraud.results import load_metrics, record

SOURCE = "scripts/v3_curves.py"
MODELS = ["lightgbm", "v2_deployed", "catboost", "xgboost", "mlp"]
CALIBRATION_MODELS = ["lightgbm", "v2_deployed"]
CHECK_DAY = 13
RECALL_GRID = np.round(np.linspace(0, 1, 51), 2)
FPR_GRID = np.logspace(-5, 0, 41)
# Bins by predicted probability. Equal-mass bins would put almost every bin
# at zero fraud, because fraud is so rare.
PROBABILITY_EDGES = [0, 1e-4, 1e-3, 1e-2, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]


def pr_points(y, scores):
    precision, recall, _ = precision_recall_curve(y, scores)
    points = []
    for r in RECALL_GRID:
        best = precision[recall >= r].max()
        points.append({"recall": float(r), "precision": float(best)})
    return points


def roc_points(y, scores):
    fpr, tpr, _ = roc_curve(y, scores)
    points = []
    for f in FPR_GRID:
        points.append({"fpr": float(f), "tpr": float(tpr[fpr <= f].max())})
    return points


def calibration_bins(y, probs):
    bins = []
    for low, high in zip(PROBABILITY_EDGES[:-1], PROBABILITY_EDGES[1:], strict=True):
        inside = (probs >= low) & (probs < high) if high < 1 else (probs >= low)
        n = int(inside.sum())
        if n == 0:
            continue
        bins.append({"from": low, "to": high, "n": n, "n_fraud": int(y[inside].sum()),
                     "mean_predicted": float(probs[inside].mean()), "observed_rate": float(y[inside].mean())})
    return bins


def main():
    settings = get_settings()
    scores = np.load(settings.models / "runs" / "v3" / "validation_scores.npz")
    y, days = scores["y"], scores["day"]
    labels = {k: v["label"] for k, v in load_metrics()["v3_validation"]["leaderboard"].items()}

    curves = {}
    for name in MODELS:
        raw = scores[name].astype(np.float64)
        curves[name] = {"label": labels[name], "pr": pr_points(y, raw), "roc": roc_points(y, raw)}
        print(f"{name}: curves done", flush=True)

    fit, check = days != CHECK_DAY, days == CHECK_DAY
    calibration = {}
    for name in CALIBRATION_MODELS:
        raw = scores[name].astype(np.float64)
        calibrator = fit_calibrator(raw[fit], y[fit], "platt")
        probs = calibrator.transform(raw[check])
        calibration[name] = {"label": labels[name], "bins": calibration_bins(y[check], probs),
                             **calibration_metrics(y[check], probs)}
        print(f"{name}: day-13 Brier {calibration[name]['brier']:.6f}", flush=True)

    record("v3_curves", {
        "curves_on": "days 11-13 (validation)",
        "calibration_on": "day 13; Platt fitted on days 11-12",
        "pr_note": "precision is interpolated: the best precision at this recall or higher",
        "roc_note": "true-positive rate at false-positive rates spaced on a log scale",
        "n_rows": int(len(y)), "n_fraud": int(y.sum()),
        "curves": curves, "calibration": calibration,
    }, SOURCE)
    print("recorded v3_curves")


if __name__ == "__main__":
    main()
