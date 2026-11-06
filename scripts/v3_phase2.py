"""Version 3, Phase 2: safer decisions for the proposed model. Days 11-13 only.

What it does:

1. Calibrates the proposed model on days 11-13 (Platt or isotonic, whichever
   generalises better from days 11-12 to day 13).
2. Picks the REVIEW / BLOCK thresholds with a safety margin: the pair whose
   worst-case cost is lowest when the calibrated odds are off by up to 2x
   either way.
3. Measures how far the review threshold is from the "cost cliff": the point
   where raising it would let the next fraud through.
4. Adds baseline costs for context: block nothing, review everything, V2 as deployed.
5. Repeats the whole recipe out of sample (fit on days 11-12, cost on day 13)
   for both calibration methods.
6. Isotonic calibration gives many transactions a probability of exactly 0, so
   a tiny review threshold means "review anything above zero". The real safety
   margin is then how far the lowest-scoring fraud sits above that zero region,
   measured on the raw model score. If that margin on day 13 is smaller than
   the 2x the thresholds must survive, isotonic is not safe enough and Platt
   is used instead.
7. Saves the calibrator and thresholds to models/runs/v3/v3_final.joblib, so
   the held-out test (v3_phase3.py) uses exactly what was approved.

    python scripts/v3_phase2.py
"""

import json

import joblib
import numpy as np

from fraud.calibration import calibration_metrics, fit_calibrator
from fraud.config import get_settings
from fraud.evaluation import cost_report
from fraud.protocol import THRESHOLD_ODDS_MARGIN, calibrate_and_threshold, choose_thresholds
from fraud.registry import load_bundle
from fraud.results import _jsonable, record

SOURCE = "scripts/v3_phase2.py"
settings = get_settings()
RUN_DIR = settings.models / "runs" / "v3"
OUT_DIR = settings.results / "v3"
CHECK_DAY = 13
POINT_KEYS = ("t_review", "t_block", "total_cost", "fraud_allowed", "fraud_allowed_amount", "fraud_reviewed",
              "fraud_blocked", "legit_reviewed", "legit_reviewed_pct", "legit_blocked", "block_precision",
              "block_recall", "block_f1")


def odds(p):
    return p / (1 - p)


def distance_to_cliff(y, probs, amounts, t_review, t_block):
    """How much room the review threshold has before a fraud slips through.

    Every fraud with probability >= t_review is currently caught (reviewed or
    blocked). The cliff is the lowest of those probabilities: move t_review
    past it and that fraud is allowed, costing its whole amount.
    """
    fraud_probs = probs[y == 1]
    caught = fraud_probs[fraud_probs >= t_review]
    if len(caught) == 0:
        return None
    cliff = float(caught.min())
    just_past = cost_report(y, probs, amounts, np.nextafter(cliff, 1.0), t_block)
    at_choice = cost_report(y, probs, amounts, t_review, t_block)
    return {
        "review_threshold": float(t_review),
        "cliff": cliff,
        "odds_factor_to_cliff": float(odds(cliff) / odds(t_review)),
        "cost_at_threshold": at_choice["total_cost"],
        "cost_just_past_cliff": just_past["total_cost"],
        "extra_fraud_allowed_past_cliff": just_past["fraud_allowed"] - at_choice["fraud_allowed"],
    }


def day13_check(raw, y, amounts, days, method):
    """The Phase 2 recipe fitted on days 11-12 and measured on day 13."""
    fit, check = days != CHECK_DAY, days == CHECK_DAY
    calibrator = fit_calibrator(raw[fit], y[fit], method)
    probs = calibrator.transform(raw)
    safe = choose_thresholds(y[fit], probs[fit], amounts[fit])["robust"]
    point = cost_report(y[check], probs[check], amounts[check], safe["t_review"], safe["t_block"])
    result = {k: point[k] for k in POINT_KEYS}
    result["day_13_frauds_given_zero"] = int(((probs[check] == 0) & (y[check] == 1)).sum())
    return result


def zero_region_margin(raw, probs, y):
    """How far the lowest-scoring fraud sits above the scores that calibrate to 0.

    ``zero_boundary`` is the lowest raw score that gets a probability above 0.
    The margin is in odds of the raw score: 2.0 means the lowest fraud's score
    would have to halve (in odds) before it fell into the zero region.
    """
    positive = raw[probs > 0]
    if len(positive) == 0:
        return None
    boundary = float(positive.min())
    lowest_fraud = float(raw[y == 1].min())
    between = (raw >= boundary) & (raw < lowest_fraud) & (y == 0)
    return {
        "zero_boundary_raw_score": boundary,
        "lowest_fraud_raw_score": lowest_fraud,
        "odds_factor": float(odds(lowest_fraud) / odds(boundary)),
        "legit_rows_between": int(between.sum()),
    }


def main():
    validation = json.loads((OUT_DIR / "validation.json").read_text(encoding="utf-8"))
    proposed = validation["proposed_v3"]
    if proposed == "stacked":
        # Its days 11-12 scores are in sample (the stacker was fitted on them),
        # so calibrating and thresholding on days 11-13 would not be honest.
        raise SystemExit("The stacked ensemble was proposed: stop and agree how to calibrate it before Phase 2.")
    saved = np.load(RUN_DIR / "validation_scores.npz")
    y, amounts, days = saved["y"], saved["amount"], saved["day"]
    print(f"proposed V3 model: {proposed}; days 11-13: {len(y):,} rows, {int(y.sum())} fraud")

    raw = saved[proposed]
    chosen = calibrate_and_threshold(raw, y, amounts, days)
    probs = chosen["calibrator"].transform(raw)

    # Isotonic's zero region: is the lowest fraud far enough above it?
    margins = None
    if chosen["calibration"]["method"] == "isotonic":
        fit, check = days != CHECK_DAY, days == CHECK_DAY
        early = fit_calibrator(raw[fit], y[fit], "isotonic")
        margins = {
            "note": ("isotonic gives many rows a probability of exactly 0, so the odds-based cliff is not a useful "
                     "margin; this measures the gap on the raw model score instead"),
            "required_odds_factor": THRESHOLD_ODDS_MARGIN,
            "days_11_13_in_sample": zero_region_margin(raw, probs, y),
            "day_13_out_of_sample": zero_region_margin(raw[check], early.transform(raw[check]), y[check]),
        }
        out_of_sample = margins["day_13_out_of_sample"]
        if out_of_sample is None or out_of_sample["odds_factor"] < THRESHOLD_ODDS_MARGIN:
            # Not safe enough: switch to Platt and choose the thresholds again.
            calibrator = fit_calibrator(raw, y, "platt")
            probs = calibrator.transform(raw)
            chosen = {
                "calibrator": calibrator,
                "calibration": {**chosen["calibration"], "method": "platt",
                                "isotonic_rejected_because": (
                                    f"its day-13 zero-region margin is x{out_of_sample['odds_factor']:.2f}, below the "
                                    f"x{THRESHOLD_ODDS_MARGIN:.0f} the thresholds must survive"
                                    if out_of_sample else "no day-13 margin could be measured"),
                                "eval_2_after": calibration_metrics(y, probs)},
                "thresholds_eval_2": choose_thresholds(y, probs, amounts),
            }
    safe, plain = chosen["thresholds_eval_2"]["robust"], chosen["thresholds_eval_2"]["plain"]
    print(f"calibration: {chosen['calibration']['method']}")

    # V2 exactly as deployed: its own calibrator and thresholds.
    v2 = load_bundle("fraud_scorer", "v2")
    v2_probs = v2["calibrator"].transform(saved["v2_deployed"])
    v2_point = cost_report(y, v2_probs, amounts, v2["thresholds"]["t_review"], v2["thresholds"]["t_block"])

    baselines = {
        "block_nothing": cost_report(y, probs, amounts, 2.0, 2.0),           # everything allowed
        "review_everything": cost_report(y, probs, amounts, 0.0, 2.0),       # everything reviewed, nothing blocked
        "v2_deployed": v2_point,
    }
    payload = {
        "data": "days 11-13 (the same days the thresholds are chosen on, so these costs are in-sample)",
        "proposed_model": proposed,
        "calibration": chosen["calibration"],
        "threshold_rule": "lowest worst-case cost when the calibrated odds are scaled anywhere from 1/2x to 2x",
        "safe_point": {k: safe[k] for k in POINT_KEYS},
        "safe_point_worst_case_cost": safe["worst_case_cost_within_margin"],
        "plain_cheapest_point": {k: plain[k] for k in POINT_KEYS},
        "cliff_safe": distance_to_cliff(y, probs, amounts, safe["t_review"], safe["t_block"]),
        "cliff_plain": distance_to_cliff(y, probs, amounts, plain["t_review"], plain["t_block"]),
        "robustness": safe["robustness"],
        "baselines": {name: {k: point[k] for k in POINT_KEYS} for name, point in baselines.items()},
        "v2_cliff": distance_to_cliff(y, v2_probs, amounts, v2["thresholds"]["t_review"], v2["thresholds"]["t_block"]),
    }
    # Out of sample: the same recipe, fitted on days 11-12 and measured on day 13.
    payload["day_13_check"] = {
        "note": "calibration and thresholds fitted on days 11-12, cost measured on day 13",
        "proposed": {m: day13_check(raw, y, amounts, days, m) for m in ("isotonic", "platt")},
        "v2_deployed": {m: day13_check(saved["v2_deployed"], y, amounts, days, m) for m in ("isotonic", "platt")},
    }
    if margins:
        payload["isotonic_zero_region_margin"] = margins

    joblib.dump({"model": proposed, "calibrator": chosen["calibrator"], "calibration": chosen["calibration"]["method"],
                 "thresholds": {"t_review": float(safe["t_review"]), "t_block": float(safe["t_block"])},
                 "fitted_on_days": [11, 12, 13], "source": SOURCE}, RUN_DIR / "v3_final.joblib")

    (OUT_DIR / "phase2.json").write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    record("v3_phase2_decisions", payload, SOURCE)

    print(f"safe point: review >= {safe['t_review']:.6g}, block >= {safe['t_block']:.4g}, cost {safe['total_cost']:,.0f}, "
          f"fraud allowed {safe['fraud_allowed']}, legit reviewed {safe['legit_reviewed_pct']:.2%}, "
          f"wrongful blocks {safe['legit_blocked']}")
    print(f"cheapest point: cost {plain['total_cost']:,.0f}")
    for name, point in baselines.items():
        print(f"baseline {name}: cost {point['total_cost']:,.0f}")
    for model in ("proposed", "v2_deployed"):
        for method, point in payload["day_13_check"][model].items():
            print(f"day 13 ({model}, {method}): cost {point['total_cost']:,.0f}, fraud allowed {point['fraud_allowed']}, "
                  f"frauds given 0: {point['day_13_frauds_given_zero']}")
    for name, margin in (margins or {}).items():
        if isinstance(margin, dict):
            print(f"zero-region margin ({name}): lowest fraud odds x{margin['odds_factor']:.2f} above the zero "
                  f"boundary, {margin['legit_rows_between']:,} legit rows in between")
    cliff = payload["cliff_safe"]
    if cliff:
        print(f"cliff: next fraud at p={cliff['cliff']:.6g}, odds x{cliff['odds_factor_to_cliff']:.2f} above the "
              f"review threshold; cost just past it {cliff['cost_just_past_cliff']:,.0f}")


if __name__ == "__main__":
    main()
