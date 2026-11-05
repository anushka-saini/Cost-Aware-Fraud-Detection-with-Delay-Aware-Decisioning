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

    python scripts/v3_phase2.py
"""

import json

import numpy as np

from fraud.config import get_settings
from fraud.evaluation import cost_report
from fraud.protocol import calibrate_and_threshold
from fraud.registry import load_bundle
from fraud.results import _jsonable, record

SOURCE = "scripts/v3_phase2.py"
settings = get_settings()
RUN_DIR = settings.models / "runs" / "v3"
OUT_DIR = settings.results / "v3"
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

    chosen = calibrate_and_threshold(saved[proposed], y, amounts, days)
    probs = chosen["calibrator"].transform(saved[proposed])
    safe, plain = chosen["thresholds_eval_2"]["robust"], chosen["thresholds_eval_2"]["plain"]

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
    (OUT_DIR / "phase2.json").write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    record("v3_phase2_decisions", payload, SOURCE)

    print(f"safe point: review >= {safe['t_review']:.6g}, block >= {safe['t_block']:.4g}, cost {safe['total_cost']:,.0f}, "
          f"fraud allowed {safe['fraud_allowed']}, legit reviewed {safe['legit_reviewed_pct']:.2%}, "
          f"wrongful blocks {safe['legit_blocked']}")
    print(f"cheapest point: cost {plain['total_cost']:,.0f}")
    for name, point in baselines.items():
        print(f"baseline {name}: cost {point['total_cost']:,.0f}")
    cliff = payload["cliff_safe"]
    if cliff:
        print(f"cliff: next fraud at p={cliff['cliff']:.6g}, odds x{cliff['odds_factor_to_cliff']:.2f} above the "
              f"review threshold; cost just past it {cliff['cost_just_past_cliff']:,.0f}")


if __name__ == "__main__":
    main()
