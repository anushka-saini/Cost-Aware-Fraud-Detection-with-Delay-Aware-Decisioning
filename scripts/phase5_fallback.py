"""What the stale-model fallback rule costs and buys on the held-out period.

The deployed model was trained on labels through day 7. Scoring days 14-16
means its labels are 7 to 9 days old. This script scores the held-out period
twice with the deployed bundle: with the fallback off, and with it forced on
(as if the label-maturity limit had been exceeded), and reports the difference.

    python scripts/phase5_fallback.py
"""

import os

from fraud.config import get_settings
from fraud.data import get_split, load_dense
from fraud.registry import load_bundle
from fraud.results import record
from fraud.serving import Scorer


def summarise(scored, y, amounts, cost_false_block, cost_review):
    fraud, legit = y == 1, y == 0
    allow, review, block = (scored["tier"] == t for t in ("ALLOW", "REVIEW", "BLOCK"))
    return {
        "total_cost": float(amounts[(allow & fraud).to_numpy()].sum() + cost_false_block * (block & legit).sum()
                            + cost_review * review.sum()),
        "fraud_allowed": int((allow & fraud).sum()),
        "legit_reviewed": int((review & legit).sum()),
        "legit_reviewed_pct": float((review & legit).sum() / legit.sum()),
        "fraud_reviewed": int((review & fraud).sum()), "fraud_blocked": int((block & fraud).sum()),
        "legit_blocked": int((block & legit).sum()),
        "escalated_by_fallback": int(scored["fallback_applied"].sum()),
        "escalated_fraud": int((scored["fallback_applied"] & fraud).sum()),
    }


def main():
    settings = get_settings()
    scorer = Scorer(load_bundle(os.environ.get("FRAUD_MODEL_NAME", "fraud_scorer")))
    X, y = get_split(load_dense(scorer.base_columns), "held_out", scorer.base_columns)
    y, amounts = y.reset_index(drop=True), X["amount"].to_numpy()
    X = X.reset_index(drop=True)

    args = (y, amounts, settings.cost_false_block, settings.cost_review)
    off = summarise(scorer.score_frame(X), *args)
    limit = scorer.staleness["max_label_age_days"]
    stale_day = scorer.staleness["labels_through_day"] + int(limit) + 1
    on = summarise(scorer.score_frame(X, current_day=stale_day), *args)

    payload = {
        "split": "held_out (days 14-16)", "model": {"name": scorer.name, "version": scorer.version},
        "rule": "when the newest training labels are older than max_label_age_days, a transaction the supervised "
                "model would ALLOW goes to REVIEW if its anomaly score is at or above the threshold",
        "staleness_config": scorer.staleness,
        "fallback_off": off, "fallback_on": on,
        "extra_cost": on["total_cost"] - off["total_cost"],
        "extra_legit_reviewed": on["legit_reviewed"] - off["legit_reviewed"],
        "fraud_rescued_from_allow": off["fraud_allowed"] - on["fraud_allowed"],
    }
    record("phase5_fallback", payload, "scripts/phase5_fallback.py")
    for name, r in (("off", off), ("on", on)):
        print(f"fallback {name}: cost {r['total_cost']:,.0f}, fraud allowed {r['fraud_allowed']}, "
              f"legit reviewed {r['legit_reviewed_pct']:.2%}, escalated {r['escalated_by_fallback']} "
              f"({r['escalated_fraud']} fraud)")


if __name__ == "__main__":
    main()
