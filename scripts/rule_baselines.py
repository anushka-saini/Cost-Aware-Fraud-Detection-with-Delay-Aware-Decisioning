"""Rule-engine baselines: what simple hand-written rules would do.

A fraud team without a model would start with rules. This script builds four
of them and scores them next to V3, so the model has to earn its place.

    python scripts/rule_baselines.py

The rules are designed on days 5-10 only (the training and model-selection
days). Days 11-13 and 14-16 are only scored: nothing about the rules, and
nothing about V3, is chosen from them. V3, its thresholds and its calibration
are not touched.

The rules:

1. paysim_flag      PaySim's own isFlaggedFraud column.
2. empties_account  A TRANSFER or CASH_OUT whose amount is the sender's whole
                    balance (to the cent).
3. large_amount     A TRANSFER or CASH_OUT of at least a cut-off amount.
4. rule_engine      Rules 1 to 3 together: the highest tier any of them gives.

Each rule sends its hits to BLOCK or REVIEW. A block costs 5 for every
legitimate hit and a review costs 1 for every hit, so blocking is cheaper only
when more than 80% of the hits are fraud. That share is measured on days 5-10.
The large_amount cut-off is the one that gives the rule engine its lowest
cost on days 5-10. The "best rule" is the rule with the lowest cost on days
5-10.

Costs follow the rest of the project: a fraud that is allowed costs its
amount, a wrongful block 5, a review 1.
"""

import numpy as np
import pandas as pd

from fraud.config import get_settings
from fraud.data import SPLITS, load_feature_frame
from fraud.evaluation import cost_report
from fraud.results import load_metrics, record

SOURCE = "scripts/rule_baselines.py"
DESIGN_DAYS = SPLITS["train"] + SPLITS["eval_1"]           # 5-10
SCORED = {"days_11_13": SPLITS["eval_2"], "days_14_16": SPLITS["held_out"]}
COLUMNS = ["day_bucket", "isFraud", "isFlaggedFraud", "amount", "oldbalanceOrg", "newbalanceOrig",
           "type_TRANSFER", "type_CASH_OUT"]

TIERS = {"ALLOW": 0, "REVIEW": 1, "BLOCK": 2}
# cost_report works on probabilities and two thresholds. A tier is turned into
# a stand-in probability that lands in the right band.
STAND_IN_PROBABILITY = np.array([0.0, 0.5, 1.0])
STAND_IN_THRESHOLDS = {"t_review": 0.25, "t_block": 0.75}

RULES = {
    "paysim_flag": {"label": "PaySim's own flag (isFlaggedFraud)",
                    "description": "the simulator's built-in flag"},
    "empties_account": {"label": "Transfer or cash-out of the whole balance",
                        "description": "a TRANSFER or CASH_OUT whose amount equals the sender's whole balance "
                                       "(to the cent, balance above 0)"},
    "large_amount": {"label": "Large transfer or cash-out",
                     "description": "a TRANSFER or CASH_OUT of at least the cut-off amount"},
    "rule_engine": {"label": "Rule engine (all three rules)",
                    "description": "the flag and the whole-balance rule with their tier, then the large-amount "
                                   "rule; each transaction gets the highest tier any rule gives"},
}


# ---------------------------------------------------------------------------
# The rules
# ---------------------------------------------------------------------------

def risky_type(d: pd.DataFrame) -> pd.Series:
    """Fraud in PaySim only happens in these two types."""
    return (d["type_TRANSFER"] == 1) | (d["type_CASH_OUT"] == 1)


def hits(d: pd.DataFrame, rule: str, cut_off: float | None = None) -> np.ndarray:
    if rule == "paysim_flag":
        hit = d["isFlaggedFraud"] == 1
    elif rule == "empties_account":
        hit = risky_type(d) & (d["oldbalanceOrg"] > 0) & ((d["amount"] - d["oldbalanceOrg"]).abs() < 0.01)
    elif rule == "large_amount":
        hit = risky_type(d) & (d["amount"] >= cut_off)
    else:
        raise ValueError(rule)
    return hit.to_numpy()


def tier_for(d: pd.DataFrame, hit: np.ndarray) -> str:
    """BLOCK when more than 80% of the hits are fraud on these days, else REVIEW."""
    n_hits = int(hit.sum())
    if n_hits == 0:
        return "REVIEW"
    precision = d["isFraud"].to_numpy()[hit].mean()
    return "BLOCK" if precision > 0.8 else "REVIEW"


def apply(d: pd.DataFrame, design: dict, rule: str) -> np.ndarray:
    """The tier (0 allow, 1 review, 2 block) each row gets from a rule."""
    tiers = np.zeros(len(d), dtype=int)
    parts = ["paysim_flag", "empties_account", "large_amount"] if rule == "rule_engine" else [rule]
    for part in parts:
        hit = hits(d, part, design["cut_off"])
        tiers = np.where(hit, np.maximum(tiers, TIERS[design["tiers"][part]]), tiers)
    return tiers


# ---------------------------------------------------------------------------
# Measuring
# ---------------------------------------------------------------------------

def report(d: pd.DataFrame, tiers: np.ndarray) -> dict:
    y = d["isFraud"].to_numpy()
    point = cost_report(y, STAND_IN_PROBABILITY[tiers], d["amount"].to_numpy(), **STAND_IN_THRESHOLDS)
    return describe(point, n_rows=len(d), n_fraud=int(y.sum()))


def describe(point: dict, n_rows: int, n_fraud: int) -> dict:
    """The measures shown for every rule, from a cost_report result."""
    fraud_flagged = point["fraud_reviewed"] + point["fraud_blocked"]
    flagged = fraud_flagged + point["legit_reviewed"] + point["legit_blocked"]
    return {
        "n_rows": n_rows, "n_fraud": n_fraud,
        "recall": fraud_flagged / n_fraud if n_fraud else None,
        "precision": fraud_flagged / flagged if flagged else None,
        "total_cost": point["total_cost"],
        "fraud_allowed": point["fraud_allowed"], "fraud_allowed_amount": point["fraud_allowed_amount"],
        "review_rate": (point["fraud_reviewed"] + point["legit_reviewed"]) / n_rows,
        "block_rate": (point["fraud_blocked"] + point["legit_blocked"]) / n_rows,
        "legit_blocked": point["legit_blocked"], "legit_reviewed_pct": point["legit_reviewed_pct"],
    }


def choose_cut_off(design: pd.DataFrame, tiers: dict) -> dict:
    """The large_amount cut-off that gives the rule engine its lowest cost on the design days.
    Candidates: every percentile (0 to 99) of TRANSFER and CASH_OUT amounts on those days."""
    amounts = design.loc[risky_type(design), "amount"]
    candidates = np.unique(np.round(np.percentile(amounts, np.arange(0, 100)), 2))
    costs = []
    for cut in candidates:
        trial = {"cut_off": float(cut), "tiers": {**tiers, "large_amount": "REVIEW"}}
        costs.append(report(design, apply(design, trial, "rule_engine"))["total_cost"])
    best = int(np.argmin(costs))
    return {"cut_off": float(candidates[best]), "engine_cost_on_design_days": float(costs[best]),
            "candidates_tried": len(candidates),
            "candidate_rule": "every percentile 0-99 of TRANSFER and CASH_OUT amounts on days 5-10"}


def v3_reference(metrics: dict, counts: dict) -> dict:
    """V3's own figures on the same days, from the sections that already hold them."""
    phase2, held = metrics["v3_phase2_decisions"], metrics["v3_held_out"]
    return {
        "days_11_13": {**describe(phase2["safe_point"], **counts["days_11_13"]),
                       "note": "in-sample for V3: its calibration and thresholds were chosen on these days",
                       "source": phase2["_source"]},
        "days_14_16": {**describe(held["v3"]["operating_point"], **counts["days_14_16"]),
                       "note": "the one-time held-out test", "source": held["_source"]},
    }


def main():
    frame = load_feature_frame(COLUMNS)
    design = frame[frame["day_bucket"].isin(DESIGN_DAYS)]

    tiers = {rule: tier_for(design, hits(design, rule)) for rule in ("paysim_flag", "empties_account")}
    search = choose_cut_off(design, tiers)
    tiers["large_amount"] = tier_for(design, hits(design, "large_amount", search["cut_off"]))
    plan = {"cut_off": search["cut_off"], "tiers": tiers}

    rules = []
    for key, text in RULES.items():
        row = {"key": key, **text, "tier": tiers.get(key, "highest of the three"),
               "design_days": report(design, apply(design, plan, key))}
        for name, days in SCORED.items():
            part = frame[frame["day_bucket"].isin(days)]
            row[name] = report(part, apply(part, plan, key))
        rules.append(row)
    best = min(rules, key=lambda r: r["design_days"]["total_cost"])

    counts = {name: {"n_rows": rules[0][name]["n_rows"], "n_fraud": rules[0][name]["n_fraud"]} for name in SCORED}
    settings = get_settings()
    payload = {
        "design_days": DESIGN_DAYS,
        "scored_days": SCORED,
        "how_rules_were_made": "designed on days 5-10 only; days 11-13 and 14-16 are scored for the "
                               "leaderboard and not used for any choice",
        "tier_rule": "BLOCK when more than 80% of a rule's hits on days 5-10 are fraud, otherwise REVIEW "
                     "(a block costs 5 per legitimate hit, a review 1 per hit)",
        "cut_off_search": search,
        "costs": {"fraud_allowed": "its amount", "wrongful_block": settings.cost_false_block,
                  "review": settings.cost_review},
        "rules": rules,
        "best_rule": best["key"],
        "best_rule_chosen_by": "lowest total cost on days 5-10",
        "v3": v3_reference(load_metrics(), counts),
        "caveat": "the whole-balance rule uses the same sender-balance bookkeeping that V3 depends on most "
                  "(see the balance-feature check), so its strength is a feature of the simulator",
    }
    record("rule_baselines", payload, SOURCE)

    print(f"tiers {tiers}, large-amount cut-off {search['cut_off']:,.2f}")
    print(f"{'rule':18s} {'days':10s} {'cost':>16s} {'missed':>6s} {'recall':>7s} {'prec':>7s} {'review':>7s} {'block':>7s}")
    shown = [*rules, {"key": "V3", **payload["v3"]}]
    for row in shown:
        for name in ("design_days", *SCORED):
            if name not in row:
                continue
            r = row[name]
            print(f"{row['key']:18s} {name:10s} {r['total_cost']:16,.2f} {r['fraud_allowed']:6d} "
                  f"{r['recall']:7.4f} {(r['precision'] or 0):7.4f} {r['review_rate']:7.4f} {r['block_rate']:7.4f}")
    print(f"best rule (days 5-10): {best['key']}")


if __name__ == "__main__":
    main()
