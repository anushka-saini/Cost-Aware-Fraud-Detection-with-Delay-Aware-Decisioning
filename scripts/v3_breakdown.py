"""Where V3 catches fraud on the held-out days: a breakdown by transaction
type and by amount quartile.

    python scripts/v3_breakdown.py

This only describes the result that already exists. The deployed V3 model
(registry version v3, with its calibration and thresholds) scores days 14-16
again and the rows are counted in groups. Nothing is chosen or tuned here.
As a check, the totals must match the one-time held-out test in
results/metrics.json (section v3_held_out) exactly.

For each group:
  recall       share of fraud sent to review or blocked
  block recall share of fraud blocked straight away
  review rate  share of all transactions in the group sent to review
  block rate   share of all transactions in the group blocked

Amount quartiles are the quartiles of all held-out transaction amounts.
"""

import numpy as np
import pandas as pd

from fraud.data import SPLITS, load_dense
from fraud.evaluation import cost_report
from fraud.features import TRANSACTION_TYPES, TYPE_COLUMNS
from fraud.registry import load_bundle
from fraud.results import load_metrics, record
from fraud.serving import Scorer

SOURCE = "scripts/v3_breakdown.py"


def group_row(name: str, part: pd.DataFrame) -> dict:
    fraud = part["isFraud"] == 1
    flagged = part["tier"] != "ALLOW"
    n_fraud = int(fraud.sum())
    return {
        "group": name, "n_rows": int(len(part)), "n_fraud": n_fraud,
        "recall": float((flagged & fraud).sum() / n_fraud) if n_fraud else None,
        "block_recall": float(((part["tier"] == "BLOCK") & fraud).sum() / n_fraud) if n_fraud else None,
        "fraud_allowed": int((~flagged & fraud).sum()),
        "review_rate": float((part["tier"] == "REVIEW").mean()),
        "block_rate": float((part["tier"] == "BLOCK").mean()),
        "legit_blocked": int(((part["tier"] == "BLOCK") & ~fraud).sum()),
    }


def main():
    scorer = Scorer(load_bundle("fraud_scorer", "v3"))
    features = scorer.feature_columns
    frame = load_dense(features, extra=TYPE_COLUMNS)
    held = frame[frame["day_bucket"].isin(SPLITS["held_out"])].copy()

    scored = scorer.score_frame(held[features])
    held["probability"] = scored["fraud_probability"].to_numpy()
    held["tier"] = scored["tier"].to_numpy()
    held["type"] = [TRANSACTION_TYPES[i] for i in np.argmax(held[TYPE_COLUMNS].to_numpy(), axis=1)]

    # The check: this must be the same result as the one-time held-out test.
    point = cost_report(held["isFraud"], held["probability"], held["amount"], scorer.t_review, scorer.t_block)
    recorded = load_metrics()["v3_held_out"]["v3"]["operating_point"]
    for key in ("total_cost", "fraud_allowed", "fraud_reviewed", "fraud_blocked", "legit_reviewed", "legit_blocked"):
        if point[key] != recorded[key]:
            raise SystemExit(f"{key} is {point[key]}, but the held-out test recorded {recorded[key]}: stopping")

    by_type = [group_row(t, held[held["type"] == t]) for t in TRANSACTION_TYPES]

    edges = held["amount"].quantile([0, 0.25, 0.5, 0.75, 1]).to_numpy()
    held["quartile"] = pd.cut(held["amount"], bins=edges, labels=["Q1", "Q2", "Q3", "Q4"], include_lowest=True)
    by_amount = []
    for i, q in enumerate(["Q1", "Q2", "Q3", "Q4"]):
        row = group_row(q, held[held["quartile"] == q])
        by_amount.append({**row, "amount_from": float(edges[i]), "amount_to": float(edges[i + 1])})

    payload = {
        "model": {"name": scorer.name, "version": scorer.version,
                  "t_review": scorer.t_review, "t_block": scorer.t_block},
        "days": SPLITS["held_out"], "n_rows": int(len(held)), "n_fraud": int(held["isFraud"].sum()),
        "what_this_is": "a description of the existing held-out result, split into groups; nothing is tuned",
        "check": "totals match results/metrics.json v3_held_out exactly",
        "quartiles_of": "all held-out transaction amounts",
        "measures": {"recall": "share of fraud sent to review or blocked",
                     "block_recall": "share of fraud blocked straight away",
                     "review_rate": "share of all transactions in the group sent to review",
                     "block_rate": "share of all transactions in the group blocked"},
        "by_type": by_type,
        "by_amount_quartile": by_amount,
    }
    record("v3_breakdown", payload, SOURCE)

    for title, rows in (("type", by_type), ("amount quartile", by_amount)):
        print(f"by {title}")
        for r in rows:
            recall = "   –  " if r["recall"] is None else f"{r['recall']:.4f}"
            print(f"  {r['group']:9s} rows {r['n_rows']:8,d} fraud {r['n_fraud']:4d} recall {recall} "
                  f"review {r['review_rate']:.4f} block {r['block_rate']:.5f}")


if __name__ == "__main__":
    main()
