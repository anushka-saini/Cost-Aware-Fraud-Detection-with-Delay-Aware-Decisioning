"""What V3 pays attention to, and real transactions to explain in the demo.

Two outputs:

1. Overall feature importance (results/metrics.json, section v3_explainability).
   SHAP values of the deployed V3 model on days 11-13: every fraud on those
   days and a random sample of legitimate transactions. Reported separately
   for fraud and legitimate rows, as the mean absolute contribution in
   log-odds. The two hour-of-day inputs are added together as "time of day".

2. Guided demo scenarios (api/data/scenarios.json). Five real held-out
   transactions (days 14-16), each picked by a fixed rule written below, so
   the demo shows typical cases and also the model's mistakes. Picking them
   changes nothing about the model; the held-out days only provide examples.

    python scripts/v3_explainability.py
"""

import json

import numpy as np
import pandas as pd

from fraud.config import get_settings
from fraud.data import SPLITS, get_days, load_dense, load_feature_frame
from fraud.features import TRANSACTION_TYPES, TYPE_COLUMNS
from fraud.registry import load_bundle
from fraud.results import record
from fraud.serving import Scorer

SOURCE = "scripts/v3_explainability.py"
N_LEGIT_SAMPLE, SEED = 20_000, 7
RAW = ["step", "nameOrig", "nameDest", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]

# Each scenario: a title, the rule that picks it, and the rows it picks from.
SCENARIOS = [
    {"id": "ordinary_payment", "title": "An ordinary payment",
     "rule": "the legitimate PAYMENT with the median fraud probability among allowed payments",
     "where": lambda d: (d.isFraud == 0) & (d.type == "PAYMENT") & (d.tier == "ALLOW"), "pick": "median"},
    {"id": "clear_fraud", "title": "A transfer that empties the account",
     "rule": "the blocked fraudulent TRANSFER with the median fraud probability among them",
     "where": lambda d: (d.isFraud == 1) & (d.type == "TRANSFER") & (d.tier == "BLOCK"), "pick": "median"},
    {"id": "fraud_near_threshold", "title": "A fraud that nearly got through",
     "rule": "the fraud with the lowest fraud probability (the one closest to being allowed)",
     "where": lambda d: d.isFraud == 1, "pick": "lowest"},
    {"id": "legit_review", "title": "A legitimate cash-out sent to review",
     "rule": "the legitimate CASH_OUT with the median fraud probability among those sent to review",
     "where": lambda d: (d.isFraud == 0) & (d.type == "CASH_OUT") & (d.tier == "REVIEW"), "pick": "median"},
    {"id": "wrongful_block", "title": "A legitimate transaction that was blocked",
     "rule": "the legitimate transaction with the highest fraud probability (one of the model's wrongful blocks)",
     "where": lambda d: (d.isFraud == 0) & (d.tier == "BLOCK"), "pick": "highest"},
]


def shap_by_feature(scorer: Scorer, X: pd.DataFrame) -> pd.DataFrame:
    """Signed SHAP values (log-odds), averaged over the 5 seeds, one column per input."""
    frame = scorer.model_frame(X)
    values = np.mean([np.asarray(e.shap_values(frame)) for e in scorer._tree_explainers()], axis=0)
    out = pd.DataFrame(values, columns=scorer.feature_columns, index=X.index)
    out["time_of_day"] = out.pop("hour_sin") + out.pop("hour_cos")
    return out


def global_importance(scorer: Scorer) -> dict:
    features = scorer.feature_columns
    X, y = get_days(load_dense(features), SPLITS["eval_2"], features)
    fraud = X[y == 1]
    legit = X[y == 0].sample(N_LEGIT_SAMPLE, random_state=SEED)
    shap_fraud, shap_legit = shap_by_feature(scorer, fraud), shap_by_feature(scorer, legit)
    rows = []
    for name in shap_fraud.columns:
        rows.append({"feature": name,
                     "fraud_mean_abs": float(shap_fraud[name].abs().mean()),
                     "fraud_mean": float(shap_fraud[name].mean()),
                     "legit_mean_abs": float(shap_legit[name].abs().mean()),
                     "legit_mean": float(shap_legit[name].mean())})
    rows.sort(key=lambda r: r["fraud_mean_abs"], reverse=True)
    return {"days": SPLITS["eval_2"], "n_fraud": len(fraud), "n_legit_sample": len(legit),
            "units": "log-odds", "sample_seed": SEED, "features": rows}


def pick_scenarios(scorer: Scorer) -> list[dict]:
    features = scorer.feature_columns
    frame = load_feature_frame(list(dict.fromkeys([*RAW, *features, *TYPE_COLUMNS, "day_bucket", "isFraud"])))
    held = frame[frame["day_bucket"].isin(SPLITS["held_out"])].copy()
    held["type"] = [TRANSACTION_TYPES[i] for i in np.argmax(held[TYPE_COLUMNS].to_numpy(), axis=1)]
    scored = scorer.score_frame(held[features])
    held["probability"], held["tier"] = scored["fraud_probability"].to_numpy(), scored["tier"].to_numpy()

    chosen = []
    for spec in SCENARIOS:
        group = held[spec["where"](held)].sort_values(["probability", "step"], kind="stable")
        if group.empty:
            print(f"{spec['id']}: no matching transaction, skipped")
            continue
        index = {"lowest": 0, "highest": len(group) - 1, "median": len(group) // 2}[spec["pick"]]
        row = group.iloc[index]
        chosen.append({
            "id": spec["id"], "title": spec["title"], "rule": spec["rule"], "group_size": len(group),
            "transaction": {k: (row[k].item() if hasattr(row[k], "item") else row[k]) for k in RAW} | {"type": row["type"]},
            "features": {k: (row[k].item() if hasattr(row[k], "item") else row[k]) for k in features},
            "actual_fraud": int(row["isFraud"]), "expected_tier": row["tier"],
        })
        print(f"{spec['id']}: {row['type']} {row['amount']:,.2f}, tier {row['tier']}, fraud {int(row['isFraud'])}")
    return chosen


def main():
    settings = get_settings()
    bundle = load_bundle("fraud_scorer", "v3")
    scorer = Scorer(bundle)

    importance = global_importance(scorer)
    record("v3_explainability", {"model": {"name": "fraud_scorer", "version": "v3"}, **importance}, SOURCE)
    print("recorded v3_explainability")

    scenarios = pick_scenarios(scorer)
    out = settings.project_root / "api" / "data" / "scenarios.json"
    out.write_text(json.dumps({"_source": SOURCE, "model": "fraud_scorer v3", "days": SPLITS["held_out"],
                               "scenarios": scenarios}, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
