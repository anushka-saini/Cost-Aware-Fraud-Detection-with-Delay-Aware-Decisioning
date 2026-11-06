"""Version 3, Phase 3: the one-time held-out test (days 14-16).

The decision rule is fixed here, before the held-out days are scored.

V3 replaces V2 only if, on days 14-16, both of these hold:

1. V3's PR-AUC is significantly higher than V2's (paired bootstrap, 95%).
2. V3's cost at its approved thresholds is not significantly higher than V2's
   cost at its deployed thresholds (paired bootstrap over transactions).

Otherwise V2 stays deployed and we say so. Nothing is tuned or tried again
after this runs.

V3 is exactly what Phase 2 approved: the tuned LightGBM (5 seeds, days 5-7)
with the calibrator and thresholds saved in models/runs/v3/v3_final.joblib.
V2 is exactly as deployed: its own models, calibrator and thresholds.

The script runs once. results/v3/held_out.json is the record; if it exists,
the script stops.

    python scripts/v3_phase3.py
"""

import json

import joblib
import numpy as np

from fraud.config import get_settings
from fraud.data import get_days, load_dense
from fraud.evaluation import (
    cost_report,
    paired_bootstrap_difference,
    paired_cost_difference,
    pr_auc_with_ci,
    roc_auc_with_ci,
    row_costs,
)
from fraud.features import FEATURE_SETS
from fraud.models import predict_fraud_proba
from fraud.registry import load_bundle
from fraud.results import _jsonable, record

SOURCE = "scripts/v3_phase3.py"
settings = get_settings()
RUN_DIR = settings.models / "runs" / "v3"
OUT_PATH = settings.results / "v3" / "held_out.json"
HELD_OUT_DAYS = [14, 15, 16]
DECISION_RULE = ("V3 replaces V2 only if its days 14-16 PR-AUC is significantly higher (paired bootstrap, 95%) "
                 "AND its cost at the approved thresholds is not significantly higher than V2's at its deployed "
                 "thresholds. Otherwise V2 stays deployed.")
POINT_KEYS = ("t_review", "t_block", "total_cost", "fraud_allowed", "fraud_allowed_amount", "fraud_reviewed",
              "fraud_blocked", "legit_reviewed", "legit_reviewed_pct", "legit_blocked", "block_precision",
              "block_recall", "block_f1")


def ensemble_scores(models, X):
    """Average raw score of the 5 seeds."""
    return np.mean([predict_fraud_proba(model, X) for model in models.values()], axis=0)


def main():
    if OUT_PATH.exists():
        raise SystemExit(f"{OUT_PATH} already exists: the held-out test has been run. It is not run twice.")

    final = joblib.load(RUN_DIR / "v3_final.joblib")
    v3_run = joblib.load(RUN_DIR / f"{final['model']}.joblib")
    v2_run = joblib.load(settings.models / "runs" / "phase2" / "lightgbm.joblib")
    v2_bundle = load_bundle("fraud_scorer", "v2")
    v2_features = FEATURE_SETS["v2"]

    columns = list(dict.fromkeys([*v3_run["features"], *v2_features]))
    frame = load_dense(columns)
    X, y = get_days(frame, HELD_OUT_DAYS, columns)
    y = y.to_numpy()
    amounts = X["amount"].to_numpy()
    print(f"held-out days {HELD_OUT_DAYS}: {len(y):,} rows, {int(y.sum())} fraud", flush=True)

    raw = {"v3": ensemble_scores(v3_run["models"], X[v3_run["features"]]),
           "v2": ensemble_scores(v2_run["models"], X[v2_features])}
    probs = {"v3": final["calibrator"].transform(raw["v3"]),
             "v2": v2_bundle["calibrator"].transform(raw["v2"])}
    thresholds = {"v3": final["thresholds"], "v2": v2_bundle["thresholds"]}

    models = {}
    for name in ("v3", "v2"):
        t = thresholds[name]
        point = cost_report(y, probs[name], amounts, t["t_review"], t["t_block"])
        models[name] = {
            "pr_auc": pr_auc_with_ci(y, raw[name]),
            "roc_auc": roc_auc_with_ci(y, raw[name]),
            "operating_point": {k: point[k] for k in POINT_KEYS},
        }
        print(f"{name}: PR-AUC {models[name]['pr_auc']['pr_auc']:.4f}, cost {point['total_cost']:,.0f}", flush=True)

    pr_test = paired_bootstrap_difference(y, raw["v3"], raw["v2"])
    roc_test = paired_bootstrap_difference(y, raw["v3"], raw["v2"], metric="roc_auc")
    costs = {name: row_costs(y, probs[name], amounts, thresholds[name]["t_review"], thresholds[name]["t_block"])
             for name in ("v3", "v2")}
    cost_test = paired_cost_difference(costs["v3"], costs["v2"])

    pr_better = bool(pr_test["significant"] and pr_test["difference"] > 0)
    cost_worse = bool(cost_test["significant"] and cost_test["difference"] > 0)
    replace = pr_better and not cost_worse
    if replace:
        verdict = "V3 is better on held-out: deploy V3."
    else:
        reasons = []
        if not pr_better:
            reasons.append("its PR-AUC is not significantly higher")
        if cost_worse:
            reasons.append("its cost is significantly higher")
        verdict = "Keep V2 deployed: " + " and ".join(reasons) + "."

    payload = {
        "days": HELD_OUT_DAYS,
        "n_rows": len(y),
        "n_fraud": int(y.sum()),
        "decision_rule": DECISION_RULE,
        "v3": {"model": final["model"], "calibration": final["calibration"], **models["v3"]},
        "v2_deployed": models["v2"],
        "v3_minus_v2": {"pr_auc": pr_test, "roc_auc": roc_test, "cost": cost_test},
        "baselines": {
            "block_nothing": {k: v for k, v in cost_report(y, probs["v3"], amounts, 2.0, 2.0).items() if k in POINT_KEYS},
            "review_everything": {k: v for k, v in cost_report(y, probs["v3"], amounts, 0.0, 2.0).items()
                                  if k in POINT_KEYS},
        },
        "pr_auc_significantly_higher": pr_better,
        "cost_significantly_higher": cost_worse,
        "deploy_v3": replace,
        "verdict": verdict,
    }
    OUT_PATH.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    record("v3_held_out", payload, SOURCE)

    print(f"PR-AUC V3 - V2: {pr_test['difference']:+.4f} [{pr_test['ci_low']:+.4f}, {pr_test['ci_high']:+.4f}]")
    print(f"cost V3 - V2: {cost_test['difference']:+,.0f} [{cost_test['ci_low']:+,.0f}, {cost_test['ci_high']:+,.0f}]")
    print(verdict)


if __name__ == "__main__":
    main()
