"""Register V3 as a servable bundle, after it won the held-out test.

V3 is the tuned LightGBM (5 seeds, trained on days 5-7) with the Platt
calibrator and thresholds approved in Phase 2 (models/runs/v3/v3_final.joblib).
Unlike V2 it carries no autoencoder: in V3 the autoencoder only feeds
monitoring, so the API scores without PyTorch.

Every number stored in the bundle is copied from results/metrics.json.

    python scripts/register_v3.py
"""

import joblib

from fraud.config import get_settings
from fraud.data import get_days, load_dense
from fraud.registry import save_bundle
from fraud.results import load_metrics

TRAIN_DAYS = [5, 6, 7]
POINT_KEYS = ("total_cost", "fraud_allowed", "legit_reviewed_pct", "legit_blocked",
              "block_precision", "block_recall", "block_f1")


def main():
    settings = get_settings()
    metrics_json = load_metrics()
    held_out = metrics_json["v3_held_out"]
    if not held_out["deploy_v3"]:
        raise SystemExit("The held-out test said to keep V2. Nothing to register.")

    final = joblib.load(settings.models / "runs" / "v3" / "v3_final.joblib")
    run = joblib.load(settings.models / "runs" / "v3" / f"{final['model']}.joblib")
    features = run["features"]
    validation = metrics_json["v3_validation"]["leaderboard"][final["model"]]
    safe = metrics_json["v3_phase2_decisions"]["safe_point"]

    metrics = {
        "eval_2_pr_auc": validation["pr_auc"]["pr_auc"],
        "eval_2_total_cost": safe["total_cost"],
        "held_out": {k: held_out["v3"]["pr_auc"][k] for k in ("pr_auc", "ci_low", "ci_high")}
        | {k: held_out["v3"]["operating_point"][k] for k in POINT_KEYS},
    }

    # Typical row, used to explain a transaction when SHAP needs a reference.
    X_train, _ = get_days(load_dense(features), TRAIN_DAYS, features)
    reference_row = X_train.median(numeric_only=False).to_dict()

    target = save_bundle(
        "fraud_scorer",
        model=[run["models"][seed] for seed in sorted(run["models"])],
        feature_columns=features,
        calibrator=final["calibrator"],
        thresholds=final["thresholds"],
        metrics=metrics,
        train_days=TRAIN_DAYS,
        protocol={"run_tag": "v3", "family": final["model"], "feature_set": "v2",
                  "hyperparameters": run["params"], "seeds": sorted(run["models"]),
                  "hyperparameters_chosen_on": "Optuna folds inside days 5-10",
                  "calibration_and_thresholds_chosen_on": "days 11-13",
                  "held_out_test": "days 14-16, run once (scripts/v3_phase3.py)"},
        extra={"reference_row": reference_row},
    )
    print("registered", target)


if __name__ == "__main__":
    main()
