"""Register a model family from a protocol run as a servable bundle.

The bundle is the 5-seed ensemble with the calibrator and thresholds that were
fixed on eval_2, plus its held-out metrics when the final stage has been run.

    python scripts/register_model.py --tag phase2 --family lightgbm
"""

import argparse
import json

import joblib
import numpy as np

from fraud.config import get_settings
from fraud.data import SPLITS, get_split, load_dense
from fraud.features import FEATURE_SETS
from fraud.registry import save_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--features", default="v2", choices=sorted(FEATURE_SETS))
    parser.add_argument("--name", default="fraud_scorer")
    parser.add_argument("--autoencoder-tag", help="attach the label-free detector from this run (e.g. phase4)")
    parser.add_argument("--autoencoder-seed", type=int, default=42)
    parser.add_argument("--max-label-age-days", type=float, default=14,
                        help="label age beyond which the anomaly-score fallback applies")
    parser.add_argument("--anomaly-review-quantile", type=float, default=0.99)
    args = parser.parse_args()

    settings = get_settings()
    features = FEATURE_SETS[args.features]
    selection = joblib.load(settings.models / "runs" / args.tag / f"{args.family}.joblib")
    ensemble = selection["ensemble"]
    chosen = ensemble["thresholds_eval_2"]["robust"]

    metrics = {"eval_1_pr_auc": ensemble["eval_1_pr_auc"], "eval_2_pr_auc": ensemble["eval_2_pr_auc"],
               "eval_2_total_cost": chosen["total_cost"]}
    final_path = settings.results / args.tag / "final.json"
    if final_path.exists():
        held = json.loads(final_path.read_text(encoding="utf-8"))["families"][args.family]["ensemble"]
        point = held["operating_points"]["robust"]
        metrics["held_out"] = {k: held[k] for k in ("pr_auc", "ci_low", "ci_high")} | {
            k: point[k] for k in ("total_cost", "fraud_allowed", "legit_reviewed_pct", "legit_blocked",
                                  "block_precision", "block_recall", "block_f1")}

    # Typical row, used to explain models that SHAP's tree explainer cannot handle.
    X_train, _ = get_split(load_dense(features), "train", features)
    reference_row = X_train.median(numeric_only=False).to_dict()

    extra = {"reference_row": reference_row}
    if args.autoencoder_tag:
        saved = joblib.load(settings.models / "runs" / args.autoencoder_tag / "autoencoder.joblib")
        autoencoder = saved["autoencoders"][args.autoencoder_seed]
        # Threshold from unlabelled eval_2 traffic: no labels are involved in the fallback rule.
        eval_2, _ = get_split(load_dense(saved["feature_columns"]), "eval_2", saved["feature_columns"])
        threshold = float(np.quantile(autoencoder.anomaly_score(eval_2), args.anomaly_review_quantile))
        extra.update(
            autoencoder=autoencoder, autoencoder_columns=saved["feature_columns"],
            staleness={"labels_through_day": max(SPLITS["train"]), "max_label_age_days": args.max_label_age_days,
                       "anomaly_review_threshold": threshold,
                       "anomaly_review_quantile": args.anomaly_review_quantile,
                       "threshold_fitted_on": "eval_2 anomaly scores (no labels)",
                       "autoencoder_run": args.autoencoder_tag, "autoencoder_seed": args.autoencoder_seed},
        )

    target = save_bundle(
        args.name,
        model=[selection["models"][seed] for seed in sorted(selection["models"])],
        feature_columns=features,
        calibrator=ensemble["calibrator"],
        thresholds={"t_review": chosen["t_review"], "t_block": chosen["t_block"]},
        metrics=metrics,
        train_days=SPLITS["train"],
        protocol={"run_tag": args.tag, "family": args.family, "feature_set": args.features,
                  "weight": {selection["weight_name"]: selection["weight"]},
                  "seeds": sorted(selection["models"]),
                  "weights_chosen_on": "eval_1", "calibration_and_thresholds_chosen_on": "eval_2"},
        extra=extra,
    )
    print("registered", target)


if __name__ == "__main__":
    main()
