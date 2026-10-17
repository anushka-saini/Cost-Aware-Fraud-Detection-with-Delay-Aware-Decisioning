"""A compact summary of the project's results for the dashboard.

Everything "new" is read from results/metrics.json (written by the scripts).
The "old" figures are the ones the project reported before the evaluation
protocol was corrected; they are kept here so both can be shown side by side.
"""

from fraud.results import load_metrics

# Held-out figures as originally reported (README; notebooks 05 and 07).
OLD = {
    "lightgbm": {"label": "LightGBM", "weight": 20, "pr_auc": 0.7198, "ci": (0.6877, 0.7492),
                 "t_review": 0.0001, "t_block": 0.90, "total_cost": 46_455_419, "fraud_allowed": 34,
                 "legit_reviewed_pct": 0.0427, "legit_blocked": 186,
                 "block_precision": 0.7745, "block_recall": 0.7774, "block_f1": 0.7760},
    "random_forest": {"label": "Random Forest", "weight": 500, "pr_auc": 0.8335, "ci": (0.8085, 0.8546),
                      "t_review": 0.32, "t_block": 0.85, "total_cost": 47_547, "fraud_allowed": 0,
                      "legit_reviewed_pct": 0.0393, "legit_blocked": 23,
                      "block_precision": 0.9639, "block_recall": 0.7482, "block_f1": 0.8425},
    "mlp": {"label": "MLP (64-32-16)", "weight": 20, "pr_auc": 0.8485, "ci": (0.8217, 0.8720),
            "t_review": 0.000631, "t_block": 0.95, "total_cost": 48_551, "fraud_allowed": 1,
            "legit_reviewed_pct": 0.0401, "legit_blocked": 12,
            "block_precision": 0.9816, "block_recall": 0.7798, "block_f1": 0.8692},
}
OLD_FLAWS = {
    "lightgbm": "weight and thresholds chosen on held-out",
    "random_forest": "weight and thresholds chosen on held-out",
    "mlp": "weight chosen on eval_1 (sound); thresholds chosen on held-out",
}
OLD_DEPLOYED = "random_forest"
POINT_KEYS = ("t_review", "t_block", "total_cost", "fraud_allowed", "legit_reviewed_pct", "legit_blocked",
              "block_precision", "block_recall", "block_f1")


def _family(key: str, fam: dict, deployed_key: str | None) -> dict:
    ensemble, point = fam["ensemble"], fam["ensemble"]["operating_points"]["robust"]
    seeds = fam["seed_summary"]
    old = OLD.get(key)
    return {
        "key": key, "label": fam["label"], "deployed": key == deployed_key,
        "weight_name": fam["weight_name"], "weight": fam["weight"], "calibration": ensemble["calibration_method"],
        "pr_auc": ensemble["pr_auc"], "ci_low": ensemble["ci_low"], "ci_high": ensemble["ci_high"],
        "seed_mean": seeds["pr_auc"]["mean"], "seed_std": seeds["pr_auc"]["std"], "seed_values": seeds["pr_auc"]["values"],
        "seed_costs": seeds["total_cost"]["values"], "seed_fraud_allowed": seeds["fraud_allowed"]["values"],
        **{k: point[k] for k in POINT_KEYS},
        "recall_by_amount_quartile": point["recall_by_amount_quartile"],
        "robustness": point["robustness"],
        "old": None if old is None else {**{k: old[k] for k in ("pr_auc", *POINT_KEYS)}, "ci_low": old["ci"][0],
                                         "ci_high": old["ci"][1], "weight": old["weight"],
                                         "flaw": OLD_FLAWS[key], "deployed": key == OLD_DEPLOYED},
    }


def build_summary(deployed_family: str = "lightgbm") -> dict:
    """Old-versus-new comparison plus the supporting experiments, or which
    sections are missing when the scripts have not been run."""
    metrics = load_metrics()
    needed = ["phase2_final"]
    missing = [s for s in needed if s not in metrics]
    if missing:
        return {"available": False, "missing_sections": missing}

    final = metrics["phase2_final"]
    out = {
        "available": True,
        "split": final["split"], "n_rows": final["n_rows"], "n_fraud": final["n_fraud"],
        "protocol": "Class weights chosen on eval_1 (days 8-10); calibration and thresholds on eval_2 (days 11-13); "
                    "held-out (days 14-16) scored once. Figures are for the 5-seed averaged ensemble.",
        "features": "Leak-free: the three whole-hour totals are replaced by the hour of day.",
        "sources": {"models": final["_source"]},
        "models": [_family(k, f, deployed_family) for k, f in final["families"].items()],
        "paired_tests": final["paired_bootstrap_ensemble"],
    }

    phase1 = metrics.get("phase1_final")
    if phase1:
        out["same_features_honest_protocol"] = [
            {"key": k, "label": f["label"], "pr_auc": f["ensemble"]["pr_auc"], "ci_low": f["ensemble"]["ci_low"],
             "ci_high": f["ensemble"]["ci_high"], **{p: f["ensemble"]["operating_points"]["robust"][p] for p in POINT_KEYS}}
            for k, f in phase1["families"].items()]

    ablation = metrics.get("phase3_balance_ablation")
    if ablation:
        out["balance_ablation"] = {k: ablation[k] for k in ("removed_features", "with_balance_features",
                                                            "without_balance_features",
                                                            "pr_auc_difference_without_minus_with")}

    phase4 = metrics.get("phase4_final")
    if phase4:
        hybrid = phase4["hybrid"]
        point = hybrid["ensemble"]["operating_points"]["robust"]
        out["deep_learning"] = {
            "autoencoder": {"seed_42": {k: phase4["autoencoder"]["seed_42"][k] for k in
                                        ("pr_auc", "ci_low", "ci_high", "share_of_fraud_in_top_1pct",
                                         "share_of_fraud_in_top_5pct")},
                            "seeds": phase4["autoencoder"]["seeds"], "base_rate": phase4["autoencoder"]["base_rate"]},
            "hybrid": {"label": hybrid["label"], "pr_auc": hybrid["ensemble"]["pr_auc"],
                       "ci_low": hybrid["ensemble"]["ci_low"], "ci_high": hybrid["ensemble"]["ci_high"],
                       "seed_mean": hybrid["seed_summary"]["pr_auc"]["mean"],
                       "seed_std": hybrid["seed_summary"]["pr_auc"]["std"], **{k: point[k] for k in POINT_KEYS}},
            "hybrid_minus_supervised": phase4["hybrid_minus_supervised"],
            "supervised_blind_spot": phase4["supervised_blind_spot"],
        }

    rolling = metrics.get("phase5_rolling")
    if rolling:
        out["rolling"] = {k: rolling[k] for k in ("design", "by_delay", "delay_effect", "champion_challenger",
                                                  "anomaly_only_daily_pr_auc")}
    if "phase5_fallback" in metrics:
        out["fallback"] = {k: metrics["phase5_fallback"][k] for k in
                           ("rule", "fallback_off", "fallback_on", "extra_cost", "extra_legit_reviewed",
                            "fraud_rescued_from_allow")}
    monitoring = metrics.get("phase6_monitoring")
    if monitoring:
        out["stress_test"] = monitoring["stress_test"]
        out["alert_rules"] = monitoring["alert_rules"]
    if "phase7_latency" in metrics:
        out["latency"] = {k: metrics["phase7_latency"][k] for k in ("endpoint", "setup", "sequential", "concurrent")}
    return out
