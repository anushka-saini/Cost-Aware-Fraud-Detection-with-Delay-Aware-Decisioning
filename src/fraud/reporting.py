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


# ---------------------------------------------------------------------------
# Version 3 (the Model lab page)
# ---------------------------------------------------------------------------

V3_SECTIONS = ["v3_validation", "v3_phase2_decisions", "v3_held_out"]
KIND = {"logreg": "linear", "random_forest": "trees", "lightgbm": "trees", "catboost": "trees", "xgboost": "trees",
        "lightgbm_v2_settings": "trees", "v2_deployed": "trees", "mlp": "neural", "resnet": "neural",
        "ft_transformer": "neural", "tabm": "neural", "autoencoder": "no labels", "stacked": "ensemble"}


def _test(t: dict | None) -> dict | None:
    if not t:
        return None
    keys = ("difference", "ci_low", "ci_high", "p_holm", "significant_after_holm", "significant")
    return {k: t[k] for k in keys if k in t}


def _leaderboard_row(key: str, row: dict) -> dict:
    day_13 = row.get("day_13", {})
    return {
        "key": key, "label": row["label"], "kind": KIND.get(key, "other"), "tests_on": row.get("tests_on"),
        "pr_auc": row.get("pr_auc"), "roc_auc": row.get("roc_auc"),
        "seed_mean": (row.get("seed_pr_auc") or {}).get("mean"), "seed_std": (row.get("seed_pr_auc") or {}).get("std"),
        "day_13_pr_auc": day_13.get("pr_auc"),
        "day_13_cost": (day_13.get("operating_point") or {}).get("total_cost"),
        "day_13_fraud_allowed": (day_13.get("operating_point") or {}).get("fraud_allowed"),
        "day_13_cost_vs_v2": _test(day_13.get("cost_minus_v2_deployed")),
        "vs_v3": _test(row.get("vs_main_lightgbm")), "vs_v2": _test(row.get("vs_v2_deployed")),
        "verdict_vs_v3": row["verdict_vs_main_lightgbm"], "verdict_vs_v2": row["verdict_vs_v2_deployed"],
    }


def build_v3_summary() -> dict:
    """Leaderboard, Phase 2 choices, the held-out result, the V1 -> V2 -> V3
    history and the curves, all read from results/metrics.json."""
    metrics = load_metrics()
    missing = [s for s in V3_SECTIONS if s not in metrics]
    if missing:
        return {"available": False, "missing_sections": missing}

    validation, phase2, held = (metrics[s] for s in V3_SECTIONS)
    v3_key = validation["proposed_v3"]
    point_keys = ("t_review", "t_block", "total_cost", "fraud_allowed", "legit_reviewed_pct", "legit_blocked",
                  "block_precision", "block_recall")

    history = []
    v1 = metrics.get("v1_reported")
    if v1:
        old = v1["models"][v1["deployed"]]
        history.append({"version": "V1", "model": old["label"], "pr_auc": old["pr_auc"], "ci_low": old["ci_low"],
                        "ci_high": old["ci_high"], "total_cost": old["total_cost"],
                        "fraud_allowed": old["fraud_allowed"], "legit_blocked": old["legit_blocked"],
                        "honest": False, "note": v1["warning"], "source": v1["_source"]})
    for version, side in (("V2", "v2_deployed"), ("V3", "v3")):
        result = held[side]
        history.append({"version": version, "model": "LightGBM, 5 seeds", "honest": True,
                        **{k: result["pr_auc"][k] for k in ("pr_auc", "ci_low", "ci_high")},
                        **{k: result["operating_point"][k] for k in ("total_cost", "fraud_allowed", "legit_blocked")},
                        "note": "held-out days scored once, after every choice was made on earlier days",
                        "source": held["_source"]})

    curves = metrics.get("v3_curves")
    rules = metrics.get("rule_baselines")
    gate = metrics.get("retraining_gate")
    return {
        "available": True,
        "v3_key": v3_key,
        "validation": {
            "evaluated_on": validation["evaluated_on"], "significance_rule": validation["significance_rule"],
            "stacking_note": validation["stacking_note"], "setup": validation["setup"],
            "leaderboard": [_leaderboard_row(k, r) for k, r in validation["leaderboard"].items()],
        },
        "phase2": {
            "calibration": phase2["calibration"]["method"],
            "isotonic_rejected_because": phase2["calibration"].get("isotonic_rejected_because"),
            "threshold_rule": phase2["threshold_rule"],
            "point": {k: phase2["safe_point"][k] for k in point_keys},
            "odds_factor_to_cliff": phase2["cliff_safe"]["odds_factor_to_cliff"],
            "baselines": {k: phase2["baselines"][k]["total_cost"] for k in phase2["baselines"]},
        },
        "held_out": {
            "days": held["days"], "n_rows": held["n_rows"], "n_fraud": held["n_fraud"],
            "decision_rule": held["decision_rule"], "verdict": held["verdict"], "deploy_v3": held["deploy_v3"],
            "v3": {"pr_auc": held["v3"]["pr_auc"], "roc_auc": held["v3"]["roc_auc"],
                   **{k: held["v3"]["operating_point"][k] for k in point_keys}},
            "v2": {"pr_auc": held["v2_deployed"]["pr_auc"], "roc_auc": held["v2_deployed"]["roc_auc"],
                   **{k: held["v2_deployed"]["operating_point"][k] for k in point_keys}},
            "difference": {k: _test(v) for k, v in held["v3_minus_v2"].items()},
            "baselines": {k: held["baselines"][k]["total_cost"] for k in held["baselines"]},
        },
        "history": history,
        "curves": None if not curves else {k: curves[k] for k in ("curves_on", "calibration_on", "pr_note",
                                                                  "roc_note", "curves", "calibration")},
        "rules": None if not rules else {k: rules[k] for k in ("design_days", "how_rules_were_made", "tier_rule",
                                                               "cut_off_search", "rules", "best_rule",
                                                               "best_rule_chosen_by", "v3", "caveat")},
        "retraining": None if not gate else {"latest": gate["latest"], "runs_in_log": gate["runs_in_log"],
                                             "gate_rule": gate["gate_rule"]},
        "sources": {s: metrics[s]["_source"] for s in [*V3_SECTIONS, "v3_curves", "v1_reported", "rule_baselines",
                                                       "retraining_gate"] if s in metrics},
    }
