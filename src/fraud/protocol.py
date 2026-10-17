"""The evaluation protocol, applied identically to every model.

1. eval_1 (days 8-10): choose the class weight (and early stopping for the
   network), by mean PR-AUC over the selection seeds.
2. eval_2 (days 11-13): choose and fit the probability calibration, then
   choose the REVIEW / BLOCK thresholds by cost.
3. held-out (days 14-16): evaluate once. Nothing is chosen here.

Steps 1 and 2 live in ``select_family``; step 3 in ``evaluate_family``. They
are separate functions so the selection code can be run and debugged without
the held-out rows ever being loaded.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from fraud.calibration import calibration_metrics, fit_calibrator
from fraud.evaluation import (
    cost_report,
    paired_bootstrap_difference,
    pr_auc,
    pr_auc_with_ci,
    recall_by_amount_quartile,
    sweep_thresholds,
    threshold_robustness,
)
from fraud.models import MLPClassifier, make_lgbm, make_logreg, make_rf, predict_fraud_proba

SELECTION_SEEDS = [42, 1, 2]
FINAL_SEEDS = [42, 1, 2, 3, 4]
THRESHOLD_ODDS_MARGIN = 2.0      # thresholds must be safe if calibrated odds are off by up to 2x
ROBUSTNESS_DELTAS = (0.02, 0.05)
BLIND_SPOT_SCORE = 0.001         # "blind spot": real fraud given a raw score below 0.1%
# Quartiles of all PaySim fraud amounts (02_modeling.ipynb), kept fixed so tables stay comparable.
AMOUNT_BINS = [0, 127091, 441423, 1517771, 10_000_001]


@dataclass
class Family:
    name: str
    label: str
    weight_name: str
    weight_grid: list
    build: Callable            # (weight, seed) -> unfitted model
    needs_validation: bool = False


FAMILIES = {
    "logreg": Family("logreg", "Logistic regression", "class_weight", ["balanced"],
                     lambda w, seed: make_logreg(seed)),
    "lightgbm": Family("lightgbm", "LightGBM", "scale_pos_weight", [1, 5, 10, 20, 50, 100, 200, 500],
                       lambda w, seed: make_lgbm(seed, scale_pos_weight=w)),
    "random_forest": Family("random_forest", "Random Forest", "fraud class weight", [20, 50, 100, 200, 500, 1000],
                            lambda w, seed: make_rf(seed, fraud_weight=w)),
    "mlp": Family("mlp", "MLP (64-32-16)", "pos_weight", [1, 5, 20, 100, 500],
                  lambda w, seed: MLPClassifier(pos_weight=w, seed=seed), needs_validation=True),
}


def fit_model(family: Family, weight, seed, train, validation):
    model = family.build(weight, seed)
    if family.needs_validation:
        return model.fit(train[0], train[1], validation[0], validation[1])
    return model.fit(train[0], train[1])


# ---------------------------------------------------------------------------
# Step 2 helpers: calibration and thresholds, both on eval_2
# ---------------------------------------------------------------------------

def choose_calibration(scores, y, days) -> dict:
    """Isotonic unless it generalises worse than Platt.

    Both are fitted on all but the last day of the period and checked on the
    last day (Brier score); for eval_2 that is days 11-12 and day 13. The
    winner is then refitted on the whole period.
    """
    scores, y, days = np.asarray(scores), np.asarray(y), np.asarray(days)
    check = days == days.max()          # the last day of the period checks what the earlier days fitted
    fit = ~check
    check_metrics = {"none": calibration_metrics(y[check], scores[check])}
    for method in ("isotonic", "platt"):
        calibrator = fit_calibrator(scores[fit], y[fit], method)
        check_metrics[method] = calibration_metrics(y[check], calibrator.transform(scores[check]))
    method = "isotonic" if check_metrics["isotonic"]["brier"] <= check_metrics["platt"]["brier"] else "platt"
    return {"method": method, "calibrator": fit_calibrator(scores, y, method),
            "day13_check": check_metrics}


def _threshold_grid(probs):
    from fraud.evaluation import candidate_thresholds
    return np.unique(np.r_[candidate_thresholds(probs, 2000), np.round(np.linspace(0, 1, 201), 4),
                           np.geomspace(1e-7, 1, 141)])


def choose_thresholds(y, probs, amounts) -> dict:
    """Cost-minimising thresholds on eval_2.

    ``robust`` is the deployed rule: the pair with the lowest worst-case cost
    when each threshold's odds are scaled by up to THRESHOLD_ODDS_MARGIN either
    way, i.e. a plateau rather than a cliff edge.
    ``plain`` is the unconstrained minimum, reported for comparison only.
    """
    grid = _threshold_grid(probs)
    out = {}
    for name, margin in (("robust", THRESHOLD_ODDS_MARGIN), ("plain", 1.0)):
        choice = sweep_thresholds(y, probs, amounts, odds_margin=margin, review_grid=grid, block_grid=grid)
        report = cost_report(y, probs, amounts, choice["t_review"], choice["t_block"])
        out[name] = {**report, "odds_margin": margin, "worst_case_cost_within_margin": choice["objective"],
                     "robustness": threshold_robustness(y, probs, amounts, choice["t_review"], choice["t_block"],
                                                        ROBUSTNESS_DELTAS)}
    return out


def calibrate_and_threshold(scores_eval2, y_eval2, amounts_eval2, days_eval2) -> dict:
    calibration = choose_calibration(scores_eval2, y_eval2, days_eval2)
    probs = calibration["calibrator"].transform(scores_eval2)
    return {
        "calibrator": calibration["calibrator"],
        "calibration": {"method": calibration["method"], "day13_check": calibration["day13_check"],
                        "eval_2_after": calibration_metrics(y_eval2, probs)},
        "thresholds_eval_2": choose_thresholds(y_eval2, probs, amounts_eval2),
        "eval_2_pr_auc": pr_auc(y_eval2, scores_eval2),
    }


# ---------------------------------------------------------------------------
# Steps 1 + 2
# ---------------------------------------------------------------------------

def select_family(family: Family, train, eval_1, eval_2, eval_2_days, log=print) -> dict:
    """Everything that is *chosen* for one model family. Never sees held-out.

    train / eval_1 / eval_2 are (X, y) pairs; the amount column of X is used
    for costs. Returns fitted models for the final seeds plus the selection record.
    """
    sweep, fitted = [], {}
    for weight in family.weight_grid:
        scores = {}
        for seed in SELECTION_SEEDS:
            model = fit_model(family, weight, seed, train, eval_1)
            fitted[(weight, seed)] = model
            scores[seed] = pr_auc(eval_1[1], predict_fraud_proba(model, eval_1[0]))
        row = {"weight": weight, "eval_1_pr_auc_by_seed": scores,
               "mean": float(np.mean(list(scores.values()))), "std": float(np.std(list(scores.values())))}
        sweep.append(row)
        log(f"  {family.name} {family.weight_name}={weight}: eval_1 PR-AUC mean {row['mean']:.4f} "
            f"(std {row['std']:.4f}) over seeds {SELECTION_SEEDS}")
    best = max(sweep, key=lambda r: r["mean"])   # first maximum: ties go to the smaller weight
    weight = best["weight"]
    log(f"  -> chosen on eval_1: {family.weight_name} = {weight}")

    models = {}
    for seed in FINAL_SEEDS:
        models[seed] = fitted.get((weight, seed)) or fit_model(family, weight, seed, train, eval_1)
    del fitted

    y2, amounts2 = np.asarray(eval_2[1]), eval_2[0]["amount"].to_numpy()
    per_seed, raw_eval2 = {}, {}
    for seed, model in models.items():
        raw_eval2[seed] = predict_fraud_proba(model, eval_2[0])
        per_seed[seed] = calibrate_and_threshold(raw_eval2[seed], y2, amounts2, eval_2_days)
        per_seed[seed]["eval_1_pr_auc"] = pr_auc(eval_1[1], predict_fraud_proba(model, eval_1[0]))
        per_seed[seed]["train_pr_auc"] = pr_auc(train[1], predict_fraud_proba(model, train[0]))
    ensemble = calibrate_and_threshold(np.mean(list(raw_eval2.values()), axis=0), y2, amounts2, eval_2_days)
    ensemble["eval_1_pr_auc"] = pr_auc(
        eval_1[1], np.mean([predict_fraud_proba(m, eval_1[0]) for m in models.values()], axis=0))
    robust = ensemble["thresholds_eval_2"]["robust"]
    log(f"  -> ensemble on eval_2: calibration={ensemble['calibration']['method']}, "
        f"review>={robust['t_review']:.4g}, block>={robust['t_block']:.4g}, cost {robust['total_cost']:,.0f}, "
        f"fraud allowed {robust['fraud_allowed']}")
    return {"family": family.name, "label": family.label, "weight_name": family.weight_name,
            "weight": weight, "weight_sweep": sweep, "models": models,
            "per_seed": per_seed, "ensemble": ensemble}


# ---------------------------------------------------------------------------
# Step 3
# ---------------------------------------------------------------------------

def _evaluate_scores(raw, selected, y, amounts) -> dict:
    """Held-out metrics for one set of raw scores, using what was fixed on eval_2."""
    probs = selected["calibrator"].transform(raw)
    out = {**pr_auc_with_ci(y, raw), "calibration_held_out": calibration_metrics(y, probs),
           "calibration_method": selected["calibration"]["method"], "operating_points": {}}
    for rule in ("robust", "plain"):
        chosen = selected["thresholds_eval_2"][rule]
        t_review, t_block = chosen["t_review"], chosen["t_block"]
        report = cost_report(y, probs, amounts, t_review, t_block)
        report["eval_2_total_cost"] = chosen["total_cost"]
        report["robustness"] = threshold_robustness(y, probs, amounts, t_review, t_block, ROBUSTNESS_DELTAS)
        report["recall_by_amount_quartile"] = recall_by_amount_quartile(y, probs, amounts, t_review, t_block,
                                                                        AMOUNT_BINS)
        out["operating_points"][rule] = report
    y_bool = np.asarray(y).astype(bool)
    low = y_bool & (raw < BLIND_SPOT_SCORE)
    t_review = selected["thresholds_eval_2"]["robust"]["t_review"]
    out["blind_spot"] = {
        "definition": f"real fraud with raw score < {BLIND_SPOT_SCORE}",
        "n": int(low.sum()), "amount": float(amounts[low].sum()),
        "n_allowed_at_deployed_threshold": int((low & (probs < t_review)).sum()),
    }
    return out, probs


def evaluate_family(selection: dict, held_out) -> dict:
    """The single held-out evaluation of one family: each seed and the ensemble."""
    X, y = held_out
    y, amounts = np.asarray(y), X["amount"].to_numpy()
    raw = {seed: predict_fraud_proba(model, X) for seed, model in selection["models"].items()}
    seeds = {}
    for seed, scores in raw.items():
        seeds[seed], _ = _evaluate_scores(scores, selection["per_seed"][seed], y, amounts)
    ensemble_raw = np.mean(list(raw.values()), axis=0)
    ensemble, ensemble_probs = _evaluate_scores(ensemble_raw, selection["ensemble"], y, amounts)

    def spread(getter):
        values = [getter(s) for s in seeds.values()]
        return {"mean": float(np.mean(values)), "std": float(np.std(values)),
                "min": float(np.min(values)), "max": float(np.max(values)), "values": values}

    point = lambda key: (lambda s: s["operating_points"]["robust"][key])  # noqa: E731
    return {
        "family": selection["family"], "label": selection["label"],
        "weight_name": selection["weight_name"], "weight": selection["weight"],
        "seeds": seeds,
        "seed_summary": {
            "pr_auc": spread(lambda s: s["pr_auc"]),
            "total_cost": spread(point("total_cost")),
            "fraud_allowed": spread(point("fraud_allowed")),
            "legit_reviewed_pct": spread(point("legit_reviewed_pct")),
            "legit_blocked": spread(point("legit_blocked")),
        },
        "ensemble": ensemble,
        "_raw": {"seed_42": raw[FINAL_SEEDS[0]], "ensemble": ensemble_raw, "ensemble_probs": ensemble_probs},
    }


def compare_families(results: dict, y, which: str = "ensemble") -> list[dict]:
    """Paired bootstrap on held-out PR-AUC for every pair of families."""
    names, rows = list(results), []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            diff = paired_bootstrap_difference(y, results[a]["_raw"][which], results[b]["_raw"][which])
            rows.append({"a": a, "b": b, "scores": which, **diff})
    return rows


def blind_spot_overlap(results: dict, y, which: str = "ensemble") -> dict:
    """For each family's blind-spot frauds (raw score below the cut), how many
    each other family would send to REVIEW or BLOCK at its own deployed threshold."""
    y = np.asarray(y).astype(bool)
    out = {}
    for a, result in results.items():
        low = y & (result["_raw"][which] < BLIND_SPOT_SCORE)
        flagged = {}
        for b, other in results.items():
            if b != a:
                t_review = other["ensemble"]["operating_points"]["robust"]["t_review"]
                flagged[b] = int((other["_raw"]["ensemble_probs"][low] >= t_review).sum())
        out[a] = {"n": int(low.sum()), "flagged_by": flagged}
    return out
