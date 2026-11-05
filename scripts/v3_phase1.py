"""Version 3, Phase 1: a better main model and a full model comparison.

Uses days 5-13 only. The held-out days (14-16) are dropped before anything else.

    python scripts/v3_phase1.py ladder                    # A: one change at a time to the V2 LightGBM
    python scripts/v3_phase1.py cost-check                # does log(amount) weighting lower the cost?
    python scripts/v3_phase1.py recipient-check           # which recipient features were tested, and why
    python scripts/v3_phase1.py tune lightgbm xgboost ... # B: Optuna search + 5-seed refit per model
    python scripts/v3_phase1.py record                    # write the tuning results to metrics.json
    python scripts/v3_phase1.py compare                   # leaderboard, stacking, paired tests

Rule for every "better" claim: a paired bootstrap whose difference is
significant after a Holm correction across all the models compared.
Anything else is "comparable".

Day 13 is used as an extra out-of-sample check: calibration and thresholds
(and the stacked ensemble) are fitted on days 11-12 and measured on day 13.
"""

import argparse
import json
import time
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from fraud.calibration import fit_calibrator
from fraud.config import get_settings
from fraud.data import TARGET, get_days, load_dense
from fraud.evaluation import (
    cost_report,
    holm_adjust,
    paired_bootstrap_difference,
    paired_cost_difference,
    pr_auc,
    pr_auc_with_ci,
    roc_auc_with_ci,
    row_costs,
)
from fraud.features import FEATURE_SETS, FEATURES_V3, RECIPIENT_PROFILE
from fraud.models import make_lgbm, predict_fraud_proba
from fraud.protocol import calibrate_and_threshold, choose_thresholds
from fraud.results import _jsonable, record
from fraud.tuning import (
    FINAL_SEEDS,
    FOLDS,
    LABELS,
    NET_LEGIT_SHARE,
    NET_VALIDATION_LEGIT_SHARE,
    NEURAL,
    TRIALS,
    VALIDATION_DAYS,
    amount_weights,
    final_params,
    fit_model,
    load_search_frame,
    run_search,
)

SOURCE = "scripts/v3_phase1.py"
settings = get_settings()
RUN_DIR = settings.models / "runs" / "v3"
OUT_DIR = settings.results / "v3"
ALL_DAYS = [5, 6, 7, 8, 9, 10]
CHECK_DAY = 13          # fit on days 11-12, measure on day 13
SIGNIFICANCE = 0.05
POINT_KEYS = ("t_review", "t_block", "total_cost", "fraud_allowed", "fraud_allowed_amount", "legit_reviewed",
              "legit_reviewed_pct", "legit_blocked", "block_precision", "block_recall", "block_f1",
              "n_fraud", "n_legit")


def log(message):
    print(message, flush=True)


def save_json(name, payload):
    path = OUT_DIR / name
    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    return path


def validation_data(frame, features=FEATURES_V3):
    X, y = get_days(frame, VALIDATION_DAYS, features)
    days = frame.loc[X.index, "day_bucket"].to_numpy()
    return X, y.to_numpy(), days


def summarise(raw_by_seed, X_val, y_val, days):
    """Days 11-13 numbers for one model: its 5-seed average (the ensemble)
    plus the spread of the single seeds."""
    ensemble = np.mean(list(raw_by_seed.values()), axis=0)
    seed_pr_auc = [pr_auc(y_val, s) for s in raw_by_seed.values()]
    chosen = calibrate_and_threshold(ensemble, y_val, X_val["amount"].to_numpy(), days)
    point = chosen["thresholds_eval_2"]["robust"]
    return ensemble, {
        "pr_auc": pr_auc_with_ci(y_val, ensemble),
        "roc_auc": roc_auc_with_ci(y_val, ensemble),
        "seed_pr_auc": {"values": seed_pr_auc, "mean": float(np.mean(seed_pr_auc)),
                        "std": float(np.std(seed_pr_auc)), "min": float(np.min(seed_pr_auc)),
                        "max": float(np.max(seed_pr_auc))},
        "calibration": chosen["calibration"]["method"],
        "operating_point": {k: point[k] for k in (
            "t_review", "t_block", "total_cost", "fraud_allowed", "legit_reviewed_pct", "legit_blocked",
            "block_precision", "block_recall", "block_f1")},
    }


def better(y, new, old):
    """Paired bootstrap of new minus old. ``keep`` only for a significant gain."""
    test = paired_bootstrap_difference(y, new, old)
    return test, bool(test["significant"] and test["difference"] > 0)


def validation_labels():
    """Label, amount and day for every days 11-13 row, in the same order as validation_data."""
    frame = load_dense(["amount"])
    rows = frame[frame["day_bucket"].isin(VALIDATION_DAYS)]
    return rows[TARGET].to_numpy(), rows["amount"].to_numpy(), rows["day_bucket"].to_numpy()


def day13_check(raw, y, amounts, days):
    """Fit on days 11-12, measure on day 13.

    Platt calibration (V2's method) and the safe thresholds are both chosen on
    days 11-12, so the day-13 cost is out of sample. Returns the summary and
    the cost of every day-13 row (for paired cost tests).
    """
    fit, check = days != CHECK_DAY, days == CHECK_DAY
    calibrator = fit_calibrator(raw[fit], y[fit], "platt")
    probs = calibrator.transform(raw)
    safe = choose_thresholds(y[fit], probs[fit], amounts[fit])["robust"]
    t_review, t_block = safe["t_review"], safe["t_block"]
    point = cost_report(y[check], probs[check], amounts[check], t_review, t_block)
    summary = {
        "pr_auc": pr_auc_with_ci(y[check], raw[check]),
        "operating_point": {k: point[k] for k in POINT_KEYS},
        "days_11_12_cost_at_these_thresholds": safe["total_cost"],
    }
    return summary, row_costs(y[check], probs[check], amounts[check], t_review, t_block)


# ---------------------------------------------------------------------------
# A. The ladder: one change at a time to the V2 LightGBM
# ---------------------------------------------------------------------------

def fit_lightgbm_v2_params(frame, config):
    """5 seeds of LightGBM with V2's settings, on the config's days, features and weights."""
    features = FEATURE_SETS[config["features"]]
    X, y = get_days(frame, config["train_days"], features)
    weights = amount_weights(X, y) if config["amount_weights"] else None
    return {seed: make_lgbm(seed, scale_pos_weight=1).fit(X, y, sample_weight=weights) for seed in FINAL_SEEDS}


def stage_ladder(args):
    frame = load_search_frame(FEATURES_V3)
    X_val, y_val, days = validation_data(frame)
    log(f"validation days {VALIDATION_DAYS}: {len(y_val):,} rows, {int(y_val.sum())} fraud")

    v2 = joblib.load(settings.models / "runs" / "phase2" / "lightgbm.joblib")
    v2_raw = {seed: predict_fraud_proba(m, X_val[FEATURE_SETS["v2"]]) for seed, m in v2["models"].items()}
    best = {"train_days": [5, 6, 7], "amount_weights": False, "features": "v2"}
    best_scores, best_name = np.mean(list(v2_raw.values()), axis=0), "v2_deployed"
    steps = [{"step": "v2_deployed", "config": dict(best), "pr_auc": pr_auc(y_val, best_scores)}]
    log(f"V2 deployed: PR-AUC {steps[0]['pr_auc']:.4f}")
    scores = {"v2_deployed": best_scores}

    changes = [
        ("more_training_data", {"train_days": ALL_DAYS}),
        ("amount_weights", {"amount_weights": True}),
        ("recipient_profile_features", {"features": "v3"}),
    ]
    for name, change in changes:
        config = {**best, **change}
        models = fit_lightgbm_v2_params(frame, config)
        features = FEATURE_SETS[config["features"]]
        raw = np.mean([predict_fraud_proba(m, X_val[features]) for m in models.values()], axis=0)
        scores[name] = raw
        test, keep = better(y_val, raw, best_scores)
        steps.append({"step": name, "config": config, "compared_with": best_name, "pr_auc": pr_auc(y_val, raw),
                      "paired_test": test, "kept": keep})
        log(f"{name}: PR-AUC {steps[-1]['pr_auc']:.4f}, difference {test['difference']:+.4f} "
            f"[{test['ci_low']:+.4f}, {test['ci_high']:+.4f}] -> {'kept' if keep else 'not kept'}")
        if keep:
            best, best_scores, best_name = config, raw, name

    # Do the recipient-profile features make the model lean less on the three
    # balance features? Same data setup, with and without them.
    def ensemble_for(features):
        models = fit_lightgbm_v2_params(frame, {**best, "features": features})
        return np.mean([predict_fraud_proba(m, X_val[FEATURE_SETS[features]]) for m in models.values()], axis=0)

    balance_scores = {f: ensemble_for(f) for f in ("v2", "v2_no_balance", "v3", "v3_no_balance")}
    balance_check = {
        "data_setup": {k: v for k, v in best.items() if k != "features"},
        "pr_auc": {f: pr_auc(y_val, s) for f, s in balance_scores.items()},
        "v3_no_balance_minus_v2_no_balance": paired_bootstrap_difference(
            y_val, balance_scores["v3_no_balance"], balance_scores["v2_no_balance"]),
    }
    log("balance check (PR-AUC): " + ", ".join(f"{f} {v:.4f}" for f, v in balance_check["pr_auc"].items()))
    test = balance_check["v3_no_balance_minus_v2_no_balance"]
    log(f"  without balance features, recipient profile adds {test['difference']:+.4f} "
        f"[{test['ci_low']:+.4f}, {test['ci_high']:+.4f}]")

    payload = {"evaluated_on": "days 11-13 (5-seed averages)", "lightgbm_settings": "V2 settings (not tuned)",
               "steps": steps, "chosen": best, "chosen_step": best_name, "balance_check": balance_check}
    save_json("ladder.json", payload)
    np.savez_compressed(RUN_DIR / "ladder_scores.npz", y=y_val, **scores,
                        **{f"balance_check_{f}": v for f, v in balance_scores.items()})
    record("v3_ladder", payload, f"{SOURCE} ladder")


def stage_cost_check(args):
    """Re-check the log(amount) weighting on cost, not just PR-AUC.

    For V2 and for the weighted variant from the ladder: thresholds chosen on
    days 11-12, cost measured on day 13. The weighting is kept only if it is
    significantly cheaper (paired bootstrap over day-13 rows).
    """
    ladder = json.loads((OUT_DIR / "ladder.json").read_text(encoding="utf-8"))
    saved = np.load(RUN_DIR / "ladder_scores.npz")
    y, amounts, days = validation_labels()
    assert np.array_equal(y, saved["y"]), "row order differs from the ladder run"

    results, costs = {}, {}
    for name in ("v2_deployed", "amount_weights"):
        results[name], costs[name] = day13_check(saved[name], y, amounts, days)
        point = results[name]["operating_point"]
        log(f"{name:14s} day 13: cost {point['total_cost']:,.0f}, fraud allowed {point['fraud_allowed']}, "
            f"legit reviewed {point['legit_reviewed_pct']:.2%}, wrongful blocks {point['legit_blocked']}")

    test = paired_cost_difference(costs["amount_weights"], costs["v2_deployed"])
    keep = bool(test["significant"] and test["difference"] < 0)
    log(f"weighted minus V2, day-13 cost: {test['difference']:+,.0f} [{test['ci_low']:+,.0f}, {test['ci_high']:+,.0f}]"
        f" -> {'keep' if keep else 'do not keep'}")

    weighted_step = next(s for s in ladder["steps"] if s["step"] == "amount_weights")
    payload = {
        "question": "does weighting frauds by log(amount) lower the cost?",
        "method": "Platt calibration and safe thresholds fitted on days 11-12; cost measured on day 13; "
                  "paired bootstrap over day-13 rows",
        "weighted_variant_config": weighted_step["config"],
        "day_13": results,
        "weighted_minus_v2_cost": test,
        "kept": keep,
    }
    save_json("weighting_cost_check.json", payload)
    record("v3_weighting_cost_check", payload, f"{SOURCE} cost-check")


def stage_recipient_check(args):
    """Which recipient features were tested in the ladder, and why fan-in was not.

    Fan-in (distinct senders to a recipient in the last 24h / 7d) equals the
    existing transaction counts when no sender ever pays the same recipient
    twice, so this counts repeated sender-recipient pairs in the raw data.
    """
    raw = pd.read_csv(settings.raw, usecols=["nameOrig", "nameDest"], engine="pyarrow",
                      dtype={"nameOrig": "string[pyarrow]", "nameDest": "string[pyarrow]"})
    repeated = int(raw.duplicated(["nameOrig", "nameDest"]).sum())
    log(f"{len(raw):,} transactions, {repeated:,} repeat an earlier sender-recipient pair")

    payload = {
        "tested_in_ladder": {
            "recipient_transfer_share": "share of the recipient's earlier incoming transactions that were TRANSFERs",
            "recipient_hours_since_previous": "hours since the recipient's previous incoming transaction "
                                              "(-1 when there is none)",
        },
        "tested_feature_columns": RECIPIENT_PROFILE,
        "planned_but_not_tested": {
            "fan_in_24h_and_7d": "distinct senders to the recipient in the last 24h / 7d. Not added: with no "
                                 "repeated sender-recipient pairs it is identical to the existing "
                                 "destination_transactions_last_24h / _last_7d features",
        },
        "rows_checked": len(raw),
        "repeated_sender_recipient_pairs": repeated,
    }
    save_json("recipient_feature_check.json", payload)
    record("v3_recipient_feature_check", payload, f"{SOURCE} recipient-check")


# ---------------------------------------------------------------------------
# B. Tune and refit each model family
# ---------------------------------------------------------------------------

def load_ladder():
    return json.loads((OUT_DIR / "ladder.json").read_text(encoding="utf-8"))["chosen"]


def stage_tune(args):
    setup = load_ladder()
    features = FEATURE_SETS[setup["features"]]
    frame = load_search_frame(features)
    X_val, y_val, _ = validation_data(frame, features)
    storage = f"sqlite:///{(RUN_DIR / 'optuna.db').as_posix()}"

    for family in args.families:
        start = time.time()
        n_trials = args.trials or TRIALS[family]
        log(f"\n[{family}] {n_trials} trials on folds {FOLDS}, setup {setup}")
        study = run_search(family, frame, features, setup["amount_weights"], n_trials, storage, log=log)
        params = final_params(family, study)
        log(f"  best mean fold PR-AUC {study.best_value:.4f} with {params}")

        X, y = get_days(frame, setup["train_days"], features)
        weights = amount_weights(X, y) if setup["amount_weights"] else None
        models, raw = {}, {}
        for seed in FINAL_SEEDS:
            models[seed] = fit_model(family, params, seed, X, y, weights)
            raw[seed] = predict_fraud_proba(models[seed], X_val[features])
            log(f"  seed {seed}: days 11-13 PR-AUC {pr_auc(y_val, raw[seed]):.4f}")

        trials = [{"number": t.number, "state": t.state.name, "value": t.value, "params": t.params,
                   **t.user_attrs} for t in study.trials]
        tuning = {"family": family, "label": LABELS[family], "setup": setup, "n_trials": n_trials,
                  "folds": FOLDS, "best_fold_pr_auc": study.best_value, "best_params": params,
                  "trials": trials, "minutes": (time.time() - start) / 60}
        if family in NEURAL:
            tuning["training_rows"] = (f"all fraud rows + {NET_LEGIT_SHARE[family]:.0%} of legitimate rows; "
                                       f"early stopping on {NET_VALIDATION_LEGIT_SHARE:.0%} of the fold's "
                                       "legitimate validation rows")
        (OUT_DIR / "tuning").mkdir(exist_ok=True)
        save_json(f"tuning/{family}.json", tuning)
        joblib.dump({"family": family, "params": params, "setup": setup, "features": features,
                     "models": models, "validation_raw": raw}, RUN_DIR / f"{family}.joblib", compress=3)
        log(f"  done in {(time.time() - start) / 60:.0f} min")


def stage_record(args):
    """Write every finished tuning run to metrics.json (section v3_tuning).

    Safe to run while other families are still tuning: it records what is
    finished and lists the rest as not finished.
    """
    y_val = np.load(RUN_DIR / "ladder_scores.npz")["y"]
    families = {}
    for family in LABELS:
        json_path = OUT_DIR / "tuning" / f"{family}.json"
        model_path = RUN_DIR / f"{family}.joblib"
        if not (json_path.exists() and model_path.exists()):
            continue
        tuning = json.loads(json_path.read_text(encoding="utf-8"))
        validation_raw = joblib.load(model_path)["validation_raw"]
        assert all(len(s) == len(y_val) for s in validation_raw.values()), f"{family}: row count differs"
        seed_scores = {int(seed): pr_auc(y_val, s) for seed, s in validation_raw.items()}
        ensemble = np.mean(list(validation_raw.values()), axis=0)
        families[family] = {
            "label": tuning["label"],
            "trial_budget": tuning["n_trials"],
            "trial_states": dict(Counter(t["state"] for t in tuning["trials"])),
            "best_mean_fold_pr_auc": tuning["best_fold_pr_auc"],
            "best_params": tuning["best_params"],
            "training_rows": tuning.get("training_rows", "all rows of days 5-7"),
            "minutes": tuning["minutes"],
            "days_11_13_pr_auc_by_seed": seed_scores,
            "days_11_13_pr_auc_seed_mean": float(np.mean(list(seed_scores.values()))),
            "days_11_13_pr_auc_seed_std": float(np.std(list(seed_scores.values()))),
            "days_11_13_pr_auc_5_seed_average": pr_auc(y_val, ensemble),
        }
        log(f"{family:15s} best fold PR-AUC {tuning['best_fold_pr_auc']:.4f}; days 11-13: 5-seed average "
            f"{families[family]['days_11_13_pr_auc_5_seed_average']:.4f}")
    payload = {
        "setup": load_ladder(), "folds": FOLDS, "final_seeds": FINAL_SEEDS,
        "search": "Optuna TPE (seed 42), median pruning after fold 1; objective = mean PR-AUC over the two folds",
        "families": families,
        "not_finished": [f for f in LABELS if f not in families],
    }
    record("v3_tuning", payload, f"{SOURCE} record")
    return payload


# ---------------------------------------------------------------------------
# Compare everything on days 11-13
# ---------------------------------------------------------------------------

def logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def stacked_scores(member_scores, y, days):
    """Logistic regression on the members' scores (log-odds).

    Fitted on days 11-12 only, so its day-13 scores are out of sample. Day 13
    is the only day it is judged on (its days 11-12 scores are in sample).
    """
    inputs = np.column_stack([logit(s) for s in member_scores.values()])
    fit = days != CHECK_DAY
    stacker = LogisticRegression(max_iter=1000).fit(inputs[fit], y[fit])
    return stacker.predict_proba(inputs)[:, 1], stacker


def add_holm(rows, test_key, names):
    """Holm-correct one family of tests (one test per model), in place."""
    if not names:
        return
    adjusted = holm_adjust({n: rows[n][test_key]["p_normal"] for n in names})
    for name in names:
        rows[name][test_key]["p_holm"] = adjusted[name]
        rows[name][test_key]["significant_after_holm"] = bool(adjusted[name] < SIGNIFICANCE)


def verdict(test):
    if not test["significant_after_holm"]:
        return "comparable"
    return "significantly better" if test["difference"] > 0 else "significantly worse"


def stage_compare(args):
    stage_record(args)          # the tuning results go into metrics.json first
    ladder = json.loads((OUT_DIR / "ladder.json").read_text(encoding="utf-8"))
    frame = load_search_frame(FEATURES_V3)
    X_val, y_val, days = validation_data(frame)
    amounts = X_val["amount"].to_numpy()
    on_day_13 = days == CHECK_DAY
    y_13 = y_val[on_day_13]

    raw = {}          # name -> {seed: scores}
    v2 = joblib.load(settings.models / "runs" / "phase2" / "lightgbm.joblib")
    raw["v2_deployed"] = {s: predict_fraud_proba(m, X_val[FEATURE_SETS["v2"]]) for s, m in v2["models"].items()}
    v2_ladder = fit_lightgbm_v2_params(frame, ladder["chosen"])
    feats = FEATURE_SETS[ladder["chosen"]["features"]]
    raw["lightgbm_v2_settings"] = {s: predict_fraud_proba(m, X_val[feats]) for s, m in v2_ladder.items()}

    for family in LABELS:
        path = RUN_DIR / f"{family}.joblib"
        if path.exists():
            raw[family] = joblib.load(path)["validation_raw"]
        else:
            log(f"  {family}: not tuned, left out")

    autoencoder = joblib.load(settings.models / "runs" / "phase4" / "autoencoder.joblib")
    raw["autoencoder"] = {s: a.anomaly_score(X_val[autoencoder["feature_columns"]])
                          for s, a in autoencoder["autoencoders"].items()}

    labels = {"v2_deployed": "V2 deployed (LightGBM, days 5-7)",
              "lightgbm_v2_settings": "LightGBM, V2 settings, V3 data setup",
              "autoencoder": "Autoencoder (no labels, from V2)", **LABELS, "stacked": "Stacked ensemble"}

    # Every model: days 11-13 numbers, plus the day-13 check.
    rows, ensembles, day13_costs = {}, {}, {}
    for name, by_seed in raw.items():
        ensembles[name], rows[name] = summarise(by_seed, X_val, y_val, days)
        rows[name]["day_13"], day13_costs[name] = day13_check(ensembles[name], y_val, amounts, days)
        log(f"{name:22s} days 11-13 PR-AUC {rows[name]['pr_auc']['pr_auc']:.4f}   "
            f"day 13 PR-AUC {rows[name]['day_13']['pr_auc']['pr_auc']:.4f}, "
            f"day-13 cost {rows[name]['day_13']['operating_point']['total_cost']:,.0f}")

    # Stacked ensemble: the three boosting models plus the best network,
    # fitted on days 11-12 and judged on day 13 only.
    nets = [n for n in NEURAL if n in ensembles]
    members = [n for n in ("lightgbm", "catboost", "xgboost") if n in ensembles]
    if nets:
        members.append(max(nets, key=lambda n: rows[n]["pr_auc"]["pr_auc"]))
    if len(members) >= 2:
        stacked, stacker = stacked_scores({n: ensembles[n] for n in members}, y_val, days)
        ensembles["stacked"] = stacked
        day_13, day13_costs["stacked"] = day13_check(stacked, y_val, amounts, days)
        rows["stacked"] = {"day_13": day_13, "members": members,
                           "stacker_coefficients": dict(zip(members, stacker.coef_[0].tolist(), strict=True)),
                           "note": "fitted on days 11-12; judged on day 13 only, so it has no days 11-13 numbers"}
        log(f"{'stacked':22s} day 13 PR-AUC {day_13['pr_auc']['pr_auc']:.4f}, "
            f"day-13 cost {day_13['operating_point']['total_cost']:,.0f}  (members {members})")
        joblib.dump({"members": members, "stacker": stacker, "fitted_on_days": [11, 12]},
                    RUN_DIR / "stacker.joblib")

    # Step 4 of the ladder (a single planned test, like steps 1-3): does tuning beat V2's settings?
    main = "lightgbm_v2_settings"
    tuning_test = None
    if "lightgbm" in ensembles:
        tuning_test, keep = better(y_val, ensembles["lightgbm"], ensembles["lightgbm_v2_settings"])
        if keep:
            main = "lightgbm"

    # Paired tests against the main LightGBM and against V2 deployed. The
    # stacked ensemble is compared on day-13 rows only; everything else on days 11-13.
    for name in rows:
        rows[name]["label"] = labels.get(name, name)
        if name == "stacked":
            scores = {n: ensembles[n][on_day_13] for n in ("stacked", main, "v2_deployed")}
            rows[name]["vs_main_lightgbm"] = paired_bootstrap_difference(y_13, scores["stacked"], scores[main])
            rows[name]["vs_v2_deployed"] = paired_bootstrap_difference(y_13, scores["stacked"], scores["v2_deployed"])
            rows[name]["tests_on"] = "day 13"
            continue
        rows[name]["tests_on"] = "days 11-13"
        rows[name]["vs_main_lightgbm"] = paired_bootstrap_difference(y_val, ensembles[name], ensembles[main])
        rows[name]["vs_v2_deployed"] = paired_bootstrap_difference(y_val, ensembles[name], ensembles["v2_deployed"])
        rows[name]["roc_vs_main_lightgbm"] = paired_bootstrap_difference(
            y_val, ensembles[name], ensembles[main], metric="roc_auc")

    # Holm correction: one family of tests per comparison, across all models.
    add_holm(rows, "vs_main_lightgbm", [n for n in rows if n != main])
    add_holm(rows, "vs_v2_deployed", [n for n in rows if n != "v2_deployed"])
    add_holm(rows, "roc_vs_main_lightgbm", [n for n in rows if n not in (main, "stacked")])
    for name in rows:
        rows[name]["verdict_vs_main_lightgbm"] = (
            "reference" if name == main else verdict(rows[name]["vs_main_lightgbm"]))
        rows[name]["verdict_vs_v2_deployed"] = (
            "reference" if name == "v2_deployed" else verdict(rows[name]["vs_v2_deployed"]))

    # Day-13 cost of each model against V2, at thresholds chosen on days 11-12.
    for name in rows:
        if name != "v2_deployed":
            rows[name]["day_13"]["cost_minus_v2_deployed"] = paired_cost_difference(
                day13_costs[name], day13_costs["v2_deployed"])

    # Another model replaces the main LightGBM only with a significant win over it after Holm.
    challengers = [n for n in rows if n not in (main, "v2_deployed", "autoencoder")
                   and rows[n]["verdict_vs_main_lightgbm"] == "significantly better"]
    proposed = (max(challengers, key=lambda n: rows[n]["vs_main_lightgbm"]["difference"])
                if challengers else main)

    payload = {
        "evaluated_on": "days 11-13; hyperparameters chosen on folds inside days 5-10",
        "setup": ladder["chosen"], "main_lightgbm": main, "tuning_vs_v2_settings": tuning_test,
        "proposed_v3": proposed, "proposed_v3_vs_v2_deployed": rows[proposed]["vs_v2_deployed"],
        "significance_rule": (f"paired bootstrap with {settings.n_bootstrap} resamples; p-value from the bootstrap "
                              "standard error; Holm correction across all models within each comparison; "
                              f"significant when the Holm-adjusted p < {SIGNIFICANCE}"),
        "stacking_note": "the stacker is fitted on days 11-12 and judged on day 13 only",
        "cost_note": ("operating_point: thresholds chosen and cost measured on days 11-13 (in sample). "
                      "day_13.operating_point: calibration and thresholds fitted on days 11-12, cost on day 13 "
                      "(out of sample)."),
        "leaderboard": rows,
    }
    save_json("validation.json", payload)
    np.savez_compressed(RUN_DIR / "validation_scores.npz", y=y_val, day=days, amount=amounts, **ensembles)
    record("v3_validation", payload, f"{SOURCE} compare")
    log(f"\nproposed V3: {proposed}")


STAGES = {"ladder": stage_ladder, "cost-check": stage_cost_check, "recipient-check": stage_recipient_check,
          "tune": stage_tune, "record": stage_record, "compare": stage_compare}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=list(STAGES))
    parser.add_argument("families", nargs="*", default=[])
    parser.add_argument("--trials", type=int, help="override the trial budget (for quick checks)")
    args = parser.parse_args()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STAGES[args.stage](args)


if __name__ == "__main__":
    main()
