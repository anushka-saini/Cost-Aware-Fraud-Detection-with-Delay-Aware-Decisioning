"""Phase 5: rolling retraining with delayed labels.

Every day t from 11 to 16 the system is retrained before scoring that day.
With a label delay of D days, the supervised model may only use labels from
days up to t-1-D. The autoencoder needs no labels, so it is refitted on the
three most recent days (t-3 .. t-1) whatever D is.

Two systems are compared as D grows from 0 to 3:

  supervised   LightGBM on the leak-free features
  hybrid       the same, plus the anomaly score of the freshly fitted autoencoder

Both use the settings fixed earlier (scale_pos_weight chosen on eval_1 in
Phase 2); nothing is tuned here. Each is a 5-seed averaged ensemble. The
statistic is the mean of the six daily PR-AUCs, with a bootstrap interval from
resampling rows within each day.

A champion/challenger policy is then replayed on the supervised system: a
retrained model replaces the serving one only if it is significantly better on
the most recent fully labelled day, a day neither model was trained on.

    python scripts/phase5_rolling.py
"""

import json
import time

import joblib
import numpy as np

from fraud.autoencoder import TabularAutoencoder
from fraud.config import get_settings
from fraud.data import get_days, load_dense
from fraud.evaluation import _FastAP, _resample_counts, paired_bootstrap_difference, pr_auc
from fraud.features import FEATURES_V2
from fraud.models import make_lgbm, predict_fraud_proba
from fraud.protocol import FINAL_SEEDS
from fraud.results import _jsonable, record
from fraud.serving import ANOMALY_COLUMN

SOURCE = "scripts/phase5_rolling.py"
FIRST_DAY = 5
SCORING_DAYS = [11, 12, 13, 14, 15, 16]
DELAYS = [0, 1, 2, 3]
AE_WINDOW = 3
HYBRID_FEATURES = [*FEATURES_V2, ANOMALY_COLUMN]


def log(message):
    print(message, flush=True)


def fit_ensemble(X, y, weight):
    return [make_lgbm(seed, scale_pos_weight=weight).fit(X, y) for seed in FINAL_SEEDS]


def ensemble_score(models, X):
    return np.mean([predict_fraud_proba(m, X) for m in models], axis=0)


def stratified_bootstrap(daily, n_iterations=200, seed=42):
    """daily: {day: (y, {variant: scores})}. Returns resampled means of daily PR-AUC per variant."""
    variants = list(next(iter(daily.values()))[1])
    samples = {v: np.zeros(n_iterations) for v in variants}
    for i, (_day, (y, scores)) in enumerate(sorted(daily.items())):
        aps = {v: _FastAP(y, scores[v]) for v in variants}
        for k, counts in enumerate(_resample_counts(len(y), n_iterations, seed + i)):
            for v in variants:
                samples[v][k] += aps[v](counts)
    return {v: s / len(daily) for v, s in samples.items()}


def main():
    settings = get_settings()
    out_dir = settings.results / "phase5"
    out_dir.mkdir(parents=True, exist_ok=True)
    weight = json.loads((settings.results / "phase2" / "selection.json").read_text(encoding="utf-8"))["lightgbm"]["weight"]
    log(f"LightGBM scale_pos_weight = {weight} (fixed in Phase 2 on eval_1)")

    frame = load_dense(FEATURES_V2)
    days = lambda a, b: list(range(a, b + 1))  # noqa: E731

    # --- supervised models, one per last-labelled day -------------------------------------------
    supervised = {}
    for end in range(FIRST_DAY + 1, max(SCORING_DAYS)):          # train on days 5..end
        start = time.time()
        X, y = get_days(frame, days(FIRST_DAY, end), FEATURES_V2)
        supervised[end] = fit_ensemble(X, y, weight)
        log(f"supervised trained on days {FIRST_DAY}-{end}: {len(X):,} rows, {int(y.sum())} fraud [{time.time() - start:.0f}s]")

    # --- daily scores -----------------------------------------------------------------------------
    daily = {d: {} for d in DELAYS}              # delay -> day -> (y, {"supervised": s, "hybrid": s})
    anomaly_only = {}
    for t in SCORING_DAYS:
        start = time.time()
        X_day, y_day = get_days(frame, [t], FEATURES_V2)
        y_arr = y_day.to_numpy()

        # Label-free: refit on the most recent traffic, whatever the label delay is.
        recent, _ = get_days(frame, days(t - AE_WINDOW, t - 1), FEATURES_V2)
        validation, _ = get_days(frame, [t - 1], FEATURES_V2)
        ae = TabularAutoencoder(seed=settings.seed).fit(recent, validation)
        day_anomaly = ae.anomaly_score(X_day)
        anomaly_only[t] = pr_auc(y_arr, day_anomaly)
        X_day_hybrid = X_day.assign(**{ANOMALY_COLUMN: day_anomaly})
        log(f"day {t}: autoencoder refitted on days {t - AE_WINDOW}-{t - 1} (epoch {ae.best_epoch_}), "
            f"anomaly-only PR-AUC {anomaly_only[t]:.4f} [{time.time() - start:.0f}s]")

        for delay in DELAYS:
            end = t - 1 - delay
            X_lab, y_lab = get_days(frame, days(FIRST_DAY, end), FEATURES_V2)
            hybrid_models = fit_ensemble(X_lab.assign(**{ANOMALY_COLUMN: ae.anomaly_score(X_lab)})[HYBRID_FEATURES],
                                         y_lab, weight)
            scores = {"supervised": ensemble_score(supervised[end], X_day),
                      "hybrid": ensemble_score(hybrid_models, X_day_hybrid[HYBRID_FEATURES])}
            daily[delay][t] = (y_arr, scores)
            log(f"  delay {delay}d (labels through day {end}): supervised {pr_auc(y_arr, scores['supervised']):.4f}, "
                f"hybrid {pr_auc(y_arr, scores['hybrid']):.4f}")
        log(f"  day {t} done in {time.time() - start:.0f}s")

    # --- summary by delay ---------------------------------------------------------------------------
    by_delay = []
    for delay in DELAYS:
        boot = stratified_bootstrap(daily[delay])
        row = {"delay_days": delay}
        for variant in ("supervised", "hybrid"):
            per_day = {t: pr_auc(y, s[variant]) for t, (y, s) in daily[delay].items()}
            low, high = np.percentile(boot[variant], [2.5, 97.5])
            row[variant] = {"mean_daily_pr_auc": float(np.mean(list(per_day.values()))),
                            "ci_low": float(low), "ci_high": float(high), "per_day": per_day}
        diff = boot["hybrid"] - boot["supervised"]
        low, high = np.percentile(diff, [2.5, 97.5])
        row["hybrid_minus_supervised"] = {
            "difference": row["hybrid"]["mean_daily_pr_auc"] - row["supervised"]["mean_daily_pr_auc"],
            "ci_low": float(low), "ci_high": float(high), "significant": bool(low > 0 or high < 0)}
        by_delay.append(row)
        log(f"delay {delay}d: supervised {row['supervised']['mean_daily_pr_auc']:.4f} "
            f"[{row['supervised']['ci_low']:.4f}, {row['supervised']['ci_high']:.4f}] | hybrid "
            f"{row['hybrid']['mean_daily_pr_auc']:.4f} [{row['hybrid']['ci_low']:.4f}, {row['hybrid']['ci_high']:.4f}] | "
            f"diff {row['hybrid_minus_supervised']['difference']:+.4f} "
            f"[{row['hybrid_minus_supervised']['ci_low']:+.4f}, {row['hybrid_minus_supervised']['ci_high']:+.4f}]")

    # Effect of delay itself: delay D against no delay, same days, same resamples.
    delay_effect = []
    for variant in ("supervised", "hybrid"):
        base = stratified_bootstrap({t: (y, {"x": s[variant]}) for t, (y, s) in daily[0].items()})["x"]
        for delay in DELAYS[1:]:
            other = stratified_bootstrap({t: (y, {"x": s[variant]}) for t, (y, s) in daily[delay].items()})["x"]
            low, high = np.percentile(other - base, [2.5, 97.5])
            point = by_delay[delay][variant]["mean_daily_pr_auc"] - by_delay[0][variant]["mean_daily_pr_auc"]
            delay_effect.append({"variant": variant, "delay_days": delay, "difference_vs_no_delay": float(point),
                                 "ci_low": float(low), "ci_high": float(high), "significant": bool(low > 0 or high < 0)})
            log(f"{variant}: delay {delay}d vs 0d: {point:+.4f} [{low:+.4f}, {high:+.4f}]")

    # --- champion / challenger (supervised system) ------------------------------------------------
    policies = {}
    for delay in DELAYS:
        champion, decisions, served = None, [], {}
        for t in SCORING_DAYS:
            validation_day = t - 1 - delay                 # most recent fully labelled day
            challenger = validation_day - 1                # trained on days 5..validation_day-1
            entry = {"scoring_day": t, "validation_day": validation_day,
                     "challenger_trained_through_day": challenger}
            if champion is None:
                champion = challenger
                entry.update(decision="promote", reason="no model in service yet")
            elif challenger == champion:
                entry.update(decision="keep", reason="no newer labelled data than the champion already has")
            else:
                X_val, y_val = get_days(frame, [validation_day], FEATURES_V2)
                test = paired_bootstrap_difference(y_val.to_numpy(), ensemble_score(supervised[challenger], X_val),
                                                   ensemble_score(supervised[champion], X_val))
                promote = test["ci_low"] > 0
                entry.update(champion_trained_through_day=champion, pr_auc_difference=test["difference"],
                             ci_low=test["ci_low"], ci_high=test["ci_high"],
                             decision="promote" if promote else "keep",
                             reason=("challenger significantly better on the validation day" if promote
                                     else "challenger not significantly better on the validation day"))
                if promote:
                    champion = challenger
            entry["serving_model_trained_through_day"] = champion
            X_day, y_day = get_days(frame, [t], FEATURES_V2)
            served[t] = pr_auc(y_day.to_numpy(), ensemble_score(supervised[champion], X_day))
            entry["serving_pr_auc_on_scoring_day"] = served[t]
            decisions.append(entry)
        policies[delay] = {
            "decisions": decisions, "n_promotions": sum(d["decision"] == "promote" for d in decisions),
            "mean_daily_pr_auc": float(np.mean(list(served.values()))),
            "always_retrain_mean_daily_pr_auc": by_delay[delay]["supervised"]["mean_daily_pr_auc"],
        }
        log(f"champion/challenger, delay {delay}d: {policies[delay]['n_promotions']} promotions, mean daily PR-AUC "
            f"{policies[delay]['mean_daily_pr_auc']:.4f} (always retrain: {policies[delay]['always_retrain_mean_daily_pr_auc']:.4f})")

    payload = {
        "design": {"scoring_days": SCORING_DAYS, "delays": DELAYS, "seeds": FINAL_SEEDS,
                   "supervised_training": "all labelled dense-window days from day 5 through day t-1-D",
                   "autoencoder_training": f"days t-{AE_WINDOW} to t-1, no labels, early stopping on day t-1",
                   "statistic": "mean of daily PR-AUC; 95% CI from 200 bootstrap resamples of rows within each day",
                   "lightgbm_scale_pos_weight": weight,
                   "note": "No setting is tuned in this experiment. It scores days 14-16 as part of the rolling "
                           "evaluation, with configurations fixed in earlier phases."},
        "by_delay": by_delay, "delay_effect": delay_effect,
        "anomaly_only_daily_pr_auc": anomaly_only,
        "champion_challenger": policies,
    }
    (out_dir / "rolling.json").write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    record("phase5_rolling", payload, SOURCE)
    joblib.dump({"supervised_through_day": supervised}, settings.models / "runs" / "phase5_supervised.joblib", compress=3)


if __name__ == "__main__":
    main()
