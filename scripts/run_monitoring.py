"""Drift monitoring job and drift stress test for the deployed model.

For each later period (eval_1, eval_2, held-out) against the training window:
PSI of every feature, of the model score and of the anomaly score, plus the
review and block rates against the band expected from eval_2. Reports are
written to results/monitoring/drift_reports.json, which the API serves at
GET /monitoring/drift. Run it on a schedule (cron / Task Scheduler) in
production; here it is run once over the historical periods.

The stress test then shifts the two balance-error features on the held-out
period and re-scores, to see how the deployed model degrades when the features
it relies on most move.

    python scripts/run_monitoring.py
"""

import os

import numpy as np

from fraud.config import get_settings
from fraud.data import SPLITS, get_split, load_dense
from fraud.evaluation import cost_report, pr_auc
from fraud.monitoring import DriftReportStore, build_report
from fraud.registry import load_bundle
from fraud.results import record
from fraud.serving import ANOMALY_COLUMN, Scorer

SOURCE = "scripts/run_monitoring.py"
PERIODS = ["eval_1", "eval_2", "held_out"]
STRESS_FEATURES = ["origin_balance_error", "destination_balance_error"]
ADDITIVE_SHIFTS = [500, 1000, 2500, 5000, 10000]
MULTIPLICATIVE = [1.5]


def log(message):
    print(message, flush=True)


def tier_rates(scored):
    return float((scored["tier"] == "REVIEW").mean()), float((scored["tier"] == "BLOCK").mean())


def main():
    settings = get_settings()
    scorer = Scorer(load_bundle(os.environ.get("FRAUD_MODEL_NAME", "fraud_scorer")))
    columns = scorer.base_columns
    frame = load_dense(columns)
    store = DriftReportStore(settings.results / "monitoring" / "drift_reports.json")

    X_ref, _ = get_split(frame, "train", columns)
    ref = scorer.score_frame(X_ref)
    score_names = {"model_score": "fraud_probability"} | ({ANOMALY_COLUMN: ANOMALY_COLUMN} if ANOMALY_COLUMN in ref else {})
    reference_scores = {name: ref[col].to_numpy() for name, col in score_names.items()}

    scored, reports = {}, {}
    for period in PERIODS:
        X, _ = get_split(frame, period, columns)
        scored[period] = scorer.score_frame(X)
    expected_review, expected_block = tier_rates(scored["eval_2"])     # thresholds were set on eval_2

    for period in PERIODS:
        X, _ = get_split(frame, period, columns)
        review_rate, block_rate = tier_rates(scored[period])
        report = build_report(
            period=period, days=SPLITS[period], reference=X_ref, current=X, feature_columns=columns,
            reference_scores=reference_scores,
            current_scores={name: scored[period][col].to_numpy() for name, col in score_names.items()},
            review_rate=review_rate, expected_review_rate=expected_review,
            block_rate=block_rate, expected_block_rate=expected_block,
            model={"name": scorer.name, "version": scorer.version},
        )
        store.add(report)
        reports[period] = report
        flagged = ", ".join(f"{a['name']} ({a['level']}, {a['value']:.3f})" for a in report["alerts"]) or "none"
        log(f"{period}: status {report['status']}; review rate {review_rate:.4f}, block rate {block_rate:.5f}; flagged: {flagged}")

    # --- drift stress test on the features the model relies on most -----------------------------
    X_held, y_held = get_split(frame, "held_out", columns)
    y, amounts = y_held.to_numpy(), X_held["amount"].to_numpy()

    def evaluate(X):
        out = scorer.score_frame(X)
        report = cost_report(y, out["fraud_probability"].to_numpy(), amounts, scorer.t_review, scorer.t_block)
        return {"pr_auc": pr_auc(y, out["raw_score"].to_numpy()),
                **{k: report[k] for k in ("total_cost", "fraud_allowed", "legit_reviewed_pct", "legit_blocked", "block_recall")}}

    baseline = evaluate(X_held)
    log(f"stress baseline: PR-AUC {baseline['pr_auc']:.4f}, cost {baseline['total_cost']:,.0f}, fraud allowed {baseline['fraud_allowed']}")
    stress = []
    for kind, values in (("multiply", MULTIPLICATIVE), ("add", ADDITIVE_SHIFTS)):
        for value in values:
            shifted = X_held.copy()
            for feature in STRESS_FEATURES:
                shifted[feature] = shifted[feature] * value if kind == "multiply" else shifted[feature] + value
            result = evaluate(shifted)
            psi_rows = build_report(period="stress", days=[], reference=X_ref, current=shifted, feature_columns=STRESS_FEATURES,
                                    reference_scores={}, current_scores={})["psi"]
            stress.append({"kind": kind, "value": value, **result, "psi": {r["name"]: r["psi"] for r in psi_rows},
                           "would_alert": any(r["level"] == "alert" for r in psi_rows)})
            log(f"  {kind} {value}: PR-AUC {result['pr_auc']:.4f}, cost {result['total_cost']:,.0f}, fraud allowed "
                f"{result['fraud_allowed']}, legit reviewed {result['legit_reviewed_pct']:.2%}, PSI alert: {stress[-1]['would_alert']}")

    record("phase6_monitoring", {
        "model": {"name": scorer.name, "version": scorer.version},
        "alert_rules": {"psi_alert_above": 0.25, "psi_warning_from": 0.10,
                        "review_and_block_rate_band": "0.5x to 2x the rate observed on eval_2",
                        "override_rate_band": "0 to 30% of flagged transactions"},
        "reports": reports,
        "stress_test": {"split": "held_out (days 14-16)", "features_shifted": STRESS_FEATURES,
                        "baseline": baseline, "scenarios": stress},
        "share_of_zero_values_held_out": {f: float(np.mean(X_held[f] == 0)) for f in STRESS_FEATURES},
    }, SOURCE)


if __name__ == "__main__":
    main()
