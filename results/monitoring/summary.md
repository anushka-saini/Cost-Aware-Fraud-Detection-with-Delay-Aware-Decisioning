# Monitoring: drift reports and stress test

All figures are from `results/monitoring/drift_reports.json` and `results/metrics.json` section `phase6_monitoring`, written by `scripts/run_monitoring.py` for the deployed model (`fraud_scorer` v2).

## Alert rules

- PSI above 0.25: alert. PSI from 0.10 to 0.25: warning. Computed for every model feature, the model score and the anomaly score, against the training window (days 5-7).
- Review rate and block rate: alert when outside 0.5× to 2× the rate observed on eval_2, where the thresholds were set.
- Override rate (flagged transactions the user pushed through anyway): alert above 30%.

The job writes one report per period, and the API serves them at `GET /monitoring/drift`. For now reports are stored in a JSON file; they move to the database in Phase 7.

## Reports for the three later periods

| Period | Status | Features flagged | Model-score PSI | Anomaly-score PSI | Review rate | Block rate |
|---|---|---|---|---|---|---|
| eval_1 (days 8-10) | warning | `destination_transactions_last_7d` 0.176 (warning) | 0.0006 | 0.0106 | 2.89% | 0.049% |
| eval_2 (days 11-13) | alert | `destination_transactions_last_24h` 0.262 (alert), `destination_transactions_last_7d` 0.164 (warning) | 0.0020 | 0.0122 | 2.87% | 0.051% |
| held-out (days 14-16) | alert | `destination_transactions_last_24h` 0.408 (alert), `destination_transactions_last_7d` 0.535 (alert) | 0.0009 | 0.0181 | 2.80% | 0.053% |

The two recipient-activity counts drift steadily, as the original analysis found. The model score, the anomaly score and both operating rates stay stable, so these alerts are a prompt to investigate, not evidence of a failing model: the model barely uses those two features. In production these two features would be given a higher alert threshold or tracked separately so they do not cause alert fatigue.

## Drift stress test on the deployed model

The earlier stability result depends on the drifting features being unimportant. To test the unlucky case, the two balance-error features the model relies on most were shifted on the held-out period and the deployed model re-scored (thresholds unchanged).

| Scenario | PR-AUC | Total cost | Fraud let through | Legit reviewed | PSI alert raised? |
|---|---|---|---|---|---|
| No shift | 0.8632 | 33,744 | 0 | 2.79% | n/a |
| Multiply both by 1.5 | 0.8639 | 33,998 | 0 | 2.81% | no |
| Add 500 to both | 0.6959 | 69,751,524 | 157 | 0.60% | yes |
| Add 1,000 | 0.6945 | 70,017,450 | 158 | 0.61% | yes |
| Add 2,500 | 0.6916 | 70,282,017 | 159 | 0.63% | yes |
| Add 5,000 | 0.6866 | 70,282,609 | 159 | 0.68% | yes |
| Add 10,000 | 0.6758 | 70,220,381 | 159 | 0.80% | yes |

- A proportional change does nothing, because many values are exactly zero and stay zero.
- A constant offset is catastrophic: about 158 of 822 frauds are let through and cost rises from 33,744 to about 70 million. The model is keying on these features being exactly zero or exactly equal to the amount.
- **The monitoring would catch it.** Every additive scenario raises a PSI alert on the shifted features, and the review rate falls from 2.79% to under 1%, which is outside the 0.5× to 2× band and raises a rate alert too.

So the model is stable under the drift that actually occurred and fragile under drift in the features it depends on, and the monitoring is what stands between that fragility and a silent failure.
