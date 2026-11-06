# Monitoring: drift reports and stress test

All figures are from `results/monitoring/drift_reports.json` and `results/metrics.json` section `phase6_monitoring`, written by `scripts/run_monitoring.py` for the deployed model (`fraud_scorer` v3).

The same page for V2 is kept in [`v2_superseded/`](v2_superseded/).

## Alert rules

- **PSI:** above 0.25 is an alert, 0.10 to 0.25 a warning. It is computed for every model input and for the model's score, against the training days (5–7).
- **Review rate and block rate:** alert when outside 0.5× to 2× the rate seen on days 11–13, where the thresholds were set.
- **Novelty share:** the share of transactions the novelty detector calls unusual. The detector is the autoencoder from V2. In V3 it is no longer part of scoring and only feeds monitoring. It flags the 1% most unusual transactions of its training days, so about 1% is expected; alert outside 0.5× to 2× of that.
- **Override rate** (flagged transactions the customer pushed through anyway): alert above 30%.

The job writes one report per period. The API stores them in its database and serves them at `GET /monitoring/drift`.

## Reports for the three later periods

| Period | Status | Inputs flagged | Model-score PSI | Review rate | Block rate | Novelty share |
|---|---|---|---|---|---|---|
| Days 8–10 | warning | `destination_transactions_last_7d` 0.176 (warning) | 0.0016 | 2.78% | 0.049% | 0.68% |
| Days 11–13 | alert | `destination_transactions_last_24h` 0.262 (alert), `destination_transactions_last_7d` 0.164 (warning) | 0.0019 | 2.76% | 0.052% | 1.00% |
| Days 14–16 | alert | `destination_transactions_last_24h` 0.408 (alert), `destination_transactions_last_7d` 0.535 (alert) | 0.0023 | 2.71% | 0.053% | 0.72% |

The two recipient-activity counts drift steadily, as the original analysis found. Everything else stays steady: the model's score, both operating rates (inside their band of 1.38% to 5.53% for reviews) and the novelty share. So these alerts are a prompt to look, not evidence of a failing model; the model makes little use of those two counts. In production they would get a higher alert threshold, or be tracked separately, so they do not cause alert fatigue.

## Drift stress test

The stability above depends on the drifting inputs being unimportant. To test the unlucky case, the two balance-error inputs the model relies on most were shifted on days 14–16, and V3 re-scored with its thresholds unchanged.

| Scenario | PR-AUC | Total cost | Fraud let through | Legitimate reviewed | PSI alert? |
|---|---|---|---|---|---|
| No shift | 0.8730 | 32,586 | 0 | 2.69% | not applicable |
| Multiply both by 1.5 | 0.8725 | 32,938 | 0 | 2.72% | no |
| Add 500 to both | 0.8448 | 15,141,745.36 | 113 | 0.14% | yes |
| Add 1,000 | 0.8449 | 15,141,760.36 | 113 | 0.14% | yes |
| Add 2,500 | 0.8445 | 15,141,784.36 | 113 | 0.14% | yes |
| Add 5,000 | 0.8434 | 15,141,866.36 | 113 | 0.15% | yes |
| Add 10,000 | 0.8411 | 15,441,223.18 | 114 | 0.16% | yes |

- **Scaling does almost nothing.** Many of these values are exactly zero, and zero stays zero.
- **A constant offset is very costly.** 113 of the 822 frauds get through and the cost goes from 32,586 to about 15 million. The model reacts to these inputs being exactly zero, or exactly equal to the amount. V3 holds up better than V2 here (V2 let 157 through in the same test), but the weakness is the same.
- **Monitoring would catch it.** Every offset raises a PSI alert on the shifted inputs. The review rate also drops to about 0.15%, far below its 1.38% lower limit, so the rate alert fires too.

So the model is steady under the drift that actually happened, and fragile under drift in the inputs it depends on. Monitoring is what stands between that fragility and a silent failure.
