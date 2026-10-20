# Delay-aware retraining: what was built and what it showed

All figures are from `results/phase5/rolling.json` (`scripts/phase5_rolling.py`), `results/metrics.json` section `phase5_fallback` (`scripts/phase5_fallback.py`) and `results/retraining_log.json` (`scripts/retrain.py`).

## Rolling retraining with delayed labels

Each day from 11 to 16 the system is retrained before scoring that day. With a label delay of D days the supervised model only sees labels through day t−1−D. The autoencoder needs no labels and is refitted on the three most recent days regardless of D. Both systems are 5-seed LightGBM ensembles with settings fixed in Phase 2. The statistic is the mean of the six daily PR-AUCs; intervals come from 200 bootstrap resamples of rows within each day.

| Label delay | Supervised [95% CI] | Hybrid (+ anomaly score) [95% CI] | Hybrid − supervised [95% CI] |
|---|---|---|---|
| 0 days | 0.8760 [0.8641, 0.8909] | 0.8726 [0.8585, 0.8890] | −0.0034 [−0.0072, −0.0000] |
| 1 day | 0.8745 [0.8619, 0.8895] | 0.8718 [0.8590, 0.8866] | −0.0028 [−0.0066, +0.0013] |
| 2 days | 0.8754 [0.8635, 0.8902] | 0.8722 [0.8592, 0.8863] | −0.0032 [−0.0060, −0.0001] |
| 3 days | 0.8709 [0.8568, 0.8864] | 0.8708 [0.8576, 0.8851] | −0.0001 [−0.0059, +0.0067] |

Effect of the delay itself, against no delay on the same days:

| System | 1 day | 2 days | 3 days |
|---|---|---|---|
| Supervised | −0.0014 [−0.0036, +0.0005] | −0.0005 [−0.0033, +0.0019] | −0.0051 [−0.0100, −0.0000] |
| Hybrid | −0.0008 [−0.0033, +0.0014] | −0.0003 [−0.0034, +0.0032] | −0.0018 [−0.0053, +0.0023] |

**What this says, plainly.**

- **Label delay barely matters on this data.** One or two days of delay have no measurable effect. Three days costs the supervised system about 0.005 PR-AUC, with an interval that only just excludes zero. This agrees with the earlier finding (notebook 04) that the delay effect was not measurable.
- **The hybrid does not rescue anything.** It is never better than the supervised system, and at 0 and 2 days it is marginally worse. The gap does close as the delay grows (−0.0034 at 0 days, −0.0001 at 3 days), which is the direction the idea predicts, but with delays this short there is no benefit to claim.
- The likely reason is that fraud in PaySim does not change over the 12 days studied, so a model trained on older labels is still a good model. A simulator with stable fraud rules cannot show the value of fresh labels.
- The autoencoder on its own is erratic from one refit to the next: daily PR-AUC of 0.098, 0.037, 0.038, 0.097, 0.047 and 0.235 on days 11 to 16.

The claim that can honestly be made is that the system has a working delay-aware mechanism, not that the mechanism improved results here.

## Champion / challenger

At each retraining point a newly trained model (the challenger) replaces the serving model (the champion) only if it is significantly better on the most recent fully labelled day, a day neither model was trained on (paired bootstrap, lower bound of the 95% interval above zero). Every decision is logged in `rolling.json`.

| Label delay | Promotions in 6 retraining points | Mean daily PR-AUC with the gate | Always retraining |
|---|---|---|---|
| 0 days | 2 | 0.8756 | 0.8760 |
| 1 day | 1 | 0.8725 | 0.8745 |
| 2 days | 1 | 0.8708 | 0.8754 |
| 3 days | 2 | 0.8714 | 0.8709 |

The first promotion in each row is the initial deployment. The gate rarely promotes, because a retrained model is rarely significantly better here, and serving the older model costs at most 0.005 PR-AUC against retraining every day. Its value is operational: far fewer model changes for almost the same performance, each one justified by a logged test.

## Retraining job

`scripts/retrain.py` is the job a scheduler would run. It uses only matured labels, lets feedback labels from `POST /labels` override stored ones once they too have matured, trains a challenger, calibrates and thresholds it on the three most recent labelled days, applies the same gate, and registers a new model version only with `--promote`.

One run, as of day 14 with a 2-day label maturity (labels through day 11; trained on days 5-8, validated on days 9-11): challenger PR-AUC 0.8481 against 0.8445 for the serving model, difference +0.0035 (95% CI +0.0005 to +0.0079). Decision: promote. It was run without `--promote`, so the serving model was not changed.

## Serving

- The API returns the supervised score, the calibrated probability and the anomaly score for every transaction.
- **Fallback rule.** When the newest labels the supervised model was trained on are older than a configurable age (`FRAUD_MAX_LABEL_AGE_DAYS`, default 14 days), a transaction the supervised model would allow is sent to review if its anomaly score is in the top 1% of scores seen on unlabelled eval_2 traffic. The rule never lowers a tier.
- **What the fallback does on held-out when forced on:** it escalates 4,948 transactions, none of them fraud. Legitimate transactions reviewed rise from 2.79% to 3.20% and cost from 33,744 to 38,692, with no fraud rescued, because the supervised model already let none through. It is insurance for a stale model, and on this data the insurance was not needed.
- **Feedback loop.** `POST /labels` stores an outcome with the time of the transaction, the time the outcome became known and its source (analyst, customer or chargeback); `GET /labels` shows which are mature. A stronger source overrides a weaker one. The analyst review queue that will feed this endpoint is part of Phase 7.
