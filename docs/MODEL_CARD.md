# Model card: fraud_scorer v3

A short, honest description of the model this project deploys: what it is for, what it was trained and tested on, how well it does, and where it breaks. Every number here is copied from `results/metrics.json`; the section and the script that wrote it are given next to each table.

## The model

| | |
|---|---|
| Name and version | `fraud_scorer` v3 (`models/registry/fraud_scorer/v3`) |
| What it is | LightGBM, trained 5 times with different seeds (1, 2, 3, 4, 42); the five scores are averaged |
| Settings | 750 trees, learning rate 0.0206, 13 leaves, depth 3, at least 237 rows per leaf, fraud weight 1.45. Chosen by Optuna on folds inside days 5–10 |
| Probabilities | Platt calibration, fitted on days 11–13 |
| Decisions | ALLOW below 0.000316, REVIEW from 0.000316 up to 0.71, BLOCK at 0.71 and above. Chosen on days 11–13 |
| Inputs | 16: the amount, the sender's and recipient's balance mismatch, a zero-balance flag, five recipient-history counts and averages, the time of day (two columns) and the transaction type (five columns) |
| Explanations | SHAP values for every flagged transaction, turned into three plain-English reasons |
| Replaced | V2 (same family, default settings), on 6 November 2026, after the held-out test below |

## Intended use

- **Scoring PaySim-style mobile-money transactions** and deciding ALLOW, REVIEW or BLOCK so that the total cost stays low. A missed fraud costs its amount, a wrongful block 5, a review 1.
- **Supporting an analyst.** REVIEW opens a case with the reasons; a person makes the decision, and every decision goes to the audit log.
- **Teaching and demonstration.** The project shows an honest way to build, test and run a fraud model: a time split, a held-out test run once, confidence intervals, monitoring.

## Out of scope

- **Real payments.** The model was trained only on PaySim, a simulation. It has never seen real data, and its strongest inputs are the simulator's bookkeeping (see the failure modes). Do not use it to block real customers.
- **Other kinds of fraud or other data.** Card fraud, account takeover, money laundering, other payment systems. The inputs and the costs would all be different.
- **Fully automatic decisions about people.** The design assumes a person reviews the REVIEW tier and that blocked customers can appeal.
- **Judging individual customers.** The model scores transactions, not people. It does not use or produce anything about a customer's identity.
- **Different costs.** The thresholds are only right for the assumed costs (5 per wrongful block, 1 per review). With other costs they must be chosen again, on days 11–13, never on the held-out days. The dashboard's threshold simulator shows what changes.

## Data

PaySim: 6,362,620 simulated transactions over 743 hours, 8,213 of them fraud (0.1291%). Fraud only happens in TRANSFER and CASH_OUT. Only days 5–16 are used, because outside them the volume is tiny and erratic. The days are split in time, never shuffled:

| Days | Used for |
|---|---|
| 5–7 | Training the five models |
| 8–10 | Choosing the settings (Optuna folds inside days 5–10) |
| 11–13 | Comparing models, calibration, thresholds |
| 14–16 | The held-out test, run once at the end: 1,202,642 transactions, 822 fraud |

Days 14–16 are not completely fresh: the first version of the project (V1) tuned on them, and V2 was scored on them. V3 made every choice on days 5–13 and looked at days 14–16 once.

## Results on the held-out days (14–16)

The rule was written into the test script before it ran: V3 replaces V2 only if its PR-AUC is significantly higher (paired bootstrap, 95%) and its cost is not significantly higher. The script refuses to run a second time.

| | V3 | V2 | V3 − V2, 95% interval |
|---|---|---|---|
| PR-AUC | 0.8730 (0.8474 to 0.8927) | 0.8632 | +0.0098 (+0.0057 to +0.0149) |
| ROC-AUC | 0.9990 (0.9987 to 0.9992) | 0.9989 | +0.00009 |
| Total cost | 32,586 | 33,744 | −1,158 (−1,286.1 to −1,026.9) |
| Frauds let through | 0 of 822 | 0 | |
| Legitimate transactions reviewed | 2.69% | 2.79% | |
| Wrongful blocks | 3 | 3 | |
| Fraud blocked at once (block recall) | 77.4% | 77.0% | |
| Precision of blocks | 99.5% | 99.5% | |

Source: `v3_held_out` (`scripts/v3_phase3.py`). Intervals: 200 bootstrap resamples.

The total cost is 0 for missed fraud + 3 wrongful blocks × 5 + 32,571 reviews × 1 = 32,586.

### Against simple alternatives

| On days 14–16 | Total cost | Frauds let through |
|---|---|---|
| **V3** | **32,586** | **0** |
| Best hand-written rule (a rule engine, picked on days 5–10) | 196,675 | 1 |
| Send every transaction to review | 1,202,642 | 0 |
| Block nothing | 1,362,518,395.67 | 822 |

Sources: `rule_baselines` (`scripts/rule_baselines.py`), `v3_held_out`. The rule engine blocks a transfer or cash-out of the sender's whole balance and PaySim's own flag, and reviews transfers and cash-outs of 228,093.63 or more. It is strong for the same reason the model is: the whole-balance rule uses the simulator's bookkeeping.

### Calibration

On day 13 (Platt fitted on days 11–12): Brier score 0.000137, expected calibration error 0.000039, mean predicted probability 0.058% against an actual fraud rate of 0.062%. Source: `v3_curves` (`scripts/v3_curves.py`).

### Where it catches fraud

Every fraud on days 14–16 was sent to review or blocked, in every transaction type and every amount quartile. How much is blocked at once, and how much legitimate traffic is reviewed, differs by group; the Explainability page shows the table. Source: `v3_breakdown` (`scripts/v3_breakdown.py`).

## Failure modes

**1. It depends on the simulator's balance bookkeeping.** The sender's balance mismatch is the strongest input (SHAP, `v3_explainability`). When the same model family was retrained without the three balance inputs, PR-AUC fell from 0.8632 to 0.5028, and 38.58% of legitimate traffic would have needed review (`phase3_balance_ablation`, measured on V2). Real payment systems rarely leave such clean traces, so these results are an upper bound, not a forecast.

**2. A small shift in those inputs lets fraud through.** The stress test adds a constant to both balance-mismatch inputs on days 14–16 (`phase6_monitoring`, `scripts/run_monitoring.py`):

| Shift | PR-AUC | Frauds let through | Total cost | Monitoring alert |
|---|---|---|---|---|
| None | 0.8730 | 0 | 32,586 | – |
| × 1.5 | 0.8725 | 0 | 32,938 | no |
| + 500 | 0.8448 | **113** | 15,141,745.36 | yes |

Adding 500 is enough: 113 frauds are allowed and the cost goes up more than 400 times. The PSI alert fires, so the monitoring notices, but it does not prevent the damage. Someone has to act on the alert.

**3. The cost cliff.** The review threshold sits 2.17 times (in odds) below the point where the next fraud would slip under it (`v3_phase2_decisions`). If every probability shrank by more than that, one fraud would be allowed and the cost on days 11–13 would jump from 32,519 to 566,697.41.

**4. Few frauds.** 822 frauds on the held-out days. The intervals above are wide for that reason, and small differences between models are not real differences.

**5. Wrongful blocks.** 3 legitimate transactions were blocked on days 14–16. The Explainability page walks through one of them.

**6. Assumed costs.** 5 and 1 are assumptions, not business figures. With different costs, other thresholds would be cheaper.

**7. No label-delay benefit.** Retraining with labels that arrive late was built and tested. On this data a delay of up to 2 days made no significant difference, and the fallback rule rescued no fraud (`phase5_rolling`, `phase5_fallback`).

## Monitoring in place

- PSI for every input, every period, against days 5–7 (warning 0.10, alert 0.25).
- Review and block rates against bands.
- A novelty detector (an autoencoder, used only for monitoring) reports the share of unusual transactions.
- A retraining job with a champion/challenger gate: a retrained model replaces the serving one only if its PR-AUC on the most recent labelled days is significantly higher (`scripts/retrain.py`).

## How it was built

All code is in this repository: `src/fraud/` (features, models, evaluation, serving), `scripts/` (every experiment and result), `tests/`. The full story, with every model that was tried, is in the README and on the Model lab page of the dashboard.
