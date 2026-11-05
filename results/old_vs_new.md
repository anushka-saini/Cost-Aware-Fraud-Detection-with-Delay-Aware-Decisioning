# Old numbers against honest numbers

Every "new" figure below is read from a results file written by code in this repository. "Old" figures are the ones previously reported in the README and notebooks 02, 05 and 07.

**What changed in the protocol.** Previously the class weights of LightGBM and Random Forest and the REVIEW / BLOCK thresholds of all models were chosen by looking at the held-out set (days 14-16). Now class weights are chosen on eval_1 (days 8-10), calibration and thresholds on eval_2 (days 11-13), and the held-out set is scored once. Results are reported over 5 training seeds and for the 5-seed probability-averaged ensemble.

## Leak-free features (hourly totals replaced by hour of day), honest protocol

Source: `results/phase2/final.json` (held-out days 14-16, 1,202,642 rows, 822 fraud), written by `scripts/run_protocol.py --tag phase2`.

### Held-out PR-AUC

| Model | Old | Old 95% CI | New: 5-seed mean ± std | New: seed 42 [95% CI] | New: 5-seed ensemble [95% CI] |
|---|---|---|---|---|---|
| Logistic regression | not reported | not reported | 0.6142 ± 0.0000 | 0.6142 [0.5772, 0.6453] | 0.6142 [0.5772, 0.6453] |
| LightGBM | 0.7198 | 0.6877 to 0.7492 | 0.8585 ± 0.0022 | 0.8591 [0.8335, 0.8799] | 0.8632 [0.8384, 0.8835] |
| Random Forest | 0.8335 | 0.8085 to 0.8546 | 0.8261 ± 0.0022 | 0.8228 [0.7942, 0.8476] | 0.8265 [0.7979, 0.8503] |
| MLP (64-32-16) | 0.8485 | 0.8217 to 0.8720 | 0.8468 ± 0.0011 | 0.8457 [0.8203, 0.8694] | 0.8493 [0.8242, 0.8734] |

### What was chosen, and where

| Model | Old class weight | New class weight (chosen on eval_1) | Calibration (chosen on eval_2) | Old flaw |
|---|---|---|---|---|
| Logistic regression | balanced | balanced | isotonic | not previously evaluated on held-out |
| LightGBM | 20 | 1 | platt | weight and thresholds chosen on held-out |
| Random Forest | 500 | 100 | isotonic | weight and thresholds chosen on held-out |
| MLP (64-32-16) | 20 | 1 | platt | weight chosen on eval_1 (sound); thresholds chosen on held-out |

### Operating point on held-out (thresholds fixed on eval_2, 5-seed ensemble)

| Model | | Review ≥ | Block ≥ | Total cost | Fraud let through | Legit reviewed | Legit blocked | Block precision | Block recall | Block F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| Logistic regression | **new** | 8.913e-06 | 0.85 | 5,043,316 | 4 | 6.82% | 7 | 0.9776 | 0.3710 | 0.5379 |
| LightGBM | old | 0.0001 | 0.9 | 46,455,419 | 34 | 4.27% | 186 | 0.7745 | 0.7774 | 0.7760 |
| LightGBM | **new** | 0.0001461 | 0.775 | 33,744 | 0 | 2.79% | 3 | 0.9953 | 0.7701 | 0.8683 |
| Random Forest | old | 0.32 | 0.85 | 47,547 | 0 | 3.93% | 23 | 0.9639 | 0.7482 | 0.8425 |
| Random Forest | **new** | 1e-07 | 0.67 | 130,171 | 2 | 3.92% | 3 | 0.9952 | 0.7616 | 0.8629 |
| MLP (64-32-16) | old | 0.000631 | 0.95 | 48,551 | 1 | 4.01% | 12 | 0.9816 | 0.7798 | 0.8692 |
| MLP (64-32-16) | **new** | 1.996e-05 | 0.83 | 59,162 | 1 | 4.90% | 1 | 0.9983 | 0.7007 | 0.8234 |

Old thresholds are on raw model scores; new thresholds are on calibrated probabilities, so the two threshold columns are not on the same scale.

### Seed spread at the deployed thresholds (each seed calibrated and thresholded on eval_2)

| Model | Total cost: mean ± std | min to max | Fraud let through: mean (min to max) | Legit reviewed: mean |
|---|---|---|---|---|
| Logistic regression | 5,043,316 ± 0 | 5,043,316 to 5,043,316 | 4.0 (4 to 4) | 6.82% |
| LightGBM | 517,890 ± 551,161 | 32,112 to 1,202,032 | 0.0 (0 to 0) | 43.07% |
| Random Forest | 95,582 ± 40,778 | 47,491 to 145,702 | 1.4 (0 to 3) | 3.93% |
| MLP (64-32-16) | 65,652 ± 8,818 | 52,186 to 77,057 | 1.6 (1 to 2) | 5.44% |

### Threshold robustness on held-out (ensemble; cost change when one threshold moves)

| Model | Threshold | −0.05 | −0.02 | +0.02 | +0.05 | odds ÷2 | odds ×2 |
|---|---|---|---|---|---|---|---|
| Logistic regression | review | -3,840,951 | -3,840,951 | +16,659,482 | +27,753,920 | +0 | +0 |
| Logistic regression | block | -2 | -1 | +1 | +1 | -2 | +2 |
| LightGBM | review | +1,168,277 | +1,168,277 | +19,089,297 | +23,420,856 | +9,355 | -2,578 |
| LightGBM | block | +7 | -3 | -3 | +4 | +13 | +12 |
| Random Forest | review | +1,071,857 | +1,071,857 | +18,906,466 | +28,132,027 | +0 | +0 |
| Random Forest | block | +0 | +0 | +0 | +0 | +0 | +0 |
| MLP (64-32-16) | review | +1,142,908 | +1,142,908 | +15,014,110 | +22,071,436 | +9,297 | -5,599 |
| MLP (64-32-16) | block | -19 | -7 | +5 | +7 | -22 | +15 |

### Is one model better than another? (paired bootstrap on held-out PR-AUC, ensembles)

| A | B | PR-AUC(A) − PR-AUC(B) | 95% CI of the difference | Significant? |
|---|---|---|---|---|
| Logistic regression | LightGBM | -0.2490 | -0.2749 to -0.2247 | yes |
| Logistic regression | Random Forest | -0.2123 | -0.2387 to -0.1895 | yes |
| Logistic regression | MLP (64-32-16) | -0.2351 | -0.2610 to -0.2088 | yes |
| LightGBM | Random Forest | +0.0367 | +0.0258 to +0.0466 | yes |
| LightGBM | MLP (64-32-16) | +0.0140 | +0.0074 to +0.0192 | yes |
| Random Forest | MLP (64-32-16) | -0.0228 | -0.0328 to -0.0137 | yes |

### Blind spot (real fraud given a raw score below 0.1%)

| Model | Held-out frauds below 0.1% | Their total amount | Of those, allowed at the deployed threshold |
|---|---|---|---|
| Logistic regression (ensemble) | 1 of 822 | 4,892,193 | 1 |
| LightGBM (ensemble) | 20 of 822 | 5,653,119 | 0 |
| Random Forest (ensemble) | 0 of 822 | 0 | 0 |
| MLP (64-32-16) (ensemble) | 50 of 822 | 2,850,159 | 1 |

Old figure: LightGBM (weight 20) scored 37 of 822 held-out frauds below 0.1%, worth 48,707,865.

### Class-weight sweeps on eval_1 (mean PR-AUC over 3 seeds)

- **Logistic regression**: balanced → 0.5867. Chosen: balanced.
- **LightGBM**: 1 → 0.8430, 5 → 0.8340, 10 → 0.8137, 20 → 0.7483, 50 → 0.3022, 100 → 0.1465, 200 → 0.1285, 500 → 0.0880. Chosen: 1.
- **Random Forest**: 20 → 0.7889, 50 → 0.8011, 100 → 0.8059, 200 → 0.7971, 500 → 0.7877, 1000 → 0.7724. Chosen: 100.
- **MLP (64-32-16)**: 1 → 0.8217, 5 → 0.8193, 20 → 0.8213, 100 → 0.8158, 500 → 0.8120. Chosen: 1.

## Same 17 features, honest protocol

Source: `results/phase1/final.json` (held-out days 14-16, 1,202,642 rows, 822 fraud), written by `scripts/run_protocol.py --tag phase1`.

### Held-out PR-AUC

| Model | Old | Old 95% CI | New: 5-seed mean ± std | New: seed 42 [95% CI] | New: 5-seed ensemble [95% CI] |
|---|---|---|---|---|---|
| Logistic regression | not reported | not reported | 0.5772 ± 0.0000 | 0.5772 [0.5424, 0.6120] | 0.5772 [0.5424, 0.6120] |
| LightGBM | 0.7198 | 0.6877 to 0.7492 | 0.8647 ± 0.0017 | 0.8652 [0.8409, 0.8870] | 0.8662 [0.8420, 0.8880] |
| Random Forest | 0.8335 | 0.8085 to 0.8546 | 0.8236 ± 0.0052 | 0.8234 [0.7974, 0.8449] | 0.8246 [0.7981, 0.8465] |
| MLP (64-32-16) | 0.8485 | 0.8217 to 0.8720 | 0.8501 ± 0.0014 | 0.8485 [0.8217, 0.8720] | 0.8519 [0.8258, 0.8750] |

### What was chosen, and where

| Model | Old class weight | New class weight (chosen on eval_1) | Calibration (chosen on eval_2) | Old flaw |
|---|---|---|---|---|
| Logistic regression | balanced | balanced | isotonic | not previously evaluated on held-out |
| LightGBM | 20 | 1 | platt | weight and thresholds chosen on held-out |
| Random Forest | 500 | 200 | isotonic | weight and thresholds chosen on held-out |
| MLP (64-32-16) | 20 | 20 | platt | weight chosen on eval_1 (sound); thresholds chosen on held-out |

### Operating point on held-out (thresholds fixed on eval_2, 5-seed ensemble)

| Model | | Review ≥ | Block ≥ | Total cost | Fraud let through | Legit reviewed | Legit blocked | Block precision | Block recall | Block F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| Logistic regression | **new** | 4.467e-05 | 0.68 | 5,556,819 | 11 | 27.52% | 15 | 0.9556 | 0.3929 | 0.5569 |
| LightGBM | old | 0.0001 | 0.9 | 46,455,419 | 34 | 4.27% | 186 | 0.7745 | 0.7774 | 0.7760 |
| LightGBM | **new** | 5.501e-05 | 0.865 | 50,338 | 0 | 4.17% | 0 | 1.0000 | 0.7603 | 0.8639 |
| Random Forest | old | 0.32 | 0.85 | 47,547 | 0 | 3.93% | 23 | 0.9639 | 0.7482 | 0.8425 |
| Random Forest | **new** | 1e-07 | 0.85 | 197,393 | 4 | 4.06% | 5 | 0.9911 | 0.6740 | 0.8023 |
| MLP (64-32-16) | old | 0.000631 | 0.95 | 48,551 | 1 | 4.01% | 12 | 0.9816 | 0.7798 | 0.8692 |
| MLP (64-32-16) | **new** | 1.819e-07 | 0.705 | 400,707 | 0 | 33.32% | 1 | 0.9984 | 0.7506 | 0.8569 |

Old thresholds are on raw model scores; new thresholds are on calibrated probabilities, so the two threshold columns are not on the same scale.

### Seed spread at the deployed thresholds (each seed calibrated and thresholded on eval_2)

| Model | Total cost: mean ± std | min to max | Fraud let through: mean (min to max) | Legit reviewed: mean |
|---|---|---|---|---|
| Logistic regression | 5,556,819 ± 0 | 5,556,819 to 5,556,819 | 11.0 (11 to 11) | 27.52% |
| LightGBM | 256,932 ± 415,867 | 34,770 to 1,088,381 | 0.0 (0 to 0) | 21.35% |
| Random Forest | 208,251 ± 62,710 | 132,213 to 289,935 | 4.0 (2 to 7) | 6.36% |
| MLP (64-32-16) | 417,494 ± 100,885 | 237,909 to 537,046 | 0.0 (0 to 0) | 34.72% |

### Threshold robustness on held-out (ensemble; cost change when one threshold moves)

| Model | Threshold | −0.05 | −0.02 | +0.02 | +0.05 | odds ÷2 | odds ×2 |
|---|---|---|---|---|---|---|---|
| Logistic regression | review | -4,354,440 | -4,354,440 | +28,349,506 | +34,353,060 | +0 | -1 |
| Logistic regression | block | +0 | +0 | +0 | +0 | +0 | +10 |
| LightGBM | review | +1,151,679 | +1,151,679 | +13,120,813 | +18,925,180 | +36,149 | -10,755 |
| LightGBM | block | -4 | +1 | +7 | +21 | -12 | +30 |
| Random Forest | review | +1,004,715 | +1,004,715 | +32,712,400 | +39,232,548 | +0 | +0 |
| Random Forest | block | -16 | +0 | +0 | +0 | -17 | +0 |
| MLP (64-32-16) | review | +801,322 | +801,322 | +18,120,337 | +23,165,368 | +88,594 | -117,280 |
| MLP (64-32-16) | block | -5 | -1 | +1 | +3 | +18 | +11 |

### Is one model better than another? (paired bootstrap on held-out PR-AUC, ensembles)

| A | B | PR-AUC(A) − PR-AUC(B) | 95% CI of the difference | Significant? |
|---|---|---|---|---|
| Logistic regression | LightGBM | -0.2890 | -0.3165 to -0.2623 | yes |
| Logistic regression | Random Forest | -0.2474 | -0.2750 to -0.2194 | yes |
| Logistic regression | MLP (64-32-16) | -0.2747 | -0.3019 to -0.2463 | yes |
| LightGBM | Random Forest | +0.0416 | +0.0339 to +0.0503 | yes |
| LightGBM | MLP (64-32-16) | +0.0143 | +0.0086 to +0.0197 | yes |
| Random Forest | MLP (64-32-16) | -0.0273 | -0.0359 to -0.0187 | yes |

### Blind spot (real fraud given a raw score below 0.1%)

| Model | Held-out frauds below 0.1% | Their total amount | Of those, allowed at the deployed threshold |
|---|---|---|---|
| Logistic regression (ensemble) | 1 of 822 | 4,892,193 | 1 |
| LightGBM (ensemble) | 23 of 822 | 1,741,096 | 0 |
| Random Forest (ensemble) | 0 of 822 | 0 | 0 |
| MLP (64-32-16) (ensemble) | 0 of 822 | 0 | 0 |

Old figure: LightGBM (weight 20) scored 37 of 822 held-out frauds below 0.1%, worth 48,707,865.

### Class-weight sweeps on eval_1 (mean PR-AUC over 3 seeds)

- **Logistic regression**: balanced → 0.5676. Chosen: balanced.
- **LightGBM**: 1 → 0.8396, 5 → 0.8031, 10 → 0.8040, 20 → 0.7092, 50 → 0.3358, 100 → 0.1370, 200 → 0.1498, 500 → 0.0695. Chosen: 1.
- **Random Forest**: 20 → 0.6730, 50 → 0.7871, 100 → 0.8003, 200 → 0.8095, 500 → 0.8090, 1000 → 0.7954. Chosen: 200.
- **MLP (64-32-16)**: 1 → 0.8271, 5 → 0.8319, 20 → 0.8330, 100 → 0.8308, 500 → 0.8292. Chosen: 20.

## Other experiments under the same protocol

| Experiment | Result | Source |
|---|---|---|
| Model without the three balance features | PR-AUC 0.5028 [0.4659, 0.5341] against 0.8632 with them; 38.58% of legitimate traffic reviewed against 2.79% | `results/phase3/summary.md` |
| Autoencoder alone (no labels) | PR-AUC 0.2091 [0.1779, 0.2430] for the pre-fixed seed; 0.1273 ± 0.0730 over 5 seeds | `results/phase4/summary.md` |
| Hybrid: LightGBM + anomaly score | PR-AUC 0.8590 [0.8338, 0.8808]; difference from supervised −0.0043 (CI −0.0093 to +0.0015), not significant; 2 frauds let through | `results/phase4/summary.md` |
| Label delay, rolling retraining | 1 to 2 days: no measurable effect. 3 days: −0.0051 (CI −0.0100 to −0.0000) | `results/phase5/summary.md` |
| Drift stress test, deployed model | Adding a constant to the balance-error features: PR-AUC about 0.69, about 158 of 822 frauds let through, cost about 70 million; PSI alert fires | `results/monitoring/summary.md` |

## Claims in the earlier report that change or no longer hold

Each row gives the earlier wording, what the honest protocol found, and wording that can be used instead.

| Earlier claim | What holds now | Suggested wording |
|---|---|---|
| "The deployed model is the Random Forest." | The deployed model is a 5-seed LightGBM ensemble on leak-free features. Its PR-AUC is significantly higher than the Random Forest's (+0.0367, CI +0.0258 to +0.0466). | "The deployed model is an ensemble of five LightGBM models, Platt-calibrated, chosen because it ranks significantly better than the Random Forest and the MLP on the held-out period." |
| "Random Forest PR-AUC 0.8335 (95% CI 0.8085 to 0.8546)." | 0.8265 [0.7979, 0.8503]. The earlier figure used a class weight picked on the held-out set. | "Random Forest: PR-AUC 0.8265 (95% CI 0.7979 to 0.8503), 0.8261 ± 0.0022 over five seeds." |
| "LightGBM PR-AUC 0.7198." | 0.8632 [0.8384, 0.8835]. The old model used a class weight of 20; with no class weighting, chosen on the validation period, it is the best model. | "LightGBM was originally mis-tuned. With its class weight chosen on a validation period it reaches PR-AUC 0.8632 (95% CI 0.8384 to 0.8835)." |
| "On the held-out set it let none of the 822 fraud cases through." (Random Forest) | In-sample: the thresholds were chosen on the held-out set. With thresholds fixed on an earlier period the Random Forest lets 2 through (0 to 3 across seeds). The deployed LightGBM ensemble lets 0 through, and so does each of its five seeds. | "With thresholds fixed on days 11–13, the deployed ensemble let none of the 822 held-out frauds through while sending 2.79% of legitimate transactions to review." |
| "Total cost 47,547." | That was the Random Forest with thresholds tuned on the test set; honestly it is 130,171. The deployed ensemble costs 33,744. | "Total cost of the deployed model on the held-out period: 33,744." |
| "It wrongly blocked 23 legitimate transactions." | The deployed ensemble wrongly blocks 3 (block precision 99.5%, recall 77.0%). | "3 legitimate transactions wrongly blocked out of 1,201,820." |
| "LightGBM scored 37 real fraud cases at under 0.1%, which explains the 46 million cost." | True of the mis-tuned model only. The retuned ensemble gives 20 held-out frauds a raw score below 0.1%, and none of them is allowed through at its deployed threshold. | "The blind spot was largely a tuning artefact: after retuning, no low-scored fraud is let through." |
| "The MLP surprised me: it matched the best tree model." | Still a strong, very stable second (0.8493, std 0.0011 over seeds), but LightGBM is significantly ahead (+0.0140, CI +0.0074 to +0.0192). | "The MLP is a close and very stable second; the LightGBM ensemble is ahead by a small but significant margin." |
| "Three hourly features leak. Disclosed, not fixed." | Fixed. They are replaced by the hour of day, at no measurable cost (0.8632 without them, 0.8662 with). | "The leaky hourly totals were removed and replaced by the hour of day, with no loss of performance." |
| "The API fills missing hourly features with training medians." | Removed. The service computes every feature itself, and a test checks 1,000 held-out transactions against the offline pipeline to 1e-6. | "A train/serve parity test guarantees the service scores a transaction exactly as the evaluation code does." |
| "Delay-aware decisioning." (title and framing) | A delay-aware mechanism exists and is tested: retraining on matured labels only, a champion/challenger gate, a label store, an anomaly-score fallback for a stale model. No benefit was measured: delays of 1 to 2 days change nothing and 3 days costs 0.005 PR-AUC. | "The system includes a delay-aware retraining mechanism. On this dataset label delays of up to three days had almost no effect, so no performance benefit is claimed for it." |
| "PR-AUC stays flat under drift." | True for the drift that occurred, because the drifting features are ones the model barely uses. Under a constant shift in the balance-error features the deployed model lets about 158 of 822 frauds through. | "The model is stable under the drift observed and fragile under drift in the features it relies on; monitoring raises an alert in that case." |
| "Thresholds 0.32 and 0.85." | Those were on uncalibrated Random Forest scores. The deployed thresholds are on calibrated probabilities: review from 0.000146, block from 0.775. | "Review from a calibrated fraud probability of 0.000146, block from 0.775." |
| Any statement that performance would carry over to real payments. | Without the three balance features PR-AUC falls from 0.8632 to 0.5028. Much of the signal is a regularity of the simulator. | "These results are an upper bound for PaySim, not an estimate of performance on real payment data." |
