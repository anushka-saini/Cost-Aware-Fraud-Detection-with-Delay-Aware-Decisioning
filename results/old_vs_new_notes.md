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
