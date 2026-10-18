# Deep learning components: what they added

All figures are from `results/phase4/final.json` and `results/phase2/final.json` (held-out days 14-16, 1,202,642 rows, 822 fraud, scored once), written by `scripts/phase4_autoencoder.py` and `scripts/run_protocol.py`.

## MLP (supervised baseline)

Re-run under the honest protocol on the leak-free features (class weight and early stopping on eval_1, calibration and thresholds on eval_2).

| | Held-out PR-AUC [95% CI] | 5 seeds | Cost | Fraud let through | Legit reviewed |
|---|---|---|---|---|---|
| MLP (64-32-16), 5-seed ensemble | 0.8493 [0.8242, 0.8734] | 0.8468 ± 0.0011 | 59,162 | 1 | 4.90% |
| LightGBM, 5-seed ensemble | 0.8632 [0.8384, 0.8835] | 0.8585 ± 0.0022 | 33,744 | 0 | 2.79% |

The MLP is the most stable model across seeds and comes second. LightGBM is ahead by 0.0140 PR-AUC (paired bootstrap CI 0.0074 to 0.0192), a small but significant difference.

## Autoencoder (label-free)

A 16-12-6-3-6-12-16 autoencoder trained on all 1,311,225 training-window transactions with no labels and no subsampling, early-stopped on eval_1 reconstruction loss. The anomaly score is the reconstruction error.

| Autoencoder alone | Held-out PR-AUC [95% CI] | Fraud in the top 1% most anomalous | Fraud in the top 5% |
|---|---|---|---|
| Seed 42 (fixed in advance; used in the hybrid) | 0.2091 [0.1779, 0.2430] | 71.2% | 92.3% |
| Mean score of the 5 seeds | 0.1453 [0.1248, 0.1701] | 61.4% | 87.6% |

- The fraud base rate is 0.068%, so a PR-AUC of 0.21 is roughly 300 times chance, achieved without a single label.
- **It is unstable across seeds.** Held-out PR-AUC for the five seeds: 0.2091, 0.0608, 0.0618, 0.2232, 0.0817 (mean 0.1273, std 0.0730). Lower reconstruction loss did not mean better fraud ranking. The seed-42 figure should not be quoted without this spread.
- As a ranking model it is far below the supervised models. Its value is that it needs no labels.

## Hybrid (LightGBM + anomaly score as a feature)

| | Held-out PR-AUC [95% CI] | 5 seeds | Cost | Fraud let through | Legit reviewed | Legit blocked |
|---|---|---|---|---|---|---|
| Supervised only | 0.8632 [0.8384, 0.8835] | 0.8585 ± 0.0022 | 33,744 | 0 | 2.79% | 3 |
| Hybrid | 0.8590 [0.8338, 0.8808] | 0.8542 ± 0.0028 | 698,581 | 2 | 2.98% | 11 |

Paired bootstrap, hybrid minus supervised: −0.0043 (95% CI −0.0093 to +0.0015), **not significant**.

**The hybrid did not help.** Adding the anomaly score did not improve ranking, and at thresholds fixed on eval_2 the hybrid let 2 frauds through (662,490 in total) where the supervised model let none through. The supervised LightGBM ensemble stays as the deployed model.

## Blind-spot check

The 41 held-out frauds (5%) to which the supervised ensemble gives its lowest scores, worth 7,421,667:

- 4 are in the top 1% most anomalous transactions, 15 in the top 5%, 29 in the top 10%.
- Their median anomaly percentile is 0.931, against 0.997 for fraud overall (759 of 822 frauds are in the top 5%).

So the autoencoder does see some of what the supervised model is unsure about, but these are also the frauds that look most like ordinary traffic, and it sees them less clearly than it sees typical fraud. It is a partial second opinion, not a fix for the blind spot. Note that none of these 41 are actually let through by the supervised model at its deployed threshold.

## Not done

The optional FT-Transformer stretch was not attempted. Training time on this laptop was already the limiting factor.
