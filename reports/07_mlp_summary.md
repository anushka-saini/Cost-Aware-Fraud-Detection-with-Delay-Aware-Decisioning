# Deep learning comparison: MLP against LightGBM and Random Forest

Source: `notebooks/07_mlp_comparison.ipynb`. Exact figures: `reports/07_mlp_results.json`.

## What was tested

Whether a neural network underperforms the tree models on this project's data, as the literature on tabular fraud detection with few positive examples suggests. The test used the same training window (days 5-7, 778 fraud rows) and the same held-out set (days 14-16, 1,202,642 rows, 822 fraud) as the other models.

The network is a multi-layer perceptron with three hidden layers (64, 32 and 16 units), ReLU activations and dropout of 0.2. Class imbalance is handled by weighting fraud rows in the loss (`pos_weight`), the counterpart of `scale_pos_weight` in LightGBM. Continuous inputs were log-transformed and standardised using training statistics. The architecture and optimiser settings were fixed in advance and not tuned.

The same fairness rules as the Random Forest comparison were applied:

- The class weight was chosen by a sweep (1 to 500) scored on eval_1 (days 8-10), which was also used for early stopping. The held-out set was not used for either.
- The REVIEW and BLOCK thresholds were derived for the MLP with the cost function used for the other models (a missed fraud costs its amount, a wrongly blocked legitimate transaction costs 5, a review costs 1).
- PR-AUC is reported with a 95% confidence interval from 200 bootstrap resamples of the held-out set.

## Results

| Model | Held-out PR-AUC | 95% CI |
|---|---|---|
| LightGBM (`scale_pos_weight=20`) | 0.7198 | [0.6877, 0.7492] |
| Random Forest (`class_weight={0: 1, 1: 500}`) | 0.8335 | [0.8085, 0.8546] |
| MLP (hidden layers 64-32-16, dropout 0.2, `pos_weight=20`) | 0.8485 | [0.8217, 0.8720] |

Each model at its own cost-derived thresholds:

| Model | Review >= | Block >= | Total cost | Fraud allowed | Legitimate sent to review | Legitimate blocked | Block precision | Block recall |
|---|---|---|---|---|---|---|---|---|
| LightGBM | 0.0001 | 0.9 | 46,455,419 | 34 | 4.27% | 186 | 0.7745 | 0.7774 |
| Random Forest | 0.32 | 0.85 | 47,547 | 0 | 3.93% | 23 | 0.9639 | 0.7482 |
| MLP | 0.000631 | 0.95 | 48,551 | 1 | 4.01% | 12 | 0.9816 | 0.7798 |

## Interpretation

The expected result was not observed. The MLP did not underperform the tree models.

- On held-out PR-AUC, the MLP has the highest point estimate. Its confidence interval overlaps that of Random Forest, so the two cannot be separated; it does not overlap that of LightGBM.
- The result is stable: five training seeds gave held-out PR-AUC between 0.8485 and 0.8526 (standard deviation 0.0014), and validation PR-AUC varied only between 0.8271 and 0.8336 across class weights from 1 to 500.
- On total cost at each model's own thresholds, Random Forest remains lowest (47,547), with the MLP close (48,551) and LightGBM far higher (46,455,419). The MLP allows 1 fraud case through; Random Forest allows 0.
- The MLP flags all 37 of the held-out fraud cases that LightGBM scored below 0.1%.

## Limitations

- One configuration of each model was compared. The LightGBM model is the project's original configuration, which has a documented blind spot; the result does not show that neural networks outperform gradient boosting in general.
- Thresholds were swept on the held-out set for all three models, so the cost figures are in-sample with respect to the thresholds.
- The MLP's review threshold (0.000631) has little margin: cost rises to 49,987 at 0.001 and 117,960 at 0.001585. Random Forest's review threshold has the same property.
- PaySim is synthetic. Its generated fraud patterns may be easier for a small network to learn than real fraud. The same network was retrained on the ULB data in `06_cross_dataset_validation.ipynb`: held-out PR-AUC 0.7972 (95% CI [0.7312, 0.8613]), the highest point estimate there, with a confidence interval that overlaps the logistic regression baseline and LightGBM.
- The MLP is a comparison only. The deployed model is unchanged.
