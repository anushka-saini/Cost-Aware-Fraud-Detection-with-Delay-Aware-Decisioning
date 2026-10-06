# Cross-dataset validation: ULB credit card fraud

Source: `notebooks/06_cross_dataset_validation.ipynb`. Exact figures: `reports/06_cross_dataset_results.json`.

## What was tested

Whether the modelling methodology used on PaySim transfers to a second dataset. The trained PaySim model was not reused, because the two datasets share no features. The ULB / Kaggle credit card fraud dataset has 284,807 transactions over two days, 492 of them fraud (0.173%), described by 28 anonymised PCA components and the transaction amount.

The method was kept the same as on PaySim:

- Time-ordered split: day 1 for training (144,786 rows, 281 fraud), the first half of day 2 for validation (47,445 rows, 96 fraud), the second half of day 2 held out (92,576 rows, 115 fraud).
- LightGBM with the same tree settings as the PaySim model (200 trees, depth 5, 15 leaves, `min_child_samples=100`).
- The MLP from the deep learning comparison (hidden layers 64-32-16, dropout 0.2, weighted loss), retrained on this dataset with its class weight and early stopping chosen on the validation period.
- `scale_pos_weight` derived for this dataset by a sweep scored on the validation period, not copied from PaySim.
- PR-AUC with a 95% confidence interval from 200 bootstrap resamples of the held-out period.

No setting was adjusted after seeing held-out results.

## Results

| Model | Held-out PR-AUC | 95% CI |
|---|---|---|
| LightGBM, `scale_pos_weight=5` (derived on validation) | 0.6818 | [0.5947, 0.7662] |
| LightGBM, `scale_pos_weight=20` (PaySim's value, reused) | 0.6343 | [0.5549, 0.7202] |
| Logistic regression baseline | 0.7365 | [0.6570, 0.8148] |
| MLP (64-32-16, dropout 0.2, `pos_weight=50`) | 0.7972 | [0.7312, 0.8613] |

For reference, the PaySim LightGBM model scored 0.7198 (95% CI [0.6877, 0.7492]) on its own held-out period. The two figures are not directly comparable, because PR-AUC depends on the fraud base rate and the datasets differ.

## Interpretation

The methodology transferred only partly.

The procedure itself carried over: the same split discipline, model family and evaluation ran on a different dataset and produced a held-out PR-AUC of 0.6818. The derived class weight (5) differed from PaySim's (20), and using the dataset's true imbalance ratio (514) as the weight reduced held-out PR-AUC to 0.0385.

The MLP from `07_mlp_comparison.ipynb`, retrained on this dataset, has the highest held-out point estimate (0.7972; 0.7915 to 0.8030 across five seeds). Its confidence interval overlaps both the baseline's and LightGBM's, so with 115 held-out fraud cases it cannot be separated from either. The ordering of the three point estimates (MLP, then the linear baseline, then LightGBM) is the same direction as the PaySim result that the MLP was ahead of LightGBM.

Three results were unfavourable:

1. LightGBM did not outperform the logistic regression baseline. The baseline's point estimate was higher (0.7365 against 0.6818) and the confidence intervals overlap. On PaySim, LightGBM was clearly ahead of the same baseline (0.7113 against 0.5676).
2. The regularisation settings chosen for PaySim did not prevent overfitting. Training PR-AUC was 1.0000 against 0.6818 held out; on PaySim the gap was 0.8206 against 0.7198.
3. The weight sweep was unstable. Validation PR-AUC ranged from 0.4527 to 0.7642 across weights 1 to 200 without a consistent trend, so the selected weight is the best of an erratic set. The held-out difference between the derived weight and PaySim's 20 lies within the overlap of their confidence intervals.

## Limitations

- The held-out period contains 115 fraud cases, so the confidence interval is wide (about +/-0.09).
- Training is deterministic for this configuration (no row or column subsampling), so a five-seed check returned identical scores and gives no information about run-to-run variance.
- Only the modelling and evaluation method was tested. Cost-based thresholds, drift analysis and SHAP checks were not repeated on this dataset.
- The reason the linear baseline is competitive on this data was not investigated.
- The MLP used a batch size of 512 here against 4096 on PaySim, because the training set is nine times smaller.
