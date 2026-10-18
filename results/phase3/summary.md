# Simulator-artefact check: the model without the balance features

Same protocol, same data, same model family (LightGBM 5-seed ensemble on the leak-free features). The only change is removing `origin_balance_error`, `destination_balance_error`, `destination_balance_is_zero`.

| Features | Held-out PR-AUC [95% CI] | 5 seeds | Total cost | Fraud let through | Legit reviewed | Legit blocked | Block precision | Block recall |
|---|---|---|---|---|---|---|---|---|
| With balance features | 0.8632 [0.8384, 0.8835] | 0.8585 ± 0.0022 | 33,744 | 0 | 2.79% | 3 | 0.9953 | 0.7701 |
| Without balance features | 0.5028 [0.4659, 0.5341] | 0.5020 ± 0.0023 | 464,257 | 0 | 38.58% | 2 | 0.9928 | 0.3358 |

Paired bootstrap, PR-AUC without minus with: -0.3604 (95% CI -0.3900 to -0.3299), significant.

## What this says

The model depends heavily on the balance features: without them held-out PR-AUC falls from 0.8632 to 0.5028 and the share of fraud that can be blocked automatically falls from 77.0% to 33.6%. It still lets no held-out fraud through, but only by sending 38.6% of legitimate transactions to review instead of 2.8%.

The balance features compare an account's balance before and after a transaction. In PaySim that relationship is produced by the simulator's own bookkeeping, and fraud rows break it in a regular way, so part of what the model has learned is a property of the simulator. A real payment system would not necessarily expose post-transaction balances at scoring time, and real fraud would not necessarily leave this trace. The headline figures should therefore be read as an upper bound for this dataset, not as an estimate of performance on real payments.

The features have not been removed from the deployed model; that decision is left open.
