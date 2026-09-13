# Cost-Aware Fraud Detection with Delay-Aware Decisioning

A fraud detection pipeline built on the PaySim synthetic mobile-money dataset
(~6.36M transactions), designed around four questions a real fraud system has
to answer beyond "can it classify fraud":

1. Does the model stay reliable over time, or silently decay? (**drift**)
2. Does it still work when fraud labels arrive late, as they do in the real
   world? (**label delay**)
3. What decision should the system actually take, given that false positives
   and false negatives have very different costs? (**cost-aware thresholding**)
4. Can the model's decisions be explained? (**SHAP**)

This repo documents the full pipeline — including two data-integrity bugs
that were found, diagnosed, and fixed mid-project, and the impact each had on
the reported numbers. That process is described in detail below because it's
as much a part of the result as the final metrics are.

## Final results (fully corrected pipeline)

| Metric | Value |
|---|---|
| Baseline PR-AUC (LightGBM, single seed) | 0.6684 |
| Baseline PR-AUC (5-seed mean, seed variance ± ~0.06) | 0.6563 |
| Delay-aware model PR-AUC (5-seed rank-ensemble) | 0.7671 |
| Cost-optimal review threshold | t_review ≈ 0.0001 |
| Cost-optimal block threshold | underdetermined by cost alone in 0.7–0.95 |

**Drift:** two destination-transaction-velocity features
(`destination_transactions_last_24h`, `destination_transactions_last_7d`)
show real, escalating distributional drift across the 12-day evaluation
window (PSI up to 0.54). All other features and — critically — overall model
PR-AUC remain statistically stable (overlapping 95% CIs across all three
evaluation periods). SHAP confirms the model places low importance on the
drifting features, which is why the drift doesn't propagate to a performance
drop.

**Label delay:** no statistically distinguishable effect of a 2-day label
delay on PR-AUC once training-seed variance is accounted for. Single-seed
delay sweeps showed a non-monotonic pattern; averaging across 5 seeds showed
this was training variance (std ≈ 0.03–0.08), not a real delay effect.

**Cost-thresholding:** under the stated cost assumptions (missed fraud = full
transaction amount, false block = 5 units, false review = 1 unit), the
review threshold does essentially all the cost-minimizing work; the block
threshold is close to cost-irrelevant across a wide range.

**Explainability:** `origin_balance_error`, `destination_balance_error`, and
`destination_amount_deviation` are the dominant signals for fraud
predictions. Transaction type (`type_TRANSFER`/`type_CASH_OUT`) was critical
early in the project (fixing its encoding was a ~15x PR-AUC improvement) but
became functionally redundant once balance-consistency features were added —
removing it now costs <0.001 PR-AUC.

## Why the numbers changed during the project

Three data-integrity issues were found and fixed after the pipeline first
reported PR-AUC = 0.8327. Each is documented here because catching them (and
showing the before/after impact) is more informative than a single clean
number would have been.

### 1. Temporal leakage in step-level features

Three engineered features (`total_transactions`, `total_transaction_amount`,
`avg_transaction_amount`) were computed as full-hour aggregates and merged
onto every transaction in that hour — including transactions at the *start*
of the hour, which in reality could not have known about transactions
happening later in the same hour. This is a form of look-ahead leakage.

- Removing these features dropped PR-AUC from 0.8327 → 0.7567 (with other
  corrections not yet applied).
- A "fixed" trailing (previous-hour) version was also tried, but performed
  *worse* than removing the features outright (0.4491 vs. 0.6219) — likely
  because these features were largely a disguised proxy for "which day is
  this," not a genuine behavioral signal. They were dropped from the final
  feature set entirely (9 features remain, down from 12/14).

### 2. Row-misalignment in destination-history features

Five features (`destination_transactions_last_24h/7d`,
`destination_avg_previous_amount`, `destination_amount_deviation`,
`destination_is_first_transaction`) were computed on a `nameDest`-sorted
copy of the dataframe, then reattached to the original dataframe using a
positional `reset_index(drop=True)` — silently pairing each computed value
with the wrong row, since the two dataframes were in different orders. This
was caught when a "first transaction ever" feature showed rates that were
provably impossible (100% of destinations in the held-out period flagged as
first-ever, contradicted by direct lookup against the full transaction
history).

- Fixed by recomputing with vectorized cumulative operations
  (`cumsum`/`cumcount`) and explicitly re-sorting back to the original row
  order using a preserved index column, with equality checks confirming
  correct alignment before trusting any downstream result.

### 3. Missing NaN fill

After fixing #2, ~55–65% of two rolling-count features were `NaN` (accounts
with no transactions in the lookback window) because a `fillna(0)` step
present in the original pipeline was dropped during the rewrite. LightGBM
silently handles NaN as a valid split value, so this did not crash — it just
quietly changed the training data. Caught via `.describe()` showing a row
count far below the expected total.

**Net effect:** PR-AUC moved 0.8327 (leaky) → 0.4999 (leakage fixed, still
misaligned) → 0.6575 (also still NaN-contaminated) → **0.6684 (fully
corrected)**. The drift analysis showed a similar pattern: apparent PSI
values of 4–7 (a false "severe drift" signal) collapsed to a real, much
smaller and specific drift signal (PSI up to 0.54, isolated to two features)
once both bugs were fixed.

## Project structure

```
├── data/               raw and processed PaySim data (parquet)
├── notebooks/
│   ├── 01_data_audit.ipynb              initial exploration, feature engineering
│   ├── 02_modeling.ipynb                baseline model, Phase B drift analysis
│   ├── 03_delay_aware_validation.ipynb  Phase C + the three-bug discovery process (historical — see note in notebook)
│   └── 04_delay_aware_corrected.ipynb   Phase C, cost-thresholding, SHAP — rerun on fully corrected data (source of truth for final numbers)
├── models/             saved model artifacts (fraud_model.pkl)
├── features/           feature engineering helper code
├── eval/               evaluation utilities (bootstrap CI, drift metrics)
├── api/
│   ├── main.py         FastAPI scoring service
│   └── Dockerfile
├── reports/            written report and figures
└── requirements.txt
```

> **Note on notebook order:** `03_delay_aware_validation.ipynb` documents the
> original Phase C analysis *and* the investigation that uncovered the three
> bugs above — kept as-is because it's the evidence trail for that
> discovery. Its early-cell outputs reflect pre-fix numbers and should not be
> cited directly; `04_delay_aware_corrected.ipynb` and this README are the
> source of truth for final results.

## Running the API

```bash
pip install -r requirements.txt
cd api
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/docs` for an interactive scoring interface.
Example request:

```json
{
  "amount": 1595587.46,
  "origin_balance_error": 0,
  "destination_balance_error": 1595587.46,
  "destination_balance_is_zero": 1,
  "destination_is_first_transaction": 1,
  "type_TRANSFER": true
}
```

**Limitation to note:** several model features
(`destination_transactions_last_24h/7d`, `destination_avg_previous_amount`,
`destination_amount_deviation`) are historical aggregates computed from a
destination account's transaction history. This API accepts them as direct
input; a production deployment would need a feature store or database
lookup to populate them from live transaction history at request time. That
piece is out of scope for this project.

## Methodology notes

- **Dataset:** PaySim synthetic mobile-money transactions. Only CASH_OUT and
  TRANSFER types can be fraudulent, per the simulator's design.
- **Sender-side features were not used** for velocity/history signals:
  99.85% of sender accounts (`nameOrig`) appear only once in the dataset, so
  destination-side (`nameDest`) history was used instead.
- **Dense evaluation window:** days 5–16 were selected as the evaluation
  window after finding severe transaction-density variation across the
  full 31-day period made calendar-week evaluation unreliable.
- **Label delay** (2 days) is simulated by withholding the most recent
  labels from the training set at a fixed offset — a simplification of
  real-world delay, which is typically variable and reporting-dependent.
- **Cost assumptions** (review = 1 unit, false block = 5 units, missed
  fraud = transaction amount) are stated, normalized assumptions for
  illustrating cost-aware thresholding — not derived from real business
  data.

## Known limitations

- PaySim is synthetic; patterns may not transfer to real payment data.
- Label delay and concept drift are both simulated/proxy constructions, not
  observed real-world phenomena.
- Single-model PR-AUC estimates showed substantial seed-to-seed variance
  (std ≈ 0.05–0.08) at this fraud sample size (~800–1900 fraud cases per
  training window); headline numbers are reported alongside this caveat
  rather than as precise point estimates.
- The API does not implement a feature store for historical features (see
  above).
- Drift and delay analyses are limited to the dense 12–16 day window
  identified in the data audit; conclusions may not extend to the full
  31-day period or beyond.
