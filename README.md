# Cost-Aware Fraud Detection with Delay-Aware Decisioning

A fraud detection system built on the PaySim synthetic mobile-money dataset (~6.36M transactions), designed around four questions a real fraud system has to answer beyond "can it classify fraud":

- **Does the model stay reliable over time, or silently decay?** (drift)
- **Does it still work when fraud labels arrive late, as they do in the real world?** (label delay)
- **What decision should the system actually take**, given that false positives and false negatives have very different costs? (cost-aware thresholding)
- **Can the model's decisions be explained, and where does it actually fail?** (SHAP + targeted stress-testing)

This repo includes a working FastAPI scoring service and a Streamlit demo app with a live "random real transaction" mode that scores genuine held-out transactions against their true labels — not just a static form.

**What makes this project different from a typical classification exercise:** four separate data-integrity and deployment issues were found, diagnosed, and either fixed or explicitly documented as accepted tradeoffs — not glossed over. Each one measurably changed the reported results, and each is documented below with before/after evidence.

---

## Results at a glance

| Metric | Value |
|---|---|
| Baseline PR-AUC (LightGBM, single seed) | 0.6684 |
| Baseline PR-AUC (5-seed mean, seed variance ±~0.06) | 0.6563 |
| Delay-aware model PR-AUC (5-seed rank-ensemble) | 0.7671 |
| Cost-optimal review threshold | t_review ≈ 0.0001 |
| Block threshold sensitivity | 0.7–0.95 evaluated; no unique optimum identified under the stated cost assumptions |
| Recall at review threshold, held-out set | 92–97% across every transaction-amount quartile |
| False-block rate, 5,000-transaction random sample | 0 legitimate transactions wrongly blocked (out of ~4,993 legitimate transactions in that sample) |
| False-review rate, same sample | ~4% of legitimate transactions flagged for human review, never auto-blocked |

**Drift:** two destination-transaction-velocity features show real, escalating distributional drift across the dense evaluation window (PSI up to 0.54). Overall model PR-AUC remains statistically stable across all evaluation periods (overlapping 95% CIs) — SHAP confirms the model places low importance on the drifting features, which is why the drift doesn't propagate to a performance drop.

**Label delay:** a 2-day label delay (the window ultimately supported by available data density, after starting from a longer initial target) showed no statistically distinguishable effect on PR-AUC once training-seed variance was accounted for. A single-seed sweep looked non-monotonic; averaging across 5 seeds showed this was training variance, not a real delay effect.

**Explainability:** `origin_balance_error`, `destination_balance_error`, and `destination_amount_deviation` are the dominant signals. Transaction type was critical early in the project (fixing its encoding was a ~15x PR-AUC improvement) but became functionally redundant once balance-consistency features were added.

---

## Four issues found, diagnosed, and resolved or disclosed

### 1. Temporal leakage in step-level features — identified, alternative tested, tradeoff accepted and disclosed
Three engineered features (`total_transactions`, `total_transaction_amount`, `avg_transaction_amount`) are computed as full-hour aggregates and merged onto every transaction in that hour — including transactions at the start of the hour, which couldn't have known about transactions later in the same hour. This is a real, acknowledged leakage source.

A leakage-safe trailing (previous-hour-only) version was built and tested as a fix. **It performed substantially worse** than the leaky version, likely because these features were functioning largely as a disguised proxy for time-of-day/hour patterns rather than a genuine behavioral signal — removing the "future" information also removed most of the feature's usable signal. Given the corrected alternative was strictly worse, the original full-hour formulation was retained in the final model, and this residual leakage risk is disclosed here as a known, accepted limitation rather than a silently ignored one. See "Known limitations" below.

### 2. Row-misalignment in destination-history features — fixed
Five destination-history features were computed on a re-sorted copy of the dataframe, then reattached using a positional `reset_index(drop=True)` — silently pairing each value with the wrong row. Caught when a "first transaction ever" feature showed a rate that was provably impossible (100% of destinations flagged as first-ever in the held-out period). Fixed with vectorized cumulative operations and explicit index preservation, verified with equality checks before trusting downstream results.

### 3. Missing NaN fill — fixed
After fixing #2, 55–65% of two rolling-count features were silently `NaN` because a `fillna(0)` step was dropped during the rewrite. LightGBM treats NaN as a valid split value, so this didn't crash — it just quietly corrupted the training data. Caught via `.describe()` showing a row count far below expected.

**Combined effect on PR-AUC across these three issues:** 0.8327 (leaky, misaligned, NaN-contaminated) → 0.4999 (leakage-related fix attempted, still misaligned) → 0.6575 (still NaN-contaminated) → **0.6684 (fully corrected, with #1's tradeoff knowingly accepted)**.

### 4. Feature-count mismatch between the trained model and the deployed API — fixed
During deployment testing, an obviously fraudulent test transaction (empty destination account, uncredited balance, first-time recipient) scored a 0% fraud probability. Root cause: the trained model was fit on **17 features**, but the API's scoring code only supplied **14** — the three system-level features from #1 above were missing entirely from the request payload. The mismatch produced a silently wrong prediction rather than a crash — the second time in this project a *silent* failure mode was a bigger risk than a loud one. Fixed by aligning the API's feature list exactly with the trained model's `feature_name_`, and validated with real held-out transactions before trusting the fix.

---

## Stress-testing the deployed model (beyond standard metrics)

Standard PR-AUC and recall numbers can hide amount-dependent or pattern-dependent weaknesses. Three additional checks were run against the live, deployed model:

**Amount-stratified recall.** Recall at the review threshold holds at 92–97% across every amount quartile of the held-out set — the system misses almost no real fraud regardless of transaction size. What varies is *confidence*: recall at the stricter auto-block threshold ranges from 63–65% for lower-amount fraud up to 96% for the highest-amount fraud, meaning lower-value fraud is more often routed to human review than auto-blocked. This is a defensible, arguably intentional characteristic for a cost-aware system, where the cost of a false block scales with transaction size.

**Random-sample validation against ground truth.** A live check against 5,000 randomly drawn held-out transactions (not cherry-picked) found 7 out of 7 real fraud cases caught, and 200 out of ~4,993 legitimate transactions flagged for review — with zero legitimate transactions wrongly auto-blocked in this sample. This is evidence from one 5,000-transaction sample, not a claim of a general 0% false-positive rate — the system's worst observed mistake on clean traffic in this sample was a review delay, never a wrongful rejection.

**A specific, named blind spot.** Deeper analysis found a consistent 4% miss rate (4 out of 100 fraud cases in a held-out sample) concentrated in one specific pattern: fraud sent to an **established** destination (not first-time) with **fully correct balance bookkeeping** and only moderate amount deviation. The model's top features — balance-error signals and first-transaction status — are largely absent in these cases, leaving it with little to act on. This is a genuine, mechanistic limitation, not noise: it points toward a concrete direction for future work (velocity- or behavior-based features, or an anomaly-detection layer, to catch fraud that doesn't trip the balance-corruption signal).

---

## Repository structure

```
├── api/
│   ├── main.py              FastAPI scoring service (17-feature aligned)
│   ├── Dockerfile           containerized deployment config (not yet validated end-to-end)
│   └── fraud_model.pkl      deployed model artifact
├── models/
│   └── fraud_model.pkl      canonical trained model
├── notebooks/
│   ├── 01_data_audit.ipynb              initial exploration, feature engineering
│   ├── 02_modeling.ipynb                baseline model, drift analysis, held-out validation
│   ├── 03_delay_aware_validation.ipynb  original delay analysis + the bug-discovery process (historical record — see note below)
│   └── 04_delay_aware_corrected.ipynb   delay analysis, cost-thresholding, SHAP — rerun on fully corrected data
├── data/
│   ├── raw/                 original PaySim data
│   └── processed/           engineered feature parquet
├── app.py                   Streamlit demo — live scoring UI with preset and random-transaction modes
├── demo_sample.csv          500 real held-out transactions (with ground truth) powering the demo's random-transaction feature
└── requirements.txt
```

**Note on notebook order:** `03_delay_aware_validation.ipynb` documents the original analysis and the investigation that uncovered issues #2 and #3 — kept as-is because it's the evidence trail for that discovery. Its early-cell outputs reflect pre-fix numbers and should not be cited directly. `04_delay_aware_corrected.ipynb` and this README are the source of truth for final results.

---

## Running it

### API
```bash
pip install -r requirements.txt
cd api
uvicorn main:app --reload
```
Interactive docs at `http://127.0.0.1:8000/docs`. Example request:
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

### Streamlit demo
```bash
streamlit run app.py
```
Requires the API running separately (above) at `localhost:8000`. Includes four verified preset scenarios (one legitimate, three fraud across amount tiers) plus a **Random Real Transaction** button that pulls a genuine held-out example each click and shows whether the model's prediction matched the true label — a live validation moment rather than a scripted demo. Note: given the ~4% blind spot documented above, an occasional miss during a live random draw is expected and explainable, not a bug.

**Limitation worth noting:** several features (destination transaction counts, average previous amount, amount deviation) are historical aggregates computed from a destination account's transaction history. The API accepts them as direct input; a production deployment would need a feature store or database lookup to populate them from live transaction history at request time. That piece is out of scope for this project.

---

## Methodology notes

- **Dataset:** PaySim synthetic mobile-money transactions. Only `CASH_OUT` and `TRANSFER` types can be fraudulent, per the simulator's design. Amount values are PaySim's simulated transaction volume, not real-world currency figures.
- Sender-side features were not used for velocity/history signals — 99.85% of sender accounts appear only once in the dataset, so destination-side history was used instead.
- **Evaluation window:** a contiguous, dense subset of the dataset was selected after finding severe transaction-density variation across the full period made calendar-week evaluation unreliable (a sparse-period stress test produced a misleadingly high PR-AUC driven entirely by base-rate differences, and was excluded from primary conclusions accordingly).
- **Label delay** is simulated by withholding the most recent labels from training at a fixed offset — a simplification of real-world delay, which is typically variable and reporting-dependent. The final delay window (2 days) was set empirically based on data density constraints after testing longer windows proved infeasible.
- **Cost assumptions** (review = 1 unit, false block = 5 units, missed fraud = transaction amount) are stated, normalized assumptions for illustrating cost-aware thresholding — not derived from real business data.

## Known limitations

- PaySim is synthetic; patterns may not transfer to real payment data.
- Label delay and concept drift are both simulated/proxy constructions, not observed real-world phenomena.
- Single-model PR-AUC estimates show substantial seed-to-seed variance (std ≈ 0.05–0.08) at this fraud sample size; headline numbers are reported alongside this caveat rather than as precise point estimates.
- **The three system-level features (`total_transactions`, `total_transaction_amount`, `avg_transaction_amount`) retain a disclosed temporal-leakage risk** — a leakage-safe alternative was tested and found to destroy the features' signal entirely, so the original formulation was knowingly retained. This is a deliberate, documented tradeoff, not an oversight.
- The API does not implement a feature store for historical features (see above).
- The Dockerfile in `api/` has not been validated end-to-end in this environment.
- Drift and delay analyses are limited to the dense evaluation window identified in the data audit; conclusions may not extend beyond it.
- The model has a specific, characterized blind spot (~4% of fraud cases) for fraud sent to established destinations with correct balance bookkeeping — see the stress-testing section above.
