# Cost-Aware Fraud Detection with Delay-Aware Decisioning

Most fraud detection projects stop at "the model got a good score." I wanted to go one step further and ask the questions a real fraud team would ask before trusting a model with live payments:

- Does it keep working next week, or does it quietly get worse as the data shifts?
- Fraud labels arrive late in real life (chargebacks take days). Does the model survive that?
- A false alarm and a missed fraud don't cost the same. So what should the system actually *do* with a score?
- Can it explain itself, and where exactly does it fail?

This repo is my attempt at answering those, built on the [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) mobile-money dataset (about 6.36 million transactions, 8,213 of them fraud). It includes the notebooks, a FastAPI scoring service, a Streamlit demo and a React dashboard.

It's also an honest record. I found four real bugs in my own pipeline along the way, and my first model turned out to have a blind spot that cost it badly. All of that is written up below rather than tidied away.

## How it works

Instead of a yes/no fraud label, every transaction gets a fraud probability and lands in one of three tiers:

| Tier | When | What happens |
|---|---|---|
| **ALLOW** | probability below 0.32 | Goes through automatically |
| **REVIEW** | 0.32 to 0.85 | Held for a human to look at |
| **BLOCK** | 0.85 and above | Stopped automatically |

The two thresholds weren't picked by feel. They come from a cost sweep with three stated assumptions: a missed fraud costs the transaction amount, wrongly blocking a legitimate payment costs 5 units, and sending anything to review costs 1 unit. Those numbers are illustrative, not real business figures, and I tested how sensitive the thresholds are to them in notebook 05.

For anything that gets flagged, the API also returns the top three reasons in plain English ("this being the first transaction ever to this recipient"), taken from that transaction's own SHAP values.

## Results

Everything below is measured on a held-out period the models never saw: days 14 to 16, about 1.2 million transactions with 822 fraud cases. Training used days 5 to 7, which has only 778 fraud rows. Splits are by time, never random, because a real model only ever predicts the future.

| Model | Held-out PR-AUC | 95% CI | Total cost at its own thresholds |
|---|---|---|---|
| LightGBM (my original model) | 0.7198 | 0.6877 to 0.7492 | 46,455,419 |
| **Random Forest (deployed)** | 0.8335 | 0.8085 to 0.8546 | 47,547 |
| MLP, 3 hidden layers | 0.8485 | 0.8217 to 0.8720 | 48,551 |

Confidence intervals are from 200 bootstrap resamples of the held-out set.

A few things worth pointing out:

- **The deployed model is the Random Forest**, with thresholds 0.32 and 0.85. On the held-out set it let none of the 822 fraud cases through as ALLOW, sent 3.93% of legitimate transactions to review, and wrongly blocked 23 legitimate ones.
- **The cost gap is almost entirely one blind spot.** LightGBM scored 37 real fraud cases at under 0.1% probability. A missed fraud costs its full amount, which is where the 46 million comes from. The Random Forest and the MLP both flag all 37.
- **The MLP surprised me.** I expected a neural network to lose to tree models on tabular data with so few fraud examples. It didn't. Its interval overlaps the Random Forest's, so I can't call a winner between those two, and I kept the Random Forest because it had the lowest cost and is easier to explain.
- **Small fraud is caught but less confidently.** Review-level recall is 100% in every amount quartile, but auto-block recall ranges from 50% for the smallest fraud to 99.6% for the largest. Small fraud tends to go to a human rather than being blocked outright, which I think is the right way round.

One caveat on the cost column: the thresholds were swept on the held-out set, so those cost figures are optimistic.

### Does it hold up over time?

Two features that track how busy a recipient account has been drift quite a lot across the evaluation window (PSI up to 0.54, where anything over 0.25 counts as a large shift). PR-AUC stays flat anyway, and the reason is simple: SHAP shows the model barely uses those features.

That's a slightly lucky result, so I also tested the unlucky case. Shifting the two balance-error features the model leans on hardest drops LightGBM's PR-AUC from 0.72 to about 0.53. So the model is stable against the drift that actually occurred, not against drift in general.

### What about late labels?

I simulated a 2-day label delay by hiding the most recent labels from training. Once I averaged over five training seeds, there was no measurable effect on PR-AUC. A single-seed run looked like it showed a pattern, but that was just training noise, which was a useful lesson in itself.

### Does the approach work on other data?

I reran the same method on the [ULB credit card fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (284,807 transactions, 492 fraud), retraining from scratch since the two datasets share no features.

| Model | Held-out PR-AUC | 95% CI |
|---|---|---|
| LightGBM | 0.6818 | 0.5947 to 0.7662 |
| Logistic regression baseline | 0.7365 | 0.6570 to 0.8148 |
| MLP | 0.7972 | 0.7312 to 0.8613 |

It transferred only partly. The process carried over fine, but LightGBM overfit (training PR-AUC of 1.0) and didn't beat plain logistic regression. With only 115 fraud cases in that held-out period, all three intervals overlap. Full write-up in [reports/06_cross_dataset_summary.md](reports/06_cross_dataset_summary.md).

## Four issues found (and what I did about them)

These changed my numbers more than any modelling choice did.

**1. Leaky hourly features. Disclosed, not fixed.**
Three features (`total_transactions`, `total_transaction_amount`, `avg_transaction_amount`) are totals for the whole hour, attached to every transaction in that hour. A transaction at the start of the hour is therefore seeing a little of the future. I built a leak-free version using only the previous hour, and it performed much worse, most likely because these features were really acting as a time-of-day signal. I kept the original and I'm flagging it here as a known weakness.

**2. Features attached to the wrong rows. Fixed.**
Five recipient-history features were computed on a re-sorted copy of the data and then glued back on by position, so every value ended up next to the wrong transaction. I caught it because a "first ever transaction to this recipient" flag was true for 100% of rows in one period, which is impossible.

**3. A dropped `fillna`. Fixed.**
While fixing #2 I lost a `fillna(0)`, leaving 55 to 65% of two count features as NaN. LightGBM accepts NaN without complaint, so nothing crashed. The training data was just quietly wrong.

**4. The API was sending 14 features to a 17-feature model. Fixed.**
During deployment testing, an obviously fraudulent transaction came back with a 0% fraud probability. The three hourly features from #1 were missing from the request. Again, no error, just a confidently wrong answer.

Here's what the first three did to the baseline PR-AUC:

| Stage | PR-AUC |
|---|---|
| Original (leaky, misaligned, NaN-filled) | 0.8327 |
| Leakage fix attempted | 0.4999 |
| Row alignment fixed | 0.6575 |
| NaN fill restored | 0.6684 |

The best-looking number was the most broken one. Three of the four issues failed silently, and that's the main thing I took away from this project: the bugs that throw errors are the easy ones.

## Running it

You'll need Python 3.12 and, for the dashboard, Node.js 18 or later.

```bash
git clone https://github.com/anushka-saini/Cost-Aware-Fraud-Detection-with-Delay-Aware-Decisioning.git
cd Cost-Aware-Fraud-Detection-with-Delay-Aware-Decisioning

python -m venv fraud_env
fraud_env\Scripts\activate        # macOS/Linux: source fraud_env/bin/activate
pip install -r requirements.txt
```

`requirements.txt` is a full freeze from my Windows machine. On macOS or Linux, remove the `pywin32` and `pywinpty` lines before installing.

The trained model and a 500-row demo sample are in the repo, so the API and both front ends work straight after cloning. You only need the raw data if you want to rerun the notebooks.

### 1. Start the API

```bash
cd api
uvicorn main:app --reload
```

Run it from inside `api/`, since the model file is loaded by relative path. Interactive docs are at http://127.0.0.1:8000/docs.

Try it with:

```bash
curl -X POST http://127.0.0.1:8000/score -H "Content-Type: application/json" -d "{\"amount\": 1595587.46, \"origin_balance_error\": 0, \"destination_balance_error\": 1595587.46, \"destination_balance_is_zero\": 1, \"destination_is_first_transaction\": 1, \"type_TRANSFER\": true}"
```

You get back a probability, a tier, the reasons, and an `assessment_id`.

| Endpoint | What it does |
|---|---|
| `POST /score` | Scores a transaction and returns the tier and reasons |
| `POST /confirm-decision` | Records whether the user proceeded or cancelled after a warning |
| `POST /explain` | SHAP contribution of every feature for one transaction |
| `GET /model-info` | Which model is loaded, its parameters and thresholds |
| `GET /decision-log` | Everything recorded by `/confirm-decision` |
| `GET /validation-metrics` | The notebook figures shown on the dashboard |
| `GET /demo-sample/random` | One random held-out transaction with its true label |
| `GET /health` | Health check |

### 2. Streamlit demo

In a second terminal, from the project root:

```bash
streamlit run app.py
```

There are preset scenarios, plus a **Random Real Transaction** button that pulls a genuine held-out transaction and shows whether the model got it right. If the model flags something, you can choose to proceed or cancel, and that choice is logged.

### 3. React dashboard

In another terminal:

```bash
cd dashboard
npm install
npm run dev
```

Open http://localhost:5173. It has three pages:

- **Live Scoring**: score a transaction and see each feature's SHAP contribution
- **Model Validation**: charts of the results from the notebooks
- **Decision Log**: how often users override warnings, broken down by tier

The dashboard talks to the API through a dev proxy, so there's no CORS setup. If your API is somewhere other than `127.0.0.1:8000`, set `FRAUD_API_URL` before `npm run dev`.

### Rerunning the notebooks

The datasets are too large for the repo. Download them from Kaggle and place them here:

- PaySim → `data/raw/transaction_data.csv`
- ULB credit card fraud → `data/raw/creditcard.csv` (only for notebook 06)

Then run the notebooks in order.

## What's in the repo

```
├── api/
│   ├── main.py                  Scoring API: /score, /confirm-decision
│   ├── dashboard_routes.py      Read-only endpoints for the dashboard
│   ├── validation_metrics.json  Figures shown on the dashboard, each with its source cell
│   ├── rf_model.pkl             Deployed Random Forest
│   ├── fraud_model.pkl          Original LightGBM model
│   └── Dockerfile
├── dashboard/                   React + Vite + Recharts dashboard
├── notebooks/                   The analysis, in order (see below)
├── reports/                     Summaries and raw results from notebooks 06 and 07
├── app.py                       Streamlit demo
├── demo_sample.csv              500 held-out transactions with true labels
└── requirements.txt
```

### The notebooks

| Notebook | What it covers |
|---|---|
| `01_data_audit` | Exploring PaySim, feature engineering, a logistic regression baseline |
| `02_modeling` | LightGBM, performance over time, drift, SHAP, decision tiers |
| `03_delay_aware_validation` | First label-delay analysis, and where bugs #2 and #3 were found |
| `04_delay_aware_corrected` | The delay analysis and cost thresholds redone on corrected data |
| `05_model_enhancement` | LightGBM vs Random Forest, cost sweeps, blind-spot and drift stress tests |
| `06_cross_dataset_validation` | The same method on the ULB credit card data |
| `07_mlp_comparison` | A PyTorch MLP against both tree models |

Notebook 03 is kept as it was because it's the record of how the bugs were found. Its early outputs are pre-fix numbers, so don't quote them. Use 04 onwards. `02_modeling_old` is the earlier version of 02, kept for reference.

## Limitations

- **PaySim is synthetic.** Its fraud follows simulator rules and is probably easier to learn than the real thing. The weaker results on the ULB data support that.
- **Thresholds were tuned on the held-out set**, so the cost figures are best-case.
- **The hourly features leak** (issue #1 above).
- **No feature store.** Features like "transactions to this recipient in the last 24 hours" are passed in by the caller. A real deployment would need to look them up from live history.
- **Review thresholds are fragile.** For the Random Forest and the MLP, a small shift in the review threshold raises cost noticeably.
- **Label delay and drift are simulated**, and only studied within days 5 to 16, the one stretch of the data with steady enough volume to trust.
- **The Dockerfile hasn't been tested end to end.** It expects to be built from the project root (`docker build -f api/Dockerfile .`) and will need the Windows-only packages removed from `requirements.txt` first.
- **Pending assessments live in memory** and the decision log is a CSV. Fine for a demo, not for production.
- The LightGBM vs Random Forest comparison is still something I'm working on, so some figures may change.

## Built with

Python, pandas, scikit-learn, LightGBM, PyTorch, SHAP, Optuna, FastAPI, Streamlit, React, Vite and Recharts.

## Data credits

- PaySim: Lopez-Rojas, Elmir and Axelsson, *PaySim: A financial mobile money simulator for fraud detection* (2016).
- Credit card fraud dataset: Machine Learning Group, Université Libre de Bruxelles.
