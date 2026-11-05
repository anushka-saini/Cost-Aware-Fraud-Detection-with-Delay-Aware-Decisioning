# Five-minute demo

A walkthrough for the presentation. Timings are targets; the whole thing fits in five minutes with a little room.

## Before you present

Do this at least ten minutes before your slot, on the machine you will present from.

1. Plug the laptop in. On battery the scoring is noticeably slower.
2. Terminal 1, from the project root: `uvicorn fraud.service.app:app --port 8000`
3. Terminal 2: `cd dashboard` then `npm run dev`
4. Open http://localhost:5173 and sign in as **admin** / `admin123`.
5. Go to **Live feed**, set the speed to **20 per second**, press **Restart**, let it run for about 30 seconds, then **Stop**. This fills the review queue and the audit log so no page is empty.
6. Open the **Threshold simulator** once so its data is cached.
7. Leave the browser on the **Overview** page, in light mode, zoomed so the back row can read it.

To start completely fresh, stop the API, delete `api/fraud.db`, and start it again.

## The walkthrough

### 0:00 Opening (30 seconds), Overview page

> "Most fraud projects stop at a model score. This one asks what a fraud team would ask: can I trust the number, what should the system do with a score, and what happens when things change. This is the operations view: what has been scored, what is waiting for a person, which model is serving, and whether its inputs have drifted."

Point at the six tiles. Do not explain each one.

### 0:30 Live feed (45 seconds)

Click **Live feed**, speed **5 per second**, press **Continue**.

> "These are real held-out transactions, days the model never saw, replayed in order. Each one goes through the same path a payment would: the service looks up the recipient's history, computes the features itself, scores it with an ensemble of five models, and puts it in one of three tiers. Green goes through, red is stopped, amber goes to a person."

Point at the **Truth** column.

> "Because it is a replay, we can see the truth. I have oversampled fraud here so you see some in a short demo; the real rate is seven in ten thousand."

Press **Pause**.

### 1:15 A case in the queue (75 seconds)

Click an amber **REVIEW** row. It opens the case.

> "The model was not sure enough to decide on its own, so an analyst gets it. Two scores: the supervised fraud probability, and an anomaly score from an autoencoder that was trained with no labels at all."

Point at **Why it was flagged**.

> "Three reasons in plain English, and underneath, the full explanation. Each bar is one input pushing the score toward fraud or away from it. These are SHAP values from this transaction, not a general feature ranking."

Point at the recipient history and similar cases, then type a note and click **Decline**.

> "The analyst has to say why. That decision is logged, and it becomes a training label for the next retraining, once it has had time to settle."

### 2:30 Threshold simulator (90 seconds). The centrepiece.

Click **Threshold simulator**.

> "A missed fraud costs its full amount. A wrongful block costs a little. A review costs a little less. So where should the two thresholds go? This recomputes the whole cost on 1.2 million transactions in the browser, instantly."

Drag the **Review threshold** slider slowly to the right, past the minimum.

> "Watch the cost. It falls as we review fewer transactions, and then it hits a cliff: one missed fraud costs more than thousands of reviews. The cheapest point is right at the edge of that cliff."

Press **Reset**.

> "So the deployed threshold is deliberately to the left of the cheapest point. It was chosen to stay safe even if the model's probabilities are off by a factor of two."

Click **Held-out (days 14–16)**.

> "And this matters: the thresholds were chosen on one period and tested on a later one. In my first version I tuned them on the test set, and the numbers looked better than they were."

Optional, if there is time: drag a cost slider and show the optimum move.

### 4:00 Honest numbers and monitoring (60 seconds)

Click **Model comparison**. Point at the interval chart.

> "Four models, each with a confidence interval, each trained five times. The orange marks are what I reported at first. My first version said the Random Forest was best and that it missed no fraud. When I stopped tuning on the test set, LightGBM turned out to be significantly better, and the Random Forest lets two frauds through. The deployed model lets none through and reviews under three percent of traffic."

Scroll to **Dependence on PaySim's balance bookkeeping**.

> "And a limitation I found myself: remove three features and performance collapses. Much of this signal is a property of the simulator, so these are upper bounds, not a promise about real payments."

Click **Monitoring**. Point at the stress-test table.

> "Last, what happens when the data moves. If the features the model depends on shift, it lets a fifth of fraud through. The monitoring catches that: the drift alert fires and the review rate drops out of its expected band."

### 5:00 Close

> "So: honest evaluation, one scoring path shared by training and serving, cost-based decisions a person can inspect and override, and monitoring for when it stops being true."

## Numbers to have ready

| Question | Answer |
|---|---|
| Deployed model | Ensemble of 5 LightGBM models, Platt-calibrated, 16 leak-free features |
| Held-out PR-AUC | 0.8632, 95% CI 0.8384 to 0.8835 |
| Frauds let through | 0 of 822 |
| Legitimate traffic reviewed | 2.79% |
| Wrongful blocks | 3 of 1,201,820 |
| Block precision / recall | 99.5% / 77.0% |
| Total cost | 33,744 |
| LightGBM against Random Forest | +0.0367 PR-AUC, CI +0.0258 to +0.0466 |
| Without balance features | PR-AUC 0.5028 |
| Autoencoder alone | PR-AUC 0.2091 for the fixed seed; 0.1273 ± 0.0730 over 5 seeds |
| Scoring latency | 81.7 ms median, 174.9 ms at the 95th percentile |
| Tests | 64, including a train/serve parity test on 1,000 held-out rows |

## Questions you should expect

**"Why is the review threshold so small?"** It is a calibrated probability, and fraud is 0.07% of traffic. A missed fraud costs hundreds of thousands; a review costs 1. So it is worth reviewing anything with even a small chance of fraud.

**"Is the deep learning doing anything?"** Honestly, less than hoped. The MLP is a strong second-place model. The autoencoder finds most fraud with no labels, but it is unstable from seed to seed, and adding its score to LightGBM did not improve it. It is in the system as a label-free second opinion and as a fallback when the supervised model's labels are stale.

**"What does delay-aware mean here?"** The retraining job only uses labels that have matured, a new model only replaces the old one if it is significantly better on recent labelled data, and there is a fallback rule for a stale model. On this dataset, delays of up to three days made almost no difference, so I am not claiming a benefit, only that the mechanism is built and tested.

**"Would this work on real data?"** Not with these numbers. PaySim is synthetic, and the ablation shows the model depends on a regularity the simulator produces. The method (the evaluation protocol, the cost-based thresholds, the monitoring) is what transfers.

**"What was wrong with the first version?"** Class weights and thresholds were chosen on the test set, three features leaked future information, and the API did not compute features the same way the training code did. All three are fixed, and a test now guards the last one.

## If the live demo fails

Work down this list. Do not debug in front of the audience.

1. **The dashboard shows "Could not reach the API".** The API terminal has stopped. Restart it with `uvicorn fraud.service.app:app --port 8000` and refresh. It takes about ten seconds.
2. **The live feed will not connect, but other pages work.** Skip the feed. The review queue already holds cases from your warm-up run; go straight to a case.
3. **The dashboard will not load at all.** Run `streamlit run app.py` and use the single scoring form: pick a preset fraud, show the tier and the reasons. Then switch to the screenshots.
4. **Nothing runs.** Present from `docs/screenshots/`, in this order: `02-overview`, `03-live-feed`, `05-case-detail`, `06-threshold-simulator`, `08-model-comparison`, `07-monitoring`. The same script works over the screenshots; for the simulator, describe the slider movement and point at the cliff in the chart.

Keep `docs/screenshots/` open in a second window before you start, so the fallback is one Alt-Tab away.
