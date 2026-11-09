# Six-minute demo

A walkthrough for the presentation. Timings are targets; the whole thing fits in six minutes with a little room. If you only have five, leave out the retraining step at 5:00.

## Before you present

Do this at least ten minutes before your slot, on the machine you will present from.

1. Plug the laptop in. On battery the scoring is noticeably slower.
2. Terminal 1, from the project root: `uvicorn fraud.service.app:app --port 8000`
3. Terminal 2: `cd dashboard` then `npm run dev`
4. Open http://localhost:5173 and sign in as **admin** / `admin123`.
5. Go to **Live feed**, set the speed to **20 per second**, press **Restart**, let it run for about 30 seconds, then **Stop**. This fills the review queue and the audit log so no page is empty.
6. Open the **Threshold simulator**, the **Explainability** page and the **Model lab** once each so their data is cached.
7. Open a third terminal in the project root, run `python scripts/retrain.py --as-of-day 16 --maturity-days 2` once (about four and a half minutes on the laptop, plugged in), and leave the terminal open. The Model lab then already has a gate decision to show if the live run is slow.
8. Turn on **presentation mode** (the screen icon in the top bar). It makes everything larger for the back row.
9. Leave the browser on the **Overview** page, in light mode.

If you present from the online version instead, open it a few minutes early: the free API goes to sleep and takes about a minute to wake up.

To start completely fresh, stop the API, delete `api/fraud.db`, and start it again.

## The walkthrough

### 0:00 Opening (30 seconds), Overview page

> "Most fraud projects stop at a model score. This one asks what a fraud team would ask: can I trust the number, what should the system do with a score, and what happens when things change. This is the operations view: what has been scored, what is waiting for a person, and whether the model's inputs have drifted."

Point at the **cost avoided vs best rule** tile.

> "The fair comparison is not 'no fraud check at all'. It is the best set of simple rules I could write from the earlier days. On the held-out days those rules cost 196,675; the model's decisions cost 32,586. Sending everything to a person would have cost over 1.2 million."

### 0:30 Live feed (40 seconds)

Click **Live feed**, speed **5 per second**, press **Continue**.

> "These are real held-out transactions, days the model never trained on, replayed in order. Each one goes through the same path a payment would: the service looks up the recipient's history, computes the inputs itself, scores it with an average of five models, and puts it in one of three tiers. Green goes through, red is stopped, amber goes to a person."

Point at the **Truth** column.

> "Because it is a replay, we can see the truth. As the banner says, fraud is oversampled so you see some in a short demo; the real rate is about seven in ten thousand, so the rates on this page are not real-world rates."

Press **Pause**.

### 1:10 A case in the queue (50 seconds)

Click an amber **REVIEW** row. It opens the case.

> "The model was not sure enough to decide on its own, so an analyst gets it. Three reasons in plain English, and underneath, the full explanation: each bar is one input pushing this transaction toward fraud or away from it."

Type a note and click **Decline**.

> "The analyst has to say why. That decision is logged, and it becomes a training label once it has had time to settle."

Decline or approve one or two more cases from the **Review queue** (about 10 seconds), then switch to the third terminal and start the retraining job:

```bash
python scripts/retrain.py --as-of-day 16 --maturity-days 2
```

> "This is the retraining job. It retrains the model on newer days, using only labels that are at least two days old, and then a gate decides whether the new model may replace the one serving. It takes about four minutes, so I'll come back to it at the end."

Do not add `--promote`: without it the job only reports its decision and never changes the serving model.

### 2:00 Explainability: where it goes wrong (40 seconds)

Click **Explainability**, then scenario **5, A legitimate transaction that was blocked**.

> "I picked these five transactions by fixed rules, and two of them are mistakes on purpose. This is a real customer the model blocked. The biggest push came from a mismatch in the sender's balance, which is the signal it relies on most. That honesty matters more than a perfect-looking demo."

### 2:40 Threshold simulator (60 seconds). The centrepiece.

Click **Threshold simulator**.

> "A missed fraud costs its full amount. A wrongful block costs 5. A review costs 1. So where should the two thresholds go? This recomputes the whole cost on 1.2 million transactions, instantly."

Drag the **Review threshold** slider slowly to the right.

> "The cost falls as we review fewer transactions, and then it hits a cliff: one missed fraud costs more than thousands of reviews. So the deployed threshold sits deliberately to the left of the cheapest point. It stays safe even if the probabilities are off by a factor of two."

Press **Reset**.

### 3:40 Model lab: how V3 earned its place (60 seconds)

Click **Model lab**. Point at the version history.

> "Three versions. My first version reported great numbers, but I had tuned it on the test days, so they were not honest. Version 2 fixed the method. For Version 3 I compared linear models, tree models and four kinds of neural network on earlier days only."

Scroll to the held-out test.

> "Then I wrote down the rule before looking: V3 replaces V2 only if it is significantly better and not more expensive. It ran once. V3 won: PR-AUC up by about 0.01, cost down by 1,158, still no fraud let through."

Scroll to the leaderboard.

> "CatBoost scored a touch higher, but the test says it is not really different, so the simpler model stayed."

### 4:40 The retraining gate (40 seconds)

Check the third terminal: the job prints its decision as JSON at the end. Then refresh the **Model lab** and scroll to **Retraining gate: the latest run**.

> "The job finished. It retrained V3's settings on days 5 to 10 and judged both models on days 11 to 13. The challenger scored 0.8827 against 0.8820 for the model that is serving, but the 95% interval of the difference runs from below zero to above it, so it is not significantly better, and the gate keeps V3."

Point at the labels line.

> "The decisions I made a minute ago are in the database, but those transactions are from days 14 to 16, so they are not two days old yet as of day 16. The job counts them and does not use them. That is the label delay rule at work."

If the job has not finished yet, the card still shows the warm-up run from before the talk, with the same result.

### 5:20 Close (20 seconds)

> "One limitation to be clear about: this is simulated data, and the model leans on a pattern the simulator creates. What transfers is the method: honest evaluation, one scoring path for training and serving, cost-based decisions a person can inspect, and monitoring for when it stops being true."

## Numbers to have ready

All from `results/metrics.json`.

| Question | Answer |
|---|---|
| Deployed model | V3: average of 5 tuned LightGBM models, Platt-calibrated, 16 inputs |
| Held-out PR-AUC | 0.8730, 95% CI 0.8474 to 0.8927 |
| V3 minus V2 on held-out | PR-AUC +0.0098 (CI +0.0057 to +0.0149); cost −1,158 (CI −1,286.1 to −1,026.9) |
| Frauds let through | 0 of 822 |
| Legitimate traffic reviewed | 2.69% |
| Wrongful blocks | 3 of 1,201,820 |
| Block precision / recall | 99.5% / 77.4% |
| Total cost | 32,586 (blocking nothing: 1,362,518,395.67) |
| Thresholds | review from 0.000316, block from 0.71 |
| Safety margin | the probabilities can be off by ×2.17 before the next fraud slips through |
| Without balance features (V2 study) | PR-AUC 0.5028 |
| Autoencoder as a fraud detector | PR-AUC 0.0541 on days 11–13; now used only for monitoring |
| Stress test | adding 500 or more to the balance inputs lets 113 of 822 frauds through; monitoring alerts |
| Best hand-written rule on held-out | rule engine, cost 196,675, 1 fraud let through, 16.35% of traffic reviewed |
| Retraining gate (as of day 16) | challenger 0.8827 vs V3 0.8820 PR-AUC on days 11–13, difference +0.0006 (CI −0.0023 to +0.0039): keep V3 |
| Speed | 15.0 to 24.5 ms per request at the median (V3, laptop, SQLite or PostgreSQL) |
| Tests | 98, including a train/serve parity test |

## Questions you should expect

**"Why is the review threshold so small?"** It is a calibrated probability, and fraud is about 0.07% of traffic. A missed fraud costs hundreds of thousands; a review costs 1. So it is worth reviewing anything with even a small chance of fraud.

**"Why not CatBoost, if it scored higher?"** Its PR-AUC was 0.8822 against 0.8820, and the paired test said the difference is not real. With no real difference, I kept the model family that was already deployed and simpler to run.

**"Is the deep learning doing anything?"** Not much on this data. The best neural network (an MLP) was significantly worse than the tree models, and the other neural networks were worse still. The autoencoder is now a monitor: it tracks how many transactions look unusual, and that share stayed in its expected band.

**"Did you look at the test set?"** The held-out days were looked at in earlier versions, so they are not brand new. For V3, every choice was made on days 5–13, the decision rule was written into the script first, and the script that scores the held-out days refuses to run twice.

**"What does delay-aware mean here?"** The retraining job only uses labels that have matured, and a new model only replaces the old one if it is significantly better on recent labelled data. On this dataset, delays of up to three days made almost no difference, so I am not claiming a benefit, only that the mechanism is built and tested.

**"Would this work on real data?"** Not with these numbers. PaySim is synthetic, and the model depends on a regularity the simulator produces. The method is what transfers.

## If the live demo fails

Work down this list. Do not debug in front of the audience.

1. **The dashboard shows "Could not reach the API".** The API terminal has stopped. Restart it with `uvicorn fraud.service.app:app --port 8000` and refresh. It takes about ten seconds.
2. **The live feed will not connect, but other pages work.** Skip the feed. The review queue already holds cases from your warm-up run; go straight to a case.
3. **The dashboard will not load at all.** Run `streamlit run app.py` and use the scenario buttons: the same five transactions as the Explainability page. Then switch to the screenshots.
4. **Nothing runs.** Present from `docs/screenshots/`, in this order: `02-overview`, `03-live-feed`, `05-case-detail`, `13-explainability`, `06-threshold-simulator`, `08-model-lab`. The same script works over the screenshots.

Keep `docs/screenshots/` open in a second window before you start, so the fallback is one Alt-Tab away.
