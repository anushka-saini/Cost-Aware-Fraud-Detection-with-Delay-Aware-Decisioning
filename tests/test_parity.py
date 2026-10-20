"""Train/serve parity.

Scores held-out transactions twice: offline, from the batch feature table
(what the evaluation scripts do), and through the live API, which has to
rebuild every feature from raw transaction fields and its own record of the
recipient's history. The two probabilities must agree to 1e-6.

This is the test that would have caught the original deployment bug, where
the API sent 14 features to a 17-feature model and returned a confident 0%.
"""

import numpy as np
import pandas as pd
import pytest
from conftest import needs_data, needs_model

from fraud.data import SPLITS, load_feature_frame
from fraud.features import TRANSACTION_TYPES, TYPE_COLUMNS
from fraud.service.core import get_scorer

N_ROWS = 1000
N_FRAUD = 200
TOLERANCE = 1e-6
RAW_COLUMNS = ["step", "amount", "nameDest", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]

pytestmark = [pytest.mark.data, needs_data, needs_model]


@pytest.fixture(scope="module")
def replay():
    scorer = get_scorer()
    columns = list(dict.fromkeys([*scorer.base_columns, *RAW_COLUMNS, *TYPE_COLUMNS, "day_bucket", "isFraud"]))
    frame = load_feature_frame(columns)
    held = frame[frame["day_bucket"].isin(SPLITS["held_out"])]
    sample = pd.concat([
        held[held["isFraud"] == 1].sample(N_FRAUD, random_state=0),
        held[held["isFraud"] == 0].sample(N_ROWS - N_FRAUD, random_state=0),
    ]).sort_index()
    # Every earlier transaction to the sampled recipients, in chronological row order.
    related = frame.loc[frame["nameDest"].isin(set(sample["nameDest"])) & (frame.index <= sample.index.max()),
                        ["nameDest", "step", "amount"]]
    offline = scorer.score_frame(sample[scorer.base_columns])
    return sample, related, offline


def _raw_payload(row) -> dict:
    return {
        "step": int(row["step"]),
        "type": next(t for t in TRANSACTION_TYPES if row[f"type_{t}"]),
        "amount": float(row["amount"]), "nameDest": row["nameDest"],
        "oldbalanceOrg": float(row["oldbalanceOrg"]), "newbalanceOrig": float(row["newbalanceOrig"]),
        "oldbalanceDest": float(row["oldbalanceDest"]), "newbalanceDest": float(row["newbalanceDest"]),
    }


def test_api_matches_offline_scores_on_held_out_rows(client, replay):
    from fraud.service import db as database

    sample, related, offline = replay
    sampled = set(sample.index)
    online, pending = {}, []

    def flush():
        # Transactions the service saw earlier, written straight to its feature store.
        if pending:
            with database.SessionLocal() as session:
                session.add_all(database.Transaction(name_dest=d, step=int(s), amount=float(a), source="history")
                                for d, s, a in pending)
                session.commit()
            pending.clear()

    for index, dest, step, amount in related.itertuples():
        if index in sampled:
            flush()
            response = client.post("/score/raw", json=_raw_payload(sample.loc[index]))
            assert response.status_code == 200, response.text
            online[index] = response.json()
        else:
            pending.append((dest, step, amount))

    assert len(online) == N_ROWS
    probability = np.array([online[i]["fraud_probability"] for i in sample.index])
    raw = np.array([online[i]["raw_score"] for i in sample.index])
    tiers = [online[i]["tier"] for i in sample.index]

    assert np.max(np.abs(probability - offline["fraud_probability"].to_numpy())) < TOLERANCE
    assert np.max(np.abs(raw - offline["raw_score"].to_numpy())) < TOLERANCE
    assert tiers == offline["tier"].tolist()
    # The sample must actually exercise the model: not everything in one tier.
    assert len(set(tiers)) >= 2


def test_precomputed_feature_endpoint_matches_offline_scores(client, replay):
    sample, _, offline = replay
    subset = sample.iloc[:: max(1, len(sample) // 100)]
    feature_fields = [c for c in subset.columns if c not in RAW_COLUMNS[2:] + ["day_bucket", "isFraud", "hour_sin", "hour_cos"]]
    for index, row in subset.iterrows():
        payload = {k: (v.item() if hasattr(v, "item") else v) for k, v in row[feature_fields].items()}
        response = client.post("/score", json=payload)
        assert response.status_code == 200, response.text
        assert abs(response.json()["fraud_probability"] - offline.loc[index, "fraud_probability"]) < TOLERANCE
