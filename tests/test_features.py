import numpy as np
import pandas as pd
import pytest

from fraud.features import (
    FEATURES_V1,
    FEATURES_V2,
    build_feature_frame,
    compute_online_features,
    destination_history_features,
    to_model_frame,
)


def make_raw(rows):
    """rows: (step, type, amount, orig, old_o, new_o, dest, old_d, new_d, fraud)"""
    columns = ["step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
               "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud"]
    return pd.DataFrame(rows, columns=columns)


@pytest.fixture
def raw():
    return make_raw([
        (1, "PAYMENT", 100.0, "A", 500.0, 400.0, "X", 0.0, 0.0, 0),
        (1, "TRANSFER", 50.0, "B", 50.0, 0.0, "Y", 0.0, 50.0, 1),
        (1, "CASH_OUT", 30.0, "C", 30.0, 0.0, "X", 10.0, 40.0, 0),    # same hour, same recipient as row 0
        (5, "CASH_OUT", 200.0, "D", 900.0, 700.0, "X", 40.0, 240.0, 0),
        (25, "TRANSFER", 0.0, "E", 0.0, 0.0, "X", 240.0, 240.0, 0),   # zero amount, exactly 24h after step 1
        (26, "DEBIT", 10.0, "F", 10.0, 0.0, "X", 240.0, 250.0, 0),
        (200, "CASH_IN", 70.0, "G", 0.0, 70.0, "X", 250.0, 180.0, 0),  # more than 7 days after everything
    ])


def test_first_transaction_and_previous_average(raw):
    h = destination_history_features(raw)
    assert h["destination_is_first_transaction"].tolist() == [1, 1, 0, 0, 0, 0, 0]
    # Row 2 is in the same hour as row 0 but comes later, so row 0 is its history.
    assert h.loc[2, "destination_avg_previous_amount"] == 100.0
    assert h.loc[3, "destination_avg_previous_amount"] == pytest.approx((100 + 30) / 2)
    assert h.loc[3, "destination_amount_deviation"] == pytest.approx(200 - 65)
    # First-ever recipient: no history, deviation equals the amount.
    assert h.loc[1, "destination_avg_previous_amount"] == 0.0
    assert h.loc[1, "destination_amount_deviation"] == 50.0


def test_window_counts_exclude_current_hour_and_respect_bounds(raw):
    h = destination_history_features(raw)
    # Same-hour transactions are not counted.
    assert h.loc[2, "destination_transactions_last_24h"] == 0
    assert h.loc[3, "destination_transactions_last_24h"] == 2
    # Step 25: window is steps 1..24, so both step-1 rows and the step-5 row count.
    assert h.loc[4, "destination_transactions_last_24h"] == 3
    # Step 26: window is steps 2..25, so the step-1 rows have dropped out.
    assert h.loc[5, "destination_transactions_last_24h"] == 2
    assert h.loc[5, "destination_transactions_last_7d"] == 4
    # Step 200: nothing in the last 168 hours, but the recipient is not new.
    assert h.loc[6, "destination_transactions_last_7d"] == 0
    assert h.loc[6, "destination_is_first_transaction"] == 0


def test_history_is_aligned_to_the_input_index():
    """The bug this guards against: features computed on a re-sorted copy and
    attached by position, so every row received another row's history."""
    raw = make_raw([
        (1, "PAYMENT", 10.0, "A", 0, 0, "Z", 0, 0, 0),
        (2, "PAYMENT", 20.0, "B", 0, 0, "A_FIRST_ALPHABETICALLY", 0, 0, 0),
        (3, "PAYMENT", 30.0, "C", 0, 0, "Z", 0, 0, 0),
    ])
    raw.index = [10, 20, 30]
    h = destination_history_features(raw)
    assert list(h.index) == [10, 20, 30]
    assert h["destination_is_first_transaction"].tolist() == [1, 1, 0]
    assert h.loc[30, "destination_avg_previous_amount"] == 10.0


def test_balance_and_time_features(raw):
    f = build_feature_frame(raw)
    assert f.loc[0, "origin_balance_error"] == 0.0
    assert f.loc[1, "destination_balance_error"] == 0.0
    assert f.loc[0, "destination_balance_error"] == 100.0
    assert f.loc[0, "destination_balance_is_zero"] == 1
    assert f.loc[2, "destination_balance_is_zero"] == 0
    assert f.loc[4, "amount"] == 0.0 and f.loc[4, "destination_amount_deviation"] < 0
    # Hour of day wraps every 24 steps.
    assert f.loc[0, "hour_sin"] == pytest.approx(f.loc[4, "hour_sin"])
    assert f.loc[0, "hour_cos"] == pytest.approx(f.loc[4, "hour_cos"])
    assert set(FEATURES_V1) <= set(f.columns) and set(FEATURES_V2) <= set(f.columns)
    assert not f[FEATURES_V2].isna().any().any()


def test_online_features_match_batch_features():
    """Serving path == training path, for every row of a random table."""
    rng = np.random.default_rng(0)
    n = 400
    steps = np.sort(rng.integers(1, 400, n))
    types = rng.choice(["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"], n)
    amounts = np.round(rng.gamma(1.5, 5000, n), 2)
    amounts[rng.random(n) < 0.05] = 0.0
    raw = make_raw([
        (int(s), t, float(a), f"O{i}", float(ob), float(nb), f"D{d}", float(od), float(nd), 0)
        for i, (s, t, a, ob, nb, d, od, nd) in enumerate(zip(
            steps, types, amounts, rng.gamma(2, 9000, n), rng.gamma(2, 9000, n),
            rng.integers(0, 25, n), rng.choice([0.0, 100.0, 5e4], n), rng.choice([0.0, 100.0, 5e4], n),
            strict=True))
    ])
    batch = build_feature_frame(raw)

    history: dict[str, tuple[list, list]] = {}
    for i, row in raw.iterrows():
        past_steps, past_amounts = history.setdefault(row["nameDest"], ([], []))
        online = compute_online_features(
            step=row["step"], type=row["type"], amount=row["amount"],
            old_balance_origin=row["oldbalanceOrg"], new_balance_origin=row["newbalanceOrig"],
            old_balance_destination=row["oldbalanceDest"], new_balance_destination=row["newbalanceDest"],
            history_steps=past_steps, history_amounts=past_amounts,
        )
        for column in FEATURES_V2:
            assert float(online[column]) == pytest.approx(float(batch.loc[i, column]), rel=1e-9, abs=1e-9), column
        past_steps.append(row["step"])
        past_amounts.append(row["amount"])


def test_online_features_reject_bad_input():
    base = dict(step=1, amount=1.0, old_balance_origin=1.0, new_balance_origin=0.0,
                old_balance_destination=0.0, new_balance_destination=1.0)
    with pytest.raises(ValueError):
        compute_online_features(type="WIRE", **base)
    with pytest.raises(ValueError):
        compute_online_features(type="PAYMENT", history_steps=[1, 2], history_amounts=[1.0], **base)


def test_to_model_frame_enforces_columns_and_order():
    row = compute_online_features(step=3, type="TRANSFER", amount=5.0, old_balance_origin=5.0,
                                  new_balance_origin=0.0, old_balance_destination=0.0,
                                  new_balance_destination=0.0)
    frame = to_model_frame([row], FEATURES_V2)
    assert list(frame.columns) == FEATURES_V2
    with pytest.raises(KeyError):
        to_model_frame([row], FEATURES_V1)  # the hourly totals are not available online
