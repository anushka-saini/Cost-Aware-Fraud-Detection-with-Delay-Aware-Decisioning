"""The held-out breakdown by type and amount (scripts/v3_breakdown.py)."""

import pytest

from fraud.results import load_metrics


def test_breakdown_adds_up_to_the_held_out_test(client, analyst):
    metrics = load_metrics()
    if "v3_breakdown" not in metrics:
        pytest.skip("scripts/v3_breakdown.py has not been run")
    breakdown = client.get("/explainability", headers=analyst).json()["breakdown"]
    assert breakdown == metrics["v3_breakdown"]

    held = metrics["v3_held_out"]
    point = held["v3"]["operating_point"]
    for rows in (breakdown["by_type"], breakdown["by_amount_quartile"]):
        assert sum(r["n_rows"] for r in rows) == held["n_rows"]
        assert sum(r["n_fraud"] for r in rows) == held["n_fraud"]
        assert sum(r["fraud_allowed"] for r in rows) == point["fraud_allowed"]
        assert sum(r["legit_blocked"] for r in rows) == point["legit_blocked"]
        blocked = sum(r["block_rate"] * r["n_rows"] for r in rows)
        assert blocked == pytest.approx(point["fraud_blocked"] + point["legit_blocked"])
