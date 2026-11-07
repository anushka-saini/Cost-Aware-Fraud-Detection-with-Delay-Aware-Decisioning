"""The rule-engine baselines (scripts/rule_baselines.py)."""

import importlib.util

import pandas as pd
import pytest

from fraud.config import PROJECT_ROOT


def load_script():
    spec = importlib.util.spec_from_file_location("rule_baselines", PROJECT_ROOT / "scripts" / "rule_baselines.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tiny_frame():
    """Six transactions: two frauds that empty the account, one large legit
    transfer, one small legit cash-out, one flagged fraud, one payment."""
    return pd.DataFrame({
        "isFraud":        [1, 1, 0, 0, 1, 0],
        "isFlaggedFraud": [0, 0, 0, 0, 1, 0],
        "amount":         [500.0, 9000.0, 20000.0, 50.0, 7000.0, 30.0],
        "oldbalanceOrg":  [500.0, 9000.0, 90000.0, 400.0, 7000.0, 100.0],
        "newbalanceOrig": [0.0, 0.0, 70000.0, 350.0, 0.0, 70.0],
        "type_TRANSFER":  [1, 0, 1, 0, 1, 0],
        "type_CASH_OUT":  [0, 1, 0, 1, 0, 0],
    })


def test_whole_balance_rule_and_tiers():
    rules = load_script()
    d = tiny_frame()
    assert list(rules.hits(d, "empties_account")) == [True, True, False, False, True, False]
    # every hit of the whole-balance rule is fraud here, so it blocks
    assert rules.tier_for(d, rules.hits(d, "empties_account")) == "BLOCK"
    # the large-amount rule at 5,000 hits two frauds and one legit: 67% fraud, so it reviews
    assert rules.tier_for(d, rules.hits(d, "large_amount", 5000)) == "REVIEW"


def test_rule_engine_takes_the_highest_tier_and_costs_add_up():
    rules = load_script()
    d = tiny_frame()
    plan = {"cut_off": 5000.0, "tiers": {"paysim_flag": "BLOCK", "empties_account": "BLOCK", "large_amount": "REVIEW"}}
    tiers = rules.apply(d, plan, "rule_engine")
    assert list(tiers) == [2, 2, 1, 0, 2, 0]           # block, block, review, allow, block, allow
    result = rules.report(d, tiers)
    assert result["fraud_allowed"] == 0
    assert result["total_cost"] == 1.0                 # one review, nothing missed, no wrongful block
    assert result["recall"] == 1.0
    assert result["precision"] == pytest.approx(3 / 4)
    assert result["review_rate"] == pytest.approx(1 / 6)
    assert result["block_rate"] == pytest.approx(3 / 6)


def test_model_lab_rule_baselines_come_from_metrics_json(client, analyst):
    from fraud.results import load_metrics

    section = load_metrics().get("rule_baselines")
    if section is None:
        pytest.skip("scripts/rule_baselines.py has not been run")
    rules = client.get("/results/v3", headers=analyst).json()["rules"]
    assert rules["best_rule"] == section["best_rule"]
    assert rules["rules"] == section["rules"]
    assert rules["v3"]["days_14_16"]["total_cost"] == load_metrics()["v3_held_out"]["v3"]["operating_point"]["total_cost"]
