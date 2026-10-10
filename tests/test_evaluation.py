import numpy as np
import pytest

from fraud.calibration import fit_calibrator
from fraud.evaluation import (
    CostCurve,
    assign_tier,
    cost_report,
    paired_bootstrap_difference,
    pr_auc_with_ci,
    precision_at_k,
    psi,
    sweep_thresholds,
    threshold_robustness,
)


@pytest.fixture
def data():
    rng = np.random.default_rng(1)
    n = 4000
    y = (rng.random(n) < 0.03).astype(int)
    probs = np.clip(np.where(y == 1, rng.beta(5, 2, n), rng.beta(1, 12, n)), 0, 1)
    amounts = np.round(rng.gamma(2, 400, n), 2)
    return y, probs, amounts


def brute_cost(y, probs, amounts, t_review, t_block, cfb=5, cr=1):
    total = 0.0
    for label, p, a in zip(y, probs, amounts, strict=True):
        tier = assign_tier(p, t_review, t_block)
        if label == 1 and tier == "ALLOW":
            total += a
        elif label == 0 and tier == "BLOCK":
            total += cfb
        elif tier == "REVIEW":
            total += cr
    return total


def test_tier_boundaries_are_inclusive():
    assert assign_tier(0.32, 0.32, 0.85) == "REVIEW"
    assert assign_tier(0.85, 0.32, 0.85) == "BLOCK"
    assert assign_tier(0.3199, 0.32, 0.85) == "ALLOW"


def test_cost_report_matches_row_by_row_cost(data):
    y, probs, amounts = data
    report = cost_report(y, probs, amounts, 0.2, 0.7, cost_false_block=5, cost_review=1)
    assert report["total_cost"] == pytest.approx(brute_cost(y, probs, amounts, 0.2, 0.7))
    assert report["fraud_allowed"] + report["fraud_reviewed"] + report["fraud_blocked"] == y.sum()


def test_cost_curve_matches_cost_report(data):
    y, probs, amounts = data
    curve = CostCurve(y, probs, amounts, 5, 1)
    for t_review, t_block in [(0.0, 0.5), (0.1, 0.1), (0.25, 0.9), (0.4, 1.0)]:
        expected = cost_report(y, probs, amounts, t_review, t_block, 5, 1)
        assert float(curve.cost(t_review, t_block)) == pytest.approx(expected["total_cost"])
        assert int(curve.fraud_allowed(t_review)) == expected["fraud_allowed"]
        assert int(curve.legit_reviewed(t_review, t_block)) == expected["legit_reviewed"]


def test_sweep_finds_the_exhaustive_minimum(data):
    y, probs, amounts = data
    grid = np.round(np.linspace(0, 1, 41), 4)
    best = sweep_thresholds(y, probs, amounts, review_grid=grid, block_grid=grid, cost_false_block=5, cost_review=1)
    exhaustive = min(brute_cost(y, probs, amounts, r, b) for r in grid for b in grid if r <= b)
    assert best["cost_at_choice"] == pytest.approx(exhaustive)
    assert best["t_review"] <= best["t_block"]


def test_robust_sweep_never_reports_a_better_objective_than_the_plain_one(data):
    y, probs, amounts = data
    plain = sweep_thresholds(y, probs, amounts)
    robust = sweep_thresholds(y, probs, amounts, odds_margin=2.0)
    assert robust["objective"] >= plain["objective"] - 1e-9
    rows = threshold_robustness(y, probs, amounts, robust["t_review"], robust["t_block"])
    assert {r["moved"] for r in rows} == {"t_review", "t_block"}
    assert {r["kind"] for r in rows} == {"absolute", "odds_factor"}


def test_bootstrap_interval_contains_point_estimate(data):
    y, probs, _ = data
    result = pr_auc_with_ci(y, probs, n_iterations=100)
    assert result["ci_low"] <= result["pr_auc"] <= result["ci_high"]


def test_paired_bootstrap_detects_only_real_differences(data):
    y, probs, _ = data
    rng = np.random.default_rng(2)
    same = paired_bootstrap_difference(y, probs, probs, n_iterations=100)
    assert same["difference"] == 0 and not same["significant"]
    worse = rng.random(len(y))
    assert paired_bootstrap_difference(y, probs, worse, n_iterations=100)["significant"]


def test_precision_at_k_ceiling():
    y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    scores = np.linspace(1, 0, 10)
    result = precision_at_k(y, scores, 0.5)
    assert result["precision"] == pytest.approx(0.2) and result["max_possible"] == pytest.approx(0.2)


def test_psi_is_zero_for_identical_and_large_for_shifted():
    rng = np.random.default_rng(3)
    reference = rng.normal(0, 1, 20000)
    assert psi(reference, reference) == pytest.approx(0.0, abs=1e-9)
    assert psi(reference, rng.normal(0, 1, 20000)) < 0.01
    assert psi(reference, rng.normal(1.5, 1, 20000)) > 0.25


def test_calibrators_preserve_ranking_and_fix_the_mean(data):
    y, probs, _ = data
    raw = probs ** 3  # monotone distortion: same ranking, wrong probabilities
    for method in ("isotonic", "platt"):
        calibrated = fit_calibrator(raw, y, method).transform(raw)
        assert calibrated.mean() == pytest.approx(y.mean(), abs=0.005)
        order = np.argsort(raw)
        assert np.all(np.diff(calibrated[order]) >= -1e-12)


def test_fast_bootstrap_equals_sklearn_on_the_same_resamples():
    from sklearn.metrics import average_precision_score

    from fraud.evaluation import bootstrap_pr_auc
    rng = np.random.default_rng(4)
    n = 3000
    y = (rng.random(n) < 0.05).astype(int)
    scores = np.round(rng.random(n) * 0.5 + y * rng.random(n) * 0.5, 2)   # rounded: plenty of ties
    fast = bootstrap_pr_auc(y, scores, n_iterations=25, seed=7)
    state = np.random.RandomState(7)
    slow = []
    for _ in range(25):
        idx = state.randint(0, n, n)
        slow.append(average_precision_score(y[idx], scores[idx]))
    assert np.allclose(fast, slow, rtol=0, atol=1e-12)
