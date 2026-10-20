import numpy as np
import pandas as pd

from fraud.monitoring import DriftReportStore, binary_psi, build_report, psi_level, rate_check


def frames(shift=0.0, flag_rate=0.3, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"amount": rng.normal(shift, 1, n),
                         "destination_is_first_transaction": (rng.random(n) < flag_rate).astype(int)})


def test_psi_levels():
    assert [psi_level(v) for v in (0.05, 0.10, 0.25, 0.26)] == ["ok", "warning", "warning", "alert"]


def test_binary_psi():
    assert binary_psi([0, 1, 0, 1], [1, 0, 1, 0]) == 0.0
    assert binary_psi([0] * 90 + [1] * 10, [0] * 50 + [1] * 50) > 0.25


def test_stable_traffic_raises_nothing():
    reference, current = frames(seed=1), frames(seed=2)
    report = build_report(period="p", days=[8], reference=reference, current=current,
                          feature_columns=list(reference.columns),
                          reference_scores={"model_score": reference["amount"].to_numpy()},
                          current_scores={"model_score": current["amount"].to_numpy()},
                          review_rate=0.03, expected_review_rate=0.028)
    assert report["status"] == "ok" and report["alerts"] == []
    assert {r["name"] for r in report["psi"]} == {"amount", "destination_is_first_transaction", "model_score"}


def test_shifted_feature_and_score_raise_alerts():
    reference, current = frames(seed=1), frames(shift=1.5, flag_rate=0.8, seed=2)
    report = build_report(period="p", days=[8], reference=reference, current=current,
                          feature_columns=list(reference.columns),
                          reference_scores={"model_score": reference["amount"].to_numpy()},
                          current_scores={"model_score": current["amount"].to_numpy()})
    assert report["status"] == "alert"
    assert {a["name"] for a in report["alerts"]} == {"amount", "destination_is_first_transaction", "model_score"}


def test_rate_outside_band_raises_an_alert():
    assert rate_check("review_rate", 0.03, 0.028)["level"] == "ok"
    assert rate_check("review_rate", 0.10, 0.028)["level"] == "alert"       # more than 2x expected
    assert rate_check("review_rate", 0.01, 0.028)["level"] == "alert"       # less than half
    assert rate_check("override_rate", 0.45, None, (0.0, 0.30), absolute=True)["level"] == "alert"
    reference = frames(seed=1)
    report = build_report(period="p", days=[8], reference=reference, current=frames(seed=2),
                          feature_columns=["amount"], reference_scores={}, current_scores={},
                          review_rate=0.2, expected_review_rate=0.028, override_rate=0.5)
    assert report["status"] == "alert" and {a["name"] for a in report["alerts"]} == {"review_rate", "override_rate"}


def test_store_keeps_one_report_per_period(tmp_path):
    store = DriftReportStore(tmp_path / "m" / "drift.json")
    assert store.all() == [] and store.latest() is None
    store.add({"period": "eval_1", "status": "ok"})
    store.add({"period": "eval_2", "status": "warning"})
    store.add({"period": "eval_1", "status": "alert"})          # a re-run replaces the old report
    assert [r["period"] for r in store.all()] == ["eval_2", "eval_1"]
    assert store.latest()["status"] == "alert"
