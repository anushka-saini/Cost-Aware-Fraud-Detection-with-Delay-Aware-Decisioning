import pytest

from fraud.labels import Label, LabelStore, last_fully_labelled_day


def test_label_is_usable_only_after_decision_and_maturity(tmp_path):
    store = LabelStore(tmp_path / "labels.csv")
    store.add(Label("t1", 1, "analyst", transaction_step=100, decided_step=110))
    # Decided, but the 2-day maturity window (48 steps) has not passed.
    assert store.usable(as_of_step=120, maturity_days=2) == {}
    assert "t1" in store.usable(as_of_step=148, maturity_days=2)
    # With no maturity requirement the decision time is the only gate.
    assert store.usable(as_of_step=109, maturity_days=0) == {}
    assert "t1" in store.usable(as_of_step=110, maturity_days=0)


def test_stronger_source_overrides_weaker_one(tmp_path):
    store = LabelStore(tmp_path / "labels.csv")
    store.add(Label("t1", 0, "analyst", transaction_step=10, decided_step=12, note="approved in review"))
    store.add(Label("t1", 1, "chargeback", transaction_step=10, decided_step=40))
    store.add(Label("t1", 0, "customer", transaction_step=10, decided_step=50))
    assert store.usable(as_of_step=30, maturity_days=0)["t1"].is_fraud == 0     # only the analyst label exists yet
    final = store.usable(as_of_step=100, maturity_days=0)["t1"]
    assert (final.is_fraud, final.source) == (1, "chargeback")


def test_store_persists_and_validates(tmp_path):
    path = tmp_path / "nested" / "labels.csv"
    LabelStore(path).add(Label("t9", 1, "analyst", 5, 6, note="a, note with a comma"))
    assert LabelStore(path).all()[0].note == "a, note with a comma"
    with pytest.raises(ValueError):
        LabelStore(path).add(Label("t9", 1, "rumour", 5, 6))
    with pytest.raises(ValueError):
        LabelStore(path).add(Label("t9", 2, "analyst", 5, 6))


def test_last_fully_labelled_day():
    assert last_fully_labelled_day(as_of_day=14, maturity_days=0) == 13
    assert last_fully_labelled_day(as_of_day=14, maturity_days=2) == 11
