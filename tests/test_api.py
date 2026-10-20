"""Endpoint tests: scoring, review queue, auth and roles, labels, registry, audit, live feed."""

import pytest
from conftest import needs_model

from fraud.service.core import get_scorer

pytestmark = needs_model

RAW = {"step": 10, "type": "TRANSFER", "amount": 250000.0, "nameDest": "C_TEST", "nameOrig": "C_ORIG",
       "oldbalanceOrg": 250000.0, "newbalanceOrig": 0.0, "oldbalanceDest": 0.0, "newbalanceDest": 0.0}
LEGACY = {"amount": 1595587.46, "origin_balance_error": 0, "destination_balance_error": 1595587.46,
          "destination_balance_is_zero": 1, "destination_is_first_transaction": 1, "type_TRANSFER": True, "step": 5}


def EXPLAINED_INPUTS():
    """What an explanation lists: every model input, with the two hour columns merged into one."""
    return (set(get_scorer().feature_columns) - {"hour_sin", "hour_cos"}) | {"hour_of_day"}


def review_payload():
    """A raw transaction the active model sends to REVIEW (found by trying a few shapes)."""
    scorer = get_scorer()
    for amount in (900.0, 5000.0, 20000.0, 60000.0, 150000.0, 400000.0, 1200000.0):
        for kind in ("CASH_OUT", "TRANSFER"):
            for new_dest in (0.0, amount):
                txn = {**RAW, "type": kind, "amount": amount, "oldbalanceOrg": amount, "newbalanceDest": new_dest}
                scored = scorer.score_transaction(
                    step=txn["step"], type=kind, amount=amount, old_balance_origin=amount, new_balance_origin=0.0,
                    old_balance_destination=0.0, new_balance_destination=new_dest)
                if scored["tier"] == "REVIEW":
                    return txn
    pytest.skip("could not construct a REVIEW-tier transaction for the active model")


@pytest.fixture
def case_id(client, analyst):
    scored = client.post("/score/raw", json=review_payload()).json()
    assert scored["tier"] == "REVIEW"
    cases = client.get("/cases", headers=analyst).json()["cases"]
    return next(c["id"] for c in cases if c["assessment_id"] == scored["assessment_id"])


# --- system ---------------------------------------------------------------------------------------

def test_health_and_model_info(client):
    assert client.get("/health").json()["status"] == "ok"
    info = client.get("/model-info").json()
    assert info["feature_columns"] == get_scorer().feature_columns
    assert info["thresholds"]["t_review"] <= info["thresholds"]["t_block"]
    # The leaky whole-hour totals must not be model inputs any more.
    assert not {"total_transactions", "total_transaction_amount", "avg_transaction_amount"} & set(info["feature_columns"])


def test_every_response_carries_a_request_id(client):
    response = client.get("/health", headers={"X-Request-ID": "trace-123"})
    assert response.headers["X-Request-ID"] == "trace-123"
    assert client.get("/health").headers["X-Request-ID"]
    missing = client.get("/cases/1")
    assert missing.status_code == 401 and missing.json()["request_id"] == missing.headers["X-Request-ID"]


# --- scoring --------------------------------------------------------------------------------------

def test_score_raw_returns_a_tier_and_builds_history(client):
    first = client.post("/score/raw", json=RAW).json()
    assert first["tier"] in {"ALLOW", "REVIEW", "BLOCK"}
    assert 0.0 <= first["fraud_probability"] <= 1.0
    assert first["features"]["destination_is_first_transaction"] == 1
    # The same recipient again: the feature store now knows about the first transaction.
    second = client.post("/score/raw", json={**RAW, "step": 12}).json()
    assert second["features"]["destination_is_first_transaction"] == 0
    assert second["features"]["destination_transactions_last_24h"] == 1
    assert second["features"]["destination_avg_previous_amount"] == RAW["amount"]


def test_recorded_transactions_count_as_history(client):
    assert client.post("/transactions", json={"nameDest": "C_TEST", "step": 3, "amount": 100.0}).status_code == 200
    scored = client.post("/score/raw", json=RAW).json()
    assert scored["features"]["destination_is_first_transaction"] == 0
    assert scored["features"]["destination_avg_previous_amount"] == 100.0


def test_score_raw_validates_input(client):
    bad_type = client.post("/score/raw", json={**RAW, "type": "WIRE"})
    assert bad_type.status_code == 422 and bad_type.json()["errors"][0]["field"] == "type"
    assert client.post("/score/raw", json={**RAW, "amount": -1}).status_code == 422
    assert client.post("/score/raw", json={k: v for k, v in RAW.items() if k != "nameDest"}).status_code == 422


def test_legacy_score_ignores_the_removed_hourly_fields(client):
    plain = client.post("/score", json=LEGACY).json()
    with_hourly = client.post("/score", json={**LEGACY, "total_transactions": 10, "total_transaction_amount": 1.0,
                                              "avg_transaction_amount": 0.1}).json()
    assert plain["fraud_probability"] == with_hourly["fraud_probability"]
    assert set(plain) >= {"assessment_id", "fraud_probability", "tier", "reasons", "thresholds_used", "anomaly_score"}


def test_flagged_transactions_come_with_reasons(client):
    scored = client.post("/score", json=LEGACY).json()
    if scored["tier"] != "ALLOW":
        assert 1 <= len(scored["reasons"]) <= 3


def test_explain_lists_every_feature(client):
    explained = client.post("/explain", json=LEGACY).json()
    assert {c["feature"] for c in explained["contributions"]} == EXPLAINED_INPUTS()
    assert explained["units"] in {"probability", "log-odds"}
    # A transfer is never described as "being a payment": false flags read as their absence.
    labels = {c["feature"]: c["label"] for c in explained["contributions"]}
    assert labels["type_TRANSFER"] == "this being a transfer" and labels["type_PAYMENT"] == "this not being a payment"


def test_confirm_decision_is_logged_once(client):
    scored = client.post("/score", json=LEGACY).json()
    confirm = {"assessment_id": scored["assessment_id"], "user_proceeded": False}
    assert client.post("/confirm-decision", json=confirm).json() == {"status": "logged"}
    assert "error" in client.post("/confirm-decision", json=confirm).json()
    assert "error" in client.post("/confirm-decision", json={"assessment_id": "nope", "user_proceeded": True}).json()
    rows = client.get("/decision-log").json()["rows"]
    assert rows[-1]["assessment_id"] == scored["assessment_id"] and rows[-1]["user_proceeded"] is False


# --- auth -----------------------------------------------------------------------------------------

def test_login_and_roles(client, analyst, admin):
    assert client.get("/auth/me", headers=analyst).json() == {"username": "analyst", "role": "analyst"}
    assert client.get("/auth/me", headers=admin).json()["role"] == "admin"
    assert client.post("/auth/login", json={"username": "analyst", "password": "wrong"}).status_code == 401
    assert client.post("/auth/login", json={"username": "nobody", "password": "x"}).status_code == 401


@pytest.mark.parametrize("path", ["/cases", "/audit", "/models", "/monitoring/drift", "/overview", "/labels",
                                  "/simulator/scores"])
def test_operations_endpoints_require_a_token(client, path):
    assert client.get(path).status_code == 401
    assert client.get(path, headers={"Authorization": "Bearer not-a-token"}).status_code == 401


def test_admin_only_endpoints_reject_analysts(client, analyst, admin):
    proposal = {"t_review": 0.001, "t_block": 0.8, "cost_false_block": 5, "cost_review": 1, "split": "eval_2",
                "rationale": "fewer reviews at the same fraud loss"}
    assert client.post("/thresholds/proposals", json=proposal, headers=analyst).status_code == 403
    assert client.post("/models/rollback", json={"reason": "testing"}, headers=analyst).status_code == 403
    before = client.get("/model-info").json()["thresholds"]
    accepted = client.post("/thresholds/proposals", json=proposal, headers=admin).json()
    assert accepted == {"status": "proposal_recorded", "applied": False}
    assert client.get("/model-info").json()["thresholds"] == before          # a proposal never changes production
    logged = client.get("/audit", params={"action": "thresholds.propose"}, headers=admin).json()["rows"]
    assert logged[0]["detail"]["t_block"] == 0.8 and logged[0]["detail"]["current"] == before
    reversed_ = {**proposal, "t_review": 0.9, "t_block": 0.1}
    assert client.post("/thresholds/proposals", json=reversed_, headers=admin).status_code == 422


# --- review queue ---------------------------------------------------------------------------------

def test_review_tier_opens_a_case_with_full_detail(client, analyst, case_id):
    detail = client.get(f"/cases/{case_id}", headers=analyst).json()
    assert detail["status"] == "open" and detail["tier"] == "REVIEW"
    assert detail["transaction"]["name_dest"] == "C_TEST"
    assert detail["scores"]["anomaly_score"] is not None
    assert {c["feature"] for c in detail["explanation"]["contributions"]} == EXPLAINED_INPUTS()
    assert not any(reason.startswith("this not being") for reason in detail["reasons"])
    assert any(e["action"] == "assessment.review" for e in detail["audit_trail"])
    assert client.get("/cases/999999", headers=analyst).status_code == 404


def test_allowed_transactions_do_not_open_cases(client, analyst):
    payload = {**RAW, "type": "PAYMENT", "amount": 12.5, "oldbalanceOrg": 5000.0, "newbalanceOrig": 4987.5,
               "oldbalanceDest": 100.0, "newbalanceDest": 112.5}
    scored = client.post("/score/raw", json=payload).json()
    if scored["tier"] == "ALLOW":
        assert client.get("/cases", params={"status": "all"}, headers=analyst).json()["total"] == 0


def test_decision_needs_a_note_and_closes_the_case(client, analyst, case_id):
    assert client.post(f"/cases/{case_id}/decision", json={"action": "decline"}, headers=analyst).status_code == 422
    assert client.post(f"/cases/{case_id}/decision", json={"action": "decline", "note": ""}, headers=analyst).status_code == 422
    done = client.post(f"/cases/{case_id}/decision", json={"action": "decline", "note": "customer did not recognise it"},
                       headers=analyst)
    assert done.json()["status"] == "declined"
    detail = client.get(f"/cases/{case_id}", headers=analyst).json()
    assert detail["decided_by"] == "analyst" and detail["closed_at"]
    assert detail["decisions"][-1]["note"] == "customer did not recognise it"
    assert any(e["action"] == "case.decline" for e in detail["audit_trail"])
    # Deciding twice is refused, and the decline became a fraud label for retraining.
    again = client.post(f"/cases/{case_id}/decision", json={"action": "approve", "note": "changed my mind"}, headers=analyst)
    assert again.status_code == 409
    labels = client.get("/labels", headers=analyst).json()["rows"]
    assert labels[-1]["is_fraud"] is True and labels[-1]["source"] == "analyst"
    assert labels[-1]["transaction_id"] == str(detail["transaction"]["id"])


def test_escalated_cases_wait_for_an_admin(client, analyst, admin, case_id):
    client.post(f"/cases/{case_id}/decision", json={"action": "escalate", "note": "unsure, large amount"}, headers=analyst)
    assert client.get("/cases", params={"status": "escalated"}, headers=analyst).json()["total"] == 1
    blocked = client.post(f"/cases/{case_id}/decision", json={"action": "approve", "note": "looks fine"}, headers=analyst)
    assert blocked.status_code == 403
    resolved = client.post(f"/cases/{case_id}/decision", json={"action": "approve", "note": "verified with the customer"},
                           headers=admin)
    assert resolved.json()["status"] == "approved"


def test_queue_sorting_filtering_and_bulk_decisions(client, analyst):
    base = review_payload()
    for i, name in enumerate(["C_A", "C_B", "C_C"]):
        client.post("/score/raw", json={**base, "nameDest": name, "step": 20 + i})
    listing = client.get("/cases", params={"sort": "age"}, headers=analyst).json()
    assert listing["total"] == 3 and listing["counts_by_status"] == {"open": 3}
    assert [c["name_dest"] for c in listing["cases"]] == ["C_A", "C_B", "C_C"]
    assert client.get("/cases", params={"q": "C_B"}, headers=analyst).json()["total"] == 1
    assert client.get("/cases", params={"sort": "nonsense"}, headers=analyst).status_code == 422
    ids = [c["id"] for c in listing["cases"]]
    bulk = client.post("/cases/bulk-decision", json={"case_ids": [*ids[:2], 424242], "action": "approve",
                                                     "note": "known merchant payouts"}, headers=analyst).json()
    assert bulk["decided"] == ids[:2] and bulk["failed"][0]["case_id"] == 424242
    assert client.get("/cases", headers=analyst).json()["total"] == 1
    note = client.post(f"/cases/{ids[2]}/notes", json={"note": "called the customer, no answer"}, headers=analyst)
    assert note.status_code == 200
    assert client.get(f"/cases/{ids[2]}", headers=analyst).json()["notes"][-1]["author"] == "analyst"


# --- labels, registry, monitoring, audit ------------------------------------------------------------

def test_labels_are_stored_with_maturity(client, analyst):
    label = {"transaction_id": "tx-1", "is_fraud": True, "source": "chargeback", "transaction_step": 300,
             "decided_step": 310, "note": "confirmed by the bank"}
    stored = client.post("/labels", json=label, headers=analyst).json()
    assert stored["usable_for_training_from_step"] == 300 + int(stored["label_maturity_days"] * 24)
    early = client.get("/labels", params={"as_of_step": 320}, headers=analyst).json()["rows"][-1]
    late = client.get("/labels", params={"as_of_step": 400}, headers=analyst).json()["rows"][-1]
    assert early["mature"] is False and late["mature"] is True
    backwards = {**label, "transaction_id": "tx-2", "decided_step": 200}
    assert client.post("/labels", json=backwards, headers=analyst).status_code == 422


def test_model_registry_lists_versions_and_rolls_back(client, analyst, admin):
    listing = client.get("/models", headers=analyst).json()
    versions = [v["version"] for v in listing["versions"]]
    assert listing["active"] == versions[-1]
    assert [v["is_active"] for v in listing["versions"]].count(True) == 1
    if len(versions) < 2:
        pytest.skip("only one registered version: nothing to roll back to")
    try:
        rolled = client.post("/models/rollback", json={"reason": "testing rollback"}, headers=admin).json()
        assert rolled == {"status": "activated", "from": versions[-1], "active": versions[-2]}
        assert client.get("/health").json()["model"]["version"] == versions[-2]
        unknown = client.post("/models/rollback", json={"version": "v999", "reason": "bad version"}, headers=admin)
        assert unknown.status_code == 404
        entry = client.get("/audit", params={"action": "model.activate"}, headers=admin).json()["rows"][0]
        assert entry["detail"]["reason"] == "testing rollback" and entry["actor"] == "admin"
    finally:
        client.post("/models/rollback", json={"version": versions[-1], "reason": "restore after test"}, headers=admin)
    assert client.get("/health").json()["model"]["version"] == versions[-1]


def test_monitoring_and_overview(client, analyst):
    client.post("/score/raw", json=RAW)
    drift = client.get("/monitoring/drift", headers=analyst).json()
    assert drift["status"] in {"ok", "warning", "alert", "no_reports"} and drift["live"]["assessments"] == 1
    overview = client.get("/overview", headers=analyst).json()
    assert overview["kpis"]["scored_total"] == 1 and overview["kpis"]["active_model"]["version"]
    assert sum(sum(b[t] for t in ("ALLOW", "REVIEW", "BLOCK")) for b in overview["tier_mix"]) == 1


def test_audit_log_search_and_csv_export(client, analyst, case_id):
    client.post(f"/cases/{case_id}/decision", json={"action": "approve", "note": "regular supplier"}, headers=analyst)
    everything = client.get("/audit", headers=analyst).json()
    assert everything["total"] >= 3 and "case.approve" in everything["actions"]
    mine = client.get("/audit", params={"actor": "analyst", "action": "case."}, headers=analyst).json()
    assert [r["action"] for r in mine["rows"]] == ["case.approve"]
    assert client.get("/audit", params={"q": "no-such-thing"}, headers=analyst).json()["total"] == 0
    export = client.get("/audit/export.csv", params={"action": "case."}, headers=analyst)
    assert export.headers["content-type"].startswith("text/csv")
    lines = export.text.strip().splitlines()
    assert lines[0].startswith("id,at,actor") and len(lines) == 2 and "regular supplier" in lines[1]


# --- live feed ------------------------------------------------------------------------------------

def test_stream_requires_a_token_and_replays_transactions(client, analyst, tmp_path, monkeypatch):
    from starlette.websockets import WebSocketDisconnect

    from fraud.service import stream
    replay = tmp_path / "replay.csv"
    replay.write_text(
        "step,type,nameOrig,nameDest,amount,oldbalanceOrg,newbalanceOrig,oldbalanceDest,newbalanceDest,actual_fraud\n"
        "340,PAYMENT,C1,M1,12.5,5000,4987.5,0,0,0\n"
        "340,TRANSFER,C2,C9,250000,250000,0,0,0,1\n"
        "341,CASH_OUT,C3,C9,250000,250000,0,0,250000,1\n", encoding="utf-8")
    original = stream.load_replay
    monkeypatch.setattr(stream, "load_replay", lambda path=None: original(replay))

    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/stream?token=bad") as ws:
            ws.receive_json()

    token = analyst["Authorization"].split()[1]
    with client.websocket_connect(f"/stream?token={token}&speed=200&restart=true") as ws:
        assert ws.receive_json() == {"event": "start", "total": 3, "offset": 0, "speed": 200.0}
        events = [ws.receive_json() for _ in range(3)]
        assert ws.receive_json() == {"event": "end", "sent": 3}
    assert [e["position"] for e in events] == [1, 2, 3]
    assert all(e["tier"] in {"ALLOW", "REVIEW", "BLOCK"} and "anomaly_score" in e for e in events)
    assert events[1]["actual_fraud"] == 1 and events[2]["name_dest"] == "C9"
    # Replayed transactions went through the real scoring path, so they are stored.
    assert client.get("/overview", headers=analyst).json()["kpis"]["scored_total"] == 3

