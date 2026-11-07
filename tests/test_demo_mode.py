"""Public demo mode: no admin action may change production, and visitors'
work is cleared once a day (fraud/service/demo.py)."""

from datetime import timedelta

import pytest
from sqlalchemy import func, select

from fraud.service import core, demo
from fraud.service import db as database
from fraud.service.settings import get_service_settings

VISITOR_TABLES = [database.Transaction, database.Assessment, database.AnalystCase, database.CaseNote,
                  database.Decision, database.FeedbackLabel]


@pytest.fixture
def demo_on(client, monkeypatch):
    monkeypatch.setattr(get_service_settings(), "demo_mode", True)
    monkeypatch.setattr(demo, "_checked_at", 0.0)
    return client


def count(session, table) -> int:
    return session.scalar(select(func.count()).select_from(table))


def add_visitor_work():
    """One replayed transaction sent to review, with a note, a decision and the label it produced."""
    with database.SessionLocal() as s:
        t = database.Transaction(step=340, type="TRANSFER", amount=10.0, name_dest="C1", source="stream")
        s.add(t)
        s.flush()
        a = database.Assessment(id="a-visitor", transaction_id=t.id, model_name="fraud_scorer", model_version="v3",
                                amount=10.0, raw_score=0.1, fraud_probability=0.01, tier="REVIEW",
                                t_review=0.0003, t_block=0.71)
        s.add(a)
        s.flush()
        case = database.AnalystCase(assessment_id=a.id, amount=10.0, fraud_probability=0.01)
        s.add(case)
        s.flush()
        s.add(database.CaseNote(case_id=case.id, author="analyst", note="looks fine"))
        s.add(database.Decision(assessment_id=a.id, actor="analyst", actor_role="analyst", action="approve"))
        s.add(database.FeedbackLabel(transaction_id=str(t.id), is_fraud=0, source="analyst",
                                     transaction_step=340, decided_step=340))
        s.add(database.AuditLog(actor="analyst", role="analyst", action="case.approve", entity_type="case",
                                entity_id=str(case.id)))
        s.commit()


def test_demo_refuses_model_switch_and_only_logs_proposals(demo_on, admin):
    scorer = core.get_scorer()
    before = (scorer.version, scorer.t_review, scorer.t_block)

    response = demo_on.post("/models/rollback", json={"reason": "a visitor tries"}, headers=admin)
    assert response.status_code == 403

    proposal = {"t_review": 0.01, "t_block": 0.5, "cost_false_block": 5, "cost_review": 1, "split": "eval_2",
                "rationale": "a visitor's idea"}
    response = demo_on.post("/thresholds/proposals", json=proposal, headers=admin)
    assert response.status_code == 200
    assert response.json()["applied"] is False

    scorer = core.get_scorer()
    assert (scorer.version, scorer.t_review, scorer.t_block) == before
    logged = demo_on.get("/audit", params={"action": "thresholds.propose"}, headers=admin).json()
    assert logged["total"] == 1


def test_first_check_only_starts_the_clock(demo_on):
    add_visitor_work()
    with database.SessionLocal() as s:
        assert demo.reset_if_due(s) is False
        assert demo.last_reset(s) is not None
        assert all(count(s, table) == 1 for table in VISITOR_TABLES)


def test_visitor_work_is_cleared_after_a_day(demo_on):
    with database.SessionLocal() as s:
        demo.reset_if_due(s)                               # starts the clock
        started = demo.last_reset(s)
        users = count(s, database.User)
        versions = count(s, database.ModelVersion)
        reports = count(s, database.DriftReport)
    add_visitor_work()

    with database.SessionLocal() as s:
        assert demo.reset_if_due(s, started + timedelta(hours=23)) is False
        assert all(count(s, table) == 1 for table in VISITOR_TABLES)

        assert demo.reset_if_due(s, started + timedelta(hours=25)) is True
        assert all(count(s, table) == 0 for table in VISITOR_TABLES)
        # the audit log keeps only the new reset entry
        entries = s.scalars(select(database.AuditLog)).all()
        assert [e.action for e in entries] == [demo.RESET_ACTION]
        assert entries[0].detail["removed"]["analyst_cases"] == 1
        # what is not visitors' work stays
        assert (count(s, database.User), count(s, database.ModelVersion), count(s, database.DriftReport)) == \
               (users, versions, reports)


def test_no_reset_outside_demo_mode(client):
    assert get_service_settings().demo_mode is False
    add_visitor_work()
    with database.SessionLocal() as s:
        assert demo.reset_if_due(s) is False
        assert demo.last_reset(s) is None
        assert count(s, database.AnalystCase) == 1


def test_status_shows_the_reset_schedule_in_demo(demo_on):
    body = demo_on.get("/status").json()            # the first request also starts the clock
    reset = body["demo_reset"]
    assert reset["every_hours"] == get_service_settings().demo_reset_hours
    assert reset["last_reset_at"] and reset["next_reset_at"]


def test_status_has_no_reset_schedule_outside_demo(client):
    assert client.get("/status").json()["demo_reset"] is None
