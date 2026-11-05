"""Phase 4 hardening: the status endpoint, demo mode, security basics and
running several workers."""

import pytest
from conftest import MODEL_NAME, needs_model
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import func, select, update

from fraud.registry import list_versions
from fraud.service import core
from fraud.service import db as database
from fraud.service.middleware import RateLimitMiddleware
from fraud.service.settings import DEV_SECRET, ServiceSettings, check_settings, get_service_settings
from fraud.service.setup import prepare_database

STRONG_SECRET = "a-long-random-secret-for-the-tests-0123456789"


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def test_production_refuses_unsafe_settings():
    unsafe = ServiceSettings(environment="production", jwt_secret=DEV_SECRET)
    with pytest.raises(RuntimeError, match="FRAUD_JWT_SECRET"):
        check_settings(unsafe)

    demo_passwords = ServiceSettings(environment="production", jwt_secret=STRONG_SECRET)
    with pytest.raises(RuntimeError, match="demo passwords"):
        check_settings(demo_passwords)


def test_production_accepts_safe_settings_and_a_public_demo():
    own_passwords = ServiceSettings(environment="production", jwt_secret=STRONG_SECRET,
                                    admin_password="a-real-password-1", analyst_password="a-real-password-2")
    assert check_settings(own_passwords) == []
    public_demo = ServiceSettings(environment="production", jwt_secret=STRONG_SECRET, demo_mode=True)
    assert check_settings(public_demo) == []


def test_development_only_warns():
    problems = check_settings(ServiceSettings(environment="development", jwt_secret=DEV_SECRET))
    assert any("FRAUD_JWT_SECRET" in p for p in problems)


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

def test_login_attempts_have_their_own_limit():
    app = FastAPI()

    @app.post("/auth/login")
    def login():
        return {"ok": True}

    app.add_middleware(RateLimitMiddleware, per_minute=100, login_per_minute=2)
    client = TestClient(app)
    codes = [client.post("/auth/login").status_code for _ in range(3)]
    assert codes == [200, 200, 429]


@needs_model
def test_security_headers_on_every_response(client):
    response = client.get("/health")
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "no-referrer"


@needs_model
def test_oversized_requests_are_refused(client):
    too_big = "x" * (get_service_settings().max_request_bytes + 1)
    response = client.post("/score/raw", content=too_big, headers={"Content-Type": "application/json"})
    assert response.status_code == 413


# ---------------------------------------------------------------------------
# Status and demo mode
# ---------------------------------------------------------------------------

@needs_model
def test_status_reports_model_database_and_mode(client):
    response = client.get("/status")
    assert response.status_code == 200, response.text
    body = response.json()
    scorer = core.get_scorer()
    assert body["status"] == "ok" and body["database"]["ok"] is True
    assert body["database"]["backend"] == "sqlite"
    assert body["model"]["version"] == scorer.version
    assert body["model"]["thresholds"] == {"t_review": scorer.t_review, "t_block": scorer.t_block}
    assert body["demo_mode"] is False
    assert "sqlite:" not in response.text          # never the connection string


@needs_model
def test_demo_accounts_are_hidden_in_production(client, monkeypatch):
    settings = get_service_settings()
    assert client.get("/status").json()["demo_accounts"]          # development, demo passwords: shown
    monkeypatch.setattr(settings, "environment", "production")
    assert client.get("/status").json()["demo_accounts"] is None
    monkeypatch.setattr(settings, "demo_mode", True)
    assert client.get("/status").json()["demo_accounts"]          # a public demo: shown


@needs_model
def test_demo_mode_shows_accounts_and_blocks_rollback(client, admin, monkeypatch):
    monkeypatch.setattr(get_service_settings(), "demo_mode", True)
    accounts = client.get("/status").json()["demo_accounts"]
    assert {a["role"] for a in accounts} == {"analyst", "admin"}
    response = client.post("/models/rollback", json={"reason": "demo visitor"}, headers=admin)
    assert response.status_code == 403


# ---------------------------------------------------------------------------
# Several workers
# ---------------------------------------------------------------------------

@needs_model
def test_setup_is_safe_to_run_twice(client):
    prepare_database()
    prepare_database()
    with database.SessionLocal() as session:
        assert session.scalar(select(func.count()).select_from(database.User)) == 2
        versions = session.scalar(select(func.count()).select_from(database.ModelVersion))
    assert versions == len(list_versions(MODEL_NAME))


@needs_model
def test_workers_follow_the_active_version_in_the_database(client):
    versions = list_versions(MODEL_NAME)
    if len(versions) < 2:
        pytest.skip("needs two registered versions")
    serving = core.get_scorer().version
    other = next(v for v in versions if v != serving)

    def mark_active(version):
        # What a rollback handled by another worker leaves behind.
        with database.SessionLocal() as session:
            session.execute(update(database.ModelVersion).values(
                is_active=database.ModelVersion.version == version))
            session.commit()
        core.state.checked_at = 0.0          # skip the 10-second wait

    try:
        mark_active(other)
        client.get("/model-info")
        assert core.get_scorer().version == other
    finally:
        mark_active(serving)
        client.get("/model-info")
    assert core.get_scorer().version == serving


def test_platform_postgres_addresses_get_the_psycopg_driver():
    assert database.with_driver("postgres://u:p@host:5432/db") == "postgresql+psycopg://u:p@host:5432/db"
    assert database.with_driver("postgresql://u:p@host/db") == "postgresql+psycopg://u:p@host/db"
    assert database.with_driver("postgresql+psycopg://u:p@host/db") == "postgresql+psycopg://u:p@host/db"
    assert database.with_driver("sqlite:///x.db") == "sqlite:///x.db"
