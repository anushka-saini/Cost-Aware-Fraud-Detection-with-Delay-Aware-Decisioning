import os

import pytest

# The service reads its configuration at import time, so point it at a
# throwaway in-memory database before anything from fraud.service is imported.
os.environ["FRAUD_DATABASE_URL"] = "sqlite:///:memory:"
os.environ["FRAUD_RATE_LIMIT_PER_MINUTE"] = "0"
os.environ["FRAUD_REPLAY_HISTORY_PATH"] = "none"       # tests start with an empty feature store
os.environ.setdefault("FRAUD_JWT_SECRET", "test-secret-not-for-production-use-0123456789")

from fraud.config import get_settings  # noqa: E402
from fraud.registry import list_versions  # noqa: E402

MODEL_NAME = os.environ.get("FRAUD_MODEL_NAME", "fraud_scorer")


def _have_model() -> bool:
    return bool(list_versions(MODEL_NAME))


def _have_data() -> bool:
    return get_settings().features.exists()


needs_model = pytest.mark.skipif(not _have_model(), reason=f"no registered model named {MODEL_NAME!r}")
needs_data = pytest.mark.skipif(not _have_data(), reason="PaySim feature table not on disk")


@pytest.fixture
def client():
    """A test client on a freshly created, empty database."""
    if not _have_model():
        pytest.skip(f"no registered model named {MODEL_NAME!r}")
    from fastapi.testclient import TestClient

    from fraud.service import db as database
    from fraud.service.app import app

    database.Base.metadata.drop_all(database.engine)
    with TestClient(app) as test_client:        # runs startup: tables, demo users, model versions
        yield test_client


def _login(client, username, password):
    response = client.post("/auth/login", json={"username": username, "password": password})
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


@pytest.fixture
def analyst(client):
    return _login(client, "analyst", "analyst123")


@pytest.fixture
def admin(client):
    return _login(client, "admin", "admin123")
