"""Prepare the database before the API starts.

Creates any missing tables, the demo users and the model-version records,
then loads the demo history and the drift reports. Every step is safe to
repeat: running it twice changes nothing the second time.

With one server process this runs inside the app at startup. With several
workers it must run once, before they start, so they do not all try at the
same moment:

    python -m fraud.service.setup
"""

from fraud.service import db as database
from fraud.service.core import sync_model_versions
from fraud.service.middleware import configure_logging, logger
from fraud.service.security import ensure_default_users
from fraud.service.seed import import_drift_reports, seed_history


def prepare_database() -> dict:
    database.init_db()
    with database.SessionLocal() as session:
        ensure_default_users(session)
        sync_model_versions(session)
        return {"history_rows_seeded": seed_history(session),
                "drift_reports_imported": import_drift_reports(session)}


if __name__ == "__main__":
    configure_logging()
    logger.info("database prepared", extra={"fields": prepare_database()})
