"""Application factory.

    uvicorn fraud.service.app:app
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from fraud import __version__
from fraud.service import db as database
from fraud.service.core import get_scorer, request_id_var, sync_model_versions
from fraud.service.middleware import RateLimitMiddleware, RequestContextMiddleware, configure_logging, logger
from fraud.service.routes import router
from fraud.service.security import ensure_default_users
from fraud.service.seed import import_drift_reports, seed_history
from fraud.service.settings import DEV_SECRET, get_service_settings
from fraud.service.stream import router as stream_router


def startup():
    """Create missing tables, the demo users and the model-version records."""
    database.init_db()
    with database.SessionLocal() as session:
        ensure_default_users(session)
        sync_model_versions(session)
        seeded, reports = seed_history(session), import_drift_reports(session)
    settings = get_service_settings()
    if settings.jwt_secret == DEV_SECRET:
        logger.warning("FRAUD_JWT_SECRET is not set: using the development secret. Set it before any real use.")
    scorer = get_scorer()
    logger.info("service started", extra={"fields": {"model": scorer.name, "version": scorer.version,
                                                     "history_rows_seeded": seeded, "drift_reports_imported": reports}})


@asynccontextmanager
async def lifespan(app: FastAPI):
    startup()
    yield


def create_app() -> FastAPI:
    settings = get_service_settings()
    configure_logging()
    app = FastAPI(title="Fraud Operations API", version=__version__, lifespan=lifespan,
                  description="Scoring, review queue, monitoring and model registry for the cost-aware fraud detection project.")

    app.add_middleware(RateLimitMiddleware, per_minute=settings.rate_limit_per_minute)
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origin_list, allow_credentials=True,
                       allow_methods=["*"], allow_headers=["*"], expose_headers=["X-Request-ID"])
    app.add_middleware(RequestContextMiddleware)      # outermost: every response gets a request ID

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        return JSONResponse(status_code=exc.status_code, headers=getattr(exc, "headers", None),
                            content={"detail": exc.detail, "request_id": request_id_var.get()})

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        errors = [{"field": ".".join(str(p) for p in e["loc"][1:]), "message": e["msg"]} for e in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": "Invalid request", "errors": errors,
                                                      "request_id": request_id_var.get()})

    app.include_router(router)
    app.include_router(stream_router)
    return app


app = create_app()
