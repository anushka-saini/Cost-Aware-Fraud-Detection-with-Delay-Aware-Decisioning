"""Application factory.

    uvicorn fraud.service.app:app
"""

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from fraud import __version__
from fraud.service.core import follow_active_version, get_scorer, request_id_var
from fraud.service.demo import reset_demo_if_due
from fraud.service.middleware import (
    RateLimitMiddleware,
    RequestContextMiddleware,
    RequestSizeLimitMiddleware,
    SecurityHeadersMiddleware,
    configure_logging,
    logger,
)
from fraud.service.routes import router
from fraud.service.settings import check_settings, get_service_settings
from fraud.service.setup import prepare_database
from fraud.service.stream import router as stream_router


def startup():
    """Check the settings, prepare the database (unless that was done before
    the workers started) and load the model."""
    settings = get_service_settings()
    for problem in check_settings(settings):        # raises in production
        logger.warning(problem)
    prepared = prepare_database() if settings.setup_on_start else {"prepared_by": "python -m fraud.service.setup"}
    scorer = get_scorer()
    logger.info("service started", extra={"fields": {"model": scorer.name, "version": scorer.version,
                                                     "environment": settings.environment,
                                                     "demo_mode": settings.demo_mode, **prepared}})


@asynccontextmanager
async def lifespan(app: FastAPI):
    startup()
    yield


def create_app() -> FastAPI:
    settings = get_service_settings()
    configure_logging()
    app = FastAPI(title="Fraud Operations API", version=__version__, lifespan=lifespan,
                  description="Scoring, review queue, monitoring and model registry for the cost-aware fraud detection project.")

    app.add_middleware(RateLimitMiddleware, per_minute=settings.rate_limit_per_minute,
                       login_per_minute=settings.login_attempts_per_minute)
    app.add_middleware(RequestSizeLimitMiddleware, max_bytes=settings.max_request_bytes)
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origin_list, allow_credentials=True,
                       allow_methods=["*"], allow_headers=["*"], expose_headers=["X-Request-ID"])
    app.add_middleware(SecurityHeadersMiddleware, https_only=settings.environment == "production")
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

    # With several workers, each one follows the active model version in the database.
    # In the public demo, visitors' work is cleared once a day.
    app.include_router(router, dependencies=[Depends(follow_active_version), Depends(reset_demo_if_due)])
    app.include_router(stream_router)
    return app


app = create_app()
