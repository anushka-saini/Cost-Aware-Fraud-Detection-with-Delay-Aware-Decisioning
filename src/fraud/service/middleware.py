"""Structured JSON logging with request IDs, and a simple rate limiter."""

import json
import logging
import time
import uuid
from collections import defaultdict, deque

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from fraud.service.core import request_id_var

logger = logging.getLogger("fraud.api")


class JsonFormatter(logging.Formatter):
    def format(self, record):
        entry = {"time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"), "level": record.levelname,
                 "logger": record.name, "message": record.getMessage(), "request_id": request_id_var.get()}
        entry.update(getattr(record, "fields", {}))
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry)


def configure_logging():
    if any(isinstance(h.formatter, JsonFormatter) for h in logger.handlers):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Give every request an ID (taken from X-Request-ID when supplied), echo
    it on the response and write one structured log line per request."""

    async def dispatch(self, request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        token = request_id_var.set(request_id)
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("unhandled error", extra={"fields": {"method": request.method, "path": request.url.path}})
            response = JSONResponse(status_code=500, content={"detail": "Internal server error", "request_id": request_id})
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        response.headers["X-Request-ID"] = request_id
        logger.info("request", extra={"fields": {"method": request.method, "path": request.url.path,
                                                 "status": response.status_code, "duration_ms": duration_ms}})
        request_id_var.reset(token)
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding one-minute window per client address, held in memory.

    Enough for a single process. Behind several workers this belongs in a
    shared store (Redis) or at the gateway.
    """

    def __init__(self, app, per_minute: int):
        super().__init__(app)
        self.per_minute = per_minute
        self.hits: dict[str, deque] = defaultdict(deque)

    async def dispatch(self, request, call_next):
        if self.per_minute <= 0 or request.url.path == "/health":
            return await call_next(request)
        client = request.client.host if request.client else "unknown"
        window, current = self.hits[client], time.monotonic()
        while window and current - window[0] > 60:
            window.popleft()
        if len(window) >= self.per_minute:
            retry = max(1, int(60 - (current - window[0])))
            return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"},
                                headers={"Retry-After": str(retry)})
        window.append(current)
        return await call_next(request)
