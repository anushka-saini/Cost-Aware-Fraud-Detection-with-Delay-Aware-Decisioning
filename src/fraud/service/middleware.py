"""Structured JSON logging with request IDs, a simple rate limiter, a request
size limit and standard security headers."""

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

    Login attempts get their own, much smaller limit (against password
    guessing). Each worker keeps its own counts, so with N workers a client
    can make up to N times the limit; a shared store (Redis) or the hosting
    platform's gateway would make it exact.
    """

    def __init__(self, app, per_minute: int, login_per_minute: int = 0):
        super().__init__(app)
        self.per_minute = per_minute
        self.login_per_minute = login_per_minute
        self.hits: dict[tuple, deque] = defaultdict(deque)

    def too_many(self, key, limit):
        """Record one request; return the seconds to wait if it is over the limit."""
        window, current = self.hits[key], time.monotonic()
        while window and current - window[0] > 60:
            window.popleft()
        if len(window) >= limit:
            return max(1, int(60 - (current - window[0])))
        window.append(current)
        return None

    async def dispatch(self, request, call_next):
        path = request.url.path
        if path in ("/health", "/status"):
            return await call_next(request)
        client = request.client.host if request.client else "unknown"
        checks = [("all", self.per_minute)]
        if path == "/auth/login" and request.method == "POST":
            checks.append(("login", self.login_per_minute))
        for kind, limit in checks:
            if limit <= 0:
                continue
            wait = self.too_many((client, kind), limit)
            if wait is not None:
                detail = "Too many login attempts" if kind == "login" else "Rate limit exceeded"
                return JSONResponse(status_code=429, content={"detail": detail}, headers={"Retry-After": str(wait)})
        return await call_next(request)


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Refuse request bodies larger than ``max_bytes`` (going by Content-Length)."""

    def __init__(self, app, max_bytes: int):
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(self, request, call_next):
        length = request.headers.get("content-length")
        if self.max_bytes > 0 and length and length.isdigit() and int(length) > self.max_bytes:
            return JSONResponse(status_code=413, content={"detail": "Request body too large"})
        return await call_next(request)


SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",               # browsers must not guess content types
    "X-Frame-Options": "DENY",                         # the API is never shown inside a frame
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add standard security headers to every response (and HSTS when served over HTTPS)."""

    def __init__(self, app, https_only: bool = False):
        super().__init__(app)
        self.headers = dict(SECURITY_HEADERS)
        if https_only:
            self.headers["Strict-Transport-Security"] = "max-age=31536000"

    async def dispatch(self, request, call_next):
        response = await call_next(request)
        for name, value in self.headers.items():
            response.headers.setdefault(name, value)
        return response
