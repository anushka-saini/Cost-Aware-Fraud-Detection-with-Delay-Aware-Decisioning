# Build from the project root:
#
#     docker build -f api/Dockerfile -t fraud-api .
#
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# libgomp is LightGBM's OpenMP runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch first, so the dependency resolver does not pull the CUDA build.
RUN pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml ./
COPY src ./src
RUN pip install ".[postgres]"

# What the running service reads: the registered model bundles, the demo
# replay data, precomputed simulator scores, drift reports and result files.
COPY alembic.ini ./
COPY migrations ./migrations
COPY models/registry ./models/registry
COPY api ./api
COPY results ./results
COPY demo_sample.csv ./

# The package is installed, so point it back at /app for data and results.
ENV FRAUD_PROJECT_ROOT=/app \
    FRAUD_REPLAY_PATH=/app/api/data/replay.csv \
    FRAUD_REPLAY_HISTORY_PATH=/app/api/data/replay_history.csv \
    FRAUD_SIMULATOR_DIR=/app/results/simulator \
    FRAUD_DATABASE_URL=sqlite:////app/api/fraud.db

# Number of server processes. Keep 1 with SQLite (one writer at a time);
# docker-compose uses PostgreSQL and sets 2. Hosting platforms set PORT.
ENV WEB_CONCURRENCY=1 \
    PORT=8000 \
    FRAUD_SETUP_ON_START=false

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=5 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/health')"

# 1. Apply migrations. 2. Prepare the database once (users, model versions,
# demo history). 3. Start the workers. --proxy-headers makes the rate limiter
# see the visitor's address rather than the hosting platform's proxy.
CMD ["sh", "-c", "alembic upgrade head && python -m fraud.service.setup && uvicorn fraud.service.app:app --host 0.0.0.0 --port ${PORT} --workers ${WEB_CONCURRENCY} --proxy-headers --forwarded-allow-ips='*'"]
