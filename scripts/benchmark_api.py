"""Latency benchmark for the scoring endpoint.

Starts the real service (uvicorn, one worker) on a scratch SQLite database
seeded with the demo history, then sends held-out transactions from
api/data/replay.csv to POST /score/raw:

  * sequentially, to measure per-request latency (p50 / p95 / p99), split into
    requests that were allowed and requests that were flagged (flagged ones
    also compute SHAP reasons);
  * from several concurrent clients, to measure throughput under load.

Each request does the full job: feature-store lookup, feature computation,
5-model ensemble, calibration, anomaly score, tiering, and database writes.

    python scripts/benchmark_api.py
"""

import os
import platform
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np

from fraud.results import record
from fraud.service.stream import load_replay

PORT = 8765
BASE = f"http://127.0.0.1:{PORT}"
N_SEQUENTIAL, N_CONCURRENT, CLIENTS = 600, 800, 8


def percentiles(values_ms):
    values = np.asarray(values_ms)
    return {"n": int(len(values)), "p50_ms": float(np.percentile(values, 50)), "p95_ms": float(np.percentile(values, 95)),
            "p99_ms": float(np.percentile(values, 99)), "mean_ms": float(values.mean()), "max_ms": float(values.max())}


def post(client, row):
    payload = {k: v for k, v in row.items() if k != "actual_fraud"}
    start = time.perf_counter()
    response = client.post("/score/raw", json=payload)
    elapsed = (time.perf_counter() - start) * 1000
    response.raise_for_status()
    return elapsed, response.json()["tier"]


def main():
    rows = load_replay()
    if not rows:
        raise SystemExit("api/data/replay.csv not found: run scripts/build_demo_data.py first")
    workdir = Path(tempfile.mkdtemp(prefix="fraud-bench-"))
    env = {**os.environ, "FRAUD_DATABASE_URL": f"sqlite:///{(workdir / 'bench.db').as_posix()}",
           "FRAUD_RATE_LIMIT_PER_MINUTE": "0", "PYTHONWARNINGS": "ignore"}
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "fraud.service.app:app", "--port", str(PORT),
                               "--log-level", "warning"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        with httpx.Client(base_url=BASE, timeout=60) as client:
            for _ in range(120):
                try:
                    if client.get("/health").status_code == 200:
                        break
                except httpx.HTTPError:
                    time.sleep(0.5)
            else:
                raise SystemExit("service did not start")
            for row in rows[:30]:                       # warm-up (first SHAP call builds the explainers)
                post(client, row)

            sequential = [post(client, row) for row in rows[30:30 + N_SEQUENTIAL]]
            allowed = [ms for ms, tier in sequential if tier == "ALLOW"]
            flagged = [ms for ms, tier in sequential if tier != "ALLOW"]

        load_rows = rows[30 + N_SEQUENTIAL:30 + N_SEQUENTIAL + N_CONCURRENT]

        def worker(chunk):
            with httpx.Client(base_url=BASE, timeout=120) as c:
                return [post(c, row)[0] for row in chunk]

        started = time.perf_counter()
        with ThreadPoolExecutor(CLIENTS) as pool:
            concurrent = [ms for chunk in pool.map(worker, [load_rows[i::CLIENTS] for i in range(CLIENTS)]) for ms in chunk]
        wall = time.perf_counter() - started
    finally:
        server.terminate()
        server.wait(timeout=20)

    payload = {
        "endpoint": "POST /score/raw",
        "what_a_request_does": "feature-store lookup, online features, 5-model LightGBM ensemble, Platt calibration, "
                               "autoencoder anomaly score, tiering, SHAP reasons when flagged, database writes",
        "setup": {"server": "uvicorn, 1 worker", "database": "SQLite file, seeded with the demo recipient history",
                  "machine": platform.processor() or platform.machine(), "cpu_count": os.cpu_count(),
                  "python": platform.python_version()},
        "sequential": {"all": percentiles([ms for ms, _ in sequential]), "allowed_no_shap": percentiles(allowed),
                       "flagged_with_shap": percentiles(flagged) if flagged else None},
        "concurrent": {"clients": CLIENTS, **percentiles(concurrent), "requests_per_second": len(concurrent) / wall},
    }
    record("phase7_latency", payload, "scripts/benchmark_api.py")
    for name, block in (("sequential, all", payload["sequential"]["all"]),
                        ("sequential, allowed (no SHAP)", payload["sequential"]["allowed_no_shap"]),
                        ("sequential, flagged (with SHAP)", payload["sequential"]["flagged_with_shap"]),
                        (f"{CLIENTS} concurrent clients", payload["concurrent"])):
        if block:
            print(f"{name:34s} n={block['n']:4d}  p50 {block['p50_ms']:.1f} ms  p95 {block['p95_ms']:.1f} ms  p99 {block['p99_ms']:.1f} ms")
    print(f"throughput under load: {payload['concurrent']['requests_per_second']:.1f} requests/s")


if __name__ == "__main__":
    main()
