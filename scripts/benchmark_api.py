"""Latency benchmark for the scoring endpoint.

Starts the real service the way the container does (prepare the database
once, then start uvicorn with N worker processes) on a fresh database seeded
with the demo history, then sends held-out transactions from
api/data/replay.csv to POST /score/raw:

  * sequentially, to measure per-request latency (p50 / p95 / p99), split into
    requests that were allowed and requests that were flagged (flagged ones
    also compute SHAP reasons);
  * from several concurrent clients, to measure throughput under load.

Each request does the full job: feature-store lookup, feature computation,
5-model ensemble, calibration, tiering, SHAP reasons when flagged, and
database writes.

    python scripts/benchmark_api.py                      # 1 worker, a scratch SQLite file
    python scripts/benchmark_api.py --workers 4 --database-url postgresql://postgres@127.0.0.1:55432/bench

Several workers need PostgreSQL (SQLite allows one writer at a time). The
database must be empty: each run creates its own tables and demo history.

Results go to results/metrics.json, section "<model version>_latency" (for
example v3_latency), one entry per setup. The V2 timings stay in
phase7_latency.
"""

import argparse
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
import psutil

from fraud.results import load_metrics, record
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


def start_server(workers: int, database_url: str, log_path: Path):
    env = {**os.environ, "FRAUD_DATABASE_URL": database_url, "FRAUD_RATE_LIMIT_PER_MINUTE": "0",
           "FRAUD_SETUP_ON_START": "false", "PYTHONWARNINGS": "ignore"}
    # Prepare the database once, before the workers start (as the container does).
    subprocess.run([sys.executable, "-m", "fraud.service.setup"], env=env, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log = open(log_path, "w", encoding="utf-8")         # the server's errors, if any
    return subprocess.Popen([sys.executable, "-m", "uvicorn", "fraud.service.app:app", "--port", str(PORT),
                             "--workers", str(workers), "--log-level", "warning"],
                            env=env, stdout=log, stderr=log)


def wait_until_up(client) -> dict:
    for _ in range(240):
        try:
            response = client.get("/health")
            if response.status_code == 200:
                return response.json()["model"]
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    raise SystemExit("service did not start")


def send_all(chunk) -> list[float]:
    """One client sending its share of the rows, one after another."""
    with httpx.Client(base_url=BASE, timeout=120) as c:
        return [post(c, row)[0] for row in chunk]


def in_parallel(rows) -> list[float]:
    with ThreadPoolExecutor(CLIENTS) as pool:
        return [ms for chunk in pool.map(send_all, [rows[i::CLIENTS] for i in range(CLIENTS)]) for ms in chunk]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--database-url", default=None, help="default: a scratch SQLite file (1 worker only)")
    args = parser.parse_args()

    rows = load_replay()
    if not rows:
        raise SystemExit("api/data/replay.csv not found: run scripts/build_demo_data.py first")
    workdir = Path(tempfile.mkdtemp(prefix="fraud-bench-"))
    database_url = args.database_url or f"sqlite:///{(workdir / 'bench.db').as_posix()}"
    backend = "postgresql" if database_url.startswith("postgres") else "sqlite"
    if backend == "sqlite" and args.workers > 1:
        raise SystemExit("several workers need PostgreSQL: pass --database-url")

    server = start_server(args.workers, database_url, workdir / "server.log")
    print(f"server log: {workdir / 'server.log'}")
    try:
        with httpx.Client(base_url=BASE, timeout=60) as client:
            model = wait_until_up(client)
            # Warm-up, spread over the workers: the first SHAP call in each one builds its explainers.
            start = 30 * args.workers
            in_parallel(rows[:start])

            sequential = [post(client, row) for row in rows[start:start + N_SEQUENTIAL]]
            allowed = [ms for ms, tier in sequential if tier == "ALLOW"]
            flagged = [ms for ms, tier in sequential if tier != "ALLOW"]

        started = time.perf_counter()
        concurrent = in_parallel(rows[start + N_SEQUENTIAL:start + N_SEQUENTIAL + N_CONCURRENT])
        wall = time.perf_counter() - started
    finally:
        server.terminate()
        server.wait(timeout=30)

    setup_name = f"{args.workers}_worker{'s' if args.workers > 1 else ''}_{backend}"
    battery = psutil.sensors_battery()
    run = {
        "setup": {"server": f"uvicorn, {args.workers} worker{'s' if args.workers > 1 else ''}",
                  "database": f"{backend}, fresh, seeded with the demo recipient history",
                  "machine": platform.processor() or platform.machine(), "cpu_count": os.cpu_count(),
                  "python": platform.python_version(),
                  "power": "no battery" if battery is None else "plugged in" if battery.power_plugged else "on battery",
                  "note": "the client runs on the same laptop as the server and shares its CPU"},
        "sequential": {"all": percentiles([ms for ms, _ in sequential]), "allowed_no_shap": percentiles(allowed),
                       "flagged_with_shap": percentiles(flagged) if flagged else None},
        "concurrent": {"clients": CLIENTS, **percentiles(concurrent), "requests_per_second": len(concurrent) / wall},
    }
    section = f"{model['version']}_latency"
    existing = {k: v for k, v in load_metrics().get(section, {}).items() if not k.startswith("_")}
    runs = {**existing.get("runs", {}), setup_name: run}
    record(section, {"endpoint": "POST /score/raw", "model": model,
                     "what_a_request_does": "feature-store lookup, online features, 5-model ensemble, calibration, "
                                            "tiering, SHAP reasons when flagged, database writes",
                     "runs": runs}, "scripts/benchmark_api.py")

    print(f"{setup_name}, model {model['version']}")
    for name, block in (("sequential, all", run["sequential"]["all"]),
                        ("sequential, allowed (no SHAP)", run["sequential"]["allowed_no_shap"]),
                        ("sequential, flagged (with SHAP)", run["sequential"]["flagged_with_shap"]),
                        (f"{CLIENTS} concurrent clients", run["concurrent"])):
        if block:
            print(f"{name:34s} n={block['n']:4d}  p50 {block['p50_ms']:.1f} ms  p95 {block['p95_ms']:.1f} ms  p99 {block['p99_ms']:.1f} ms")
    print(f"throughput under load: {run['concurrent']['requests_per_second']:.1f} requests/s")


if __name__ == "__main__":
    main()
