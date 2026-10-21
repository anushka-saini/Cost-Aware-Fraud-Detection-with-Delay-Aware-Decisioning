"""End-to-end smoke test against a real server process.

Starts the service on a scratch database, then walks the demo path the way
the dashboard does: log in, open the live feed over WebSocket, check that
REVIEW-tier transactions became cases, decide one, and read the overview,
drift, simulator, registry and audit endpoints.

    python scripts/smoke_test.py
"""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
import websockets

PORT = 8766
BASE = f"http://127.0.0.1:{PORT}"
N_STREAM = 400


async def read_stream(token):
    events = []
    async with websockets.connect(f"ws://127.0.0.1:{PORT}/stream?token={token}&speed=200&limit={N_STREAM}") as ws:
        async for message in ws:
            events.append(json.loads(message))
    return events


def main():
    workdir = Path(tempfile.mkdtemp(prefix="fraud-smoke-"))
    env = {**os.environ, "FRAUD_DATABASE_URL": f"sqlite:///{(workdir / 'smoke.db').as_posix()}", "PYTHONWARNINGS": "ignore"}
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
            print("health:", client.get("/health").json())

            login = client.post("/auth/login", json={"username": "analyst", "password": "analyst123"}).json()
            headers = {"Authorization": f"Bearer {login['access_token']}"}

            events = asyncio.run(read_stream(login["access_token"]))
            scored = [e for e in events if e["event"] == "transaction"]
            tiers = {t: sum(e["tier"] == t for e in scored) for t in ("ALLOW", "REVIEW", "BLOCK")}
            fraud = [e for e in scored if e["actual_fraud"] == 1]
            print(f"stream: {events[0]} ... {len(scored)} transactions, tiers {tiers}, "
                  f"{len(fraud)} real frauds of which {sum(e['tier'] != 'ALLOW' for e in fraud)} flagged")
            assert len(scored) == N_STREAM and events[-1]["event"] == "end"

            cases = client.get("/cases", params={"sort": "risk"}, headers=headers).json()
            print(f"review queue: {cases['total']} open cases")
            assert cases["total"] == tiers["REVIEW"]
            if cases["cases"]:
                top = cases["cases"][0]
                detail = client.get(f"/cases/{top['id']}", headers=headers).json()
                print(f"  case {top['id']}: amount {top['amount']:,.2f}, probability {top['fraud_probability']:.4f}, "
                      f"reasons {detail['reasons']}, {len(detail['recipient_history'])} earlier transactions to the recipient")
                done = client.post(f"/cases/{top['id']}/decision", headers=headers,
                                   json={"action": "decline", "note": "smoke test: confirmed fraud"}).json()
                print("  decision:", done)

            overview = client.get("/overview", headers=headers).json()["kpis"]
            print("overview:", {k: overview[k] for k in ("scored_total", "review_queue", "block_rate", "drift_status")})
            drift = client.get("/monitoring/drift", headers=headers).json()
            print("drift:", drift["status"], [r["period"] for r in drift["reports"]])
            simulator = client.get("/simulator/scores", params={"split": "held_out"}, headers=headers).json()
            print("simulator:", simulator["n_fraud"], "frauds,", len(simulator["legit_histogram"]), "legit bins")
            print("models:", [(m["version"], m["is_active"]) for m in client.get("/models", headers=headers).json()["versions"]])
            print("audit entries:", client.get("/audit", headers=headers).json()["total"])
            print("labels:", len(client.get("/labels", headers=headers).json()["rows"]))
        print("\nSMOKE TEST PASSED")
    finally:
        server.terminate()
        server.wait(timeout=20)


if __name__ == "__main__":
    main()
