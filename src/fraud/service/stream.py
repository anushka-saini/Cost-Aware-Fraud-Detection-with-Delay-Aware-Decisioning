"""Live feed: replay held-out transactions through the scoring path.

``WS /stream`` reads api/data/replay.csv (built by scripts/build_demo_data.py)
in time order, scores each transaction exactly as POST /score/raw would,
stores it, and pushes the result to the client. Cases created along the way
appear in the review queue.
"""

import asyncio
import csv
from pathlib import Path

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import func, select

from fraud.service.core import assessment_response, score_raw
from fraud.service.db import SessionLocal, Transaction
from fraud.service.security import ROLES, user_from_token
from fraud.service.settings import get_service_settings

router = APIRouter()
RAW_FLOATS = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]


def load_replay(path=None) -> list[dict]:
    path = Path(path or get_service_settings().replay_path)
    if not path.is_file():
        return []
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            row = {"step": int(r["step"]), "type": r["type"], "nameOrig": r["nameOrig"], "nameDest": r["nameDest"],
                   **{k: float(r[k]) for k in RAW_FLOATS}, "actual_fraud": int(r["actual_fraud"])}
            rows.append(row)
    return rows


def _score_one(row: dict) -> dict:
    txn = {k: v for k, v in row.items() if k != "actual_fraud"}
    with SessionLocal() as db:
        assessment = score_raw(db, txn, source="stream", true_label=row["actual_fraud"], actor="stream-replay")
        return {**assessment_response(assessment), "transaction_id": assessment.transaction_id, "amount": txn["amount"],
                "type": txn["type"], "step": txn["step"], "name_orig": txn["nameOrig"], "name_dest": txn["nameDest"],
                "actual_fraud": row["actual_fraud"], "created_at": assessment.created_at.isoformat()}


def _already_streamed() -> int:
    with SessionLocal() as db:
        return db.scalar(select(func.count()).select_from(Transaction).where(Transaction.source == "stream")) or 0


@router.websocket("/stream")
async def stream(websocket: WebSocket, token: str = "", speed: float = 5.0, limit: int = 0, restart: bool = False):
    """Query parameters: ``token`` (JWT), ``speed`` (transactions per second,
    0.2 to 200), ``limit`` (stop after this many; 0 = all), ``restart``
    (replay from the first row even if some were already streamed)."""
    await websocket.accept()
    try:
        with SessionLocal() as db:
            user = user_from_token(token, db)
        if user.role not in ROLES:
            raise ValueError("role")
    except Exception:
        await websocket.close(code=4401, reason="Not authenticated")
        return

    rows = load_replay()
    if not rows:
        await websocket.send_json({"event": "error", "detail": "No replay data. Run scripts/build_demo_data.py"})
        await websocket.close()
        return
    offset = 0 if restart else (await run_in_threadpool(_already_streamed)) % len(rows)
    speed = min(max(speed, 0.2), 200.0)
    await websocket.send_json({"event": "start", "total": len(rows), "offset": offset, "speed": speed})
    sent = 0
    try:
        for row in rows[offset:]:
            if limit and sent >= limit:
                break
            result = await run_in_threadpool(_score_one, row)
            await websocket.send_json({"event": "transaction", "position": offset + sent + 1, **result})
            sent += 1
            await asyncio.sleep(1.0 / speed)
        await websocket.send_json({"event": "end", "sent": sent})
        await websocket.close()
    except WebSocketDisconnect:
        return
