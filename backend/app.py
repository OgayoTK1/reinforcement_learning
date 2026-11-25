"""FastAPI server: REST for catalogues, WebSocket for live playback of real Gymnasium episodes.

    uv run main.py            # serves the viewer on http://127.0.0.1:8000
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from environment import OBS_NAMES, spec_hash
from environment.params import ENV_VERSION
from environment.rendering import SCHEMA_VERSION

from .session import ROOT, Session, catalogue, list_policies

log = logging.getLogger("agri.backend")
FRONTEND = ROOT / "frontend"

app = FastAPI(title="AgriResilience RL", version=ENV_VERSION)


@app.get("/api/health")
def health():
    return {"status": "ok", "env_version": ENV_VERSION, "schema_version": SCHEMA_VERSION, "spec_hash": spec_hash()}


@app.get("/api/catalogue")
def get_catalogue():
    return catalogue()


@app.get("/api/policies")
def get_policies():
    return list_policies()


@app.get("/api/observation_spec")
def observation_spec():
    return {"names": OBS_NAMES, "spec_hash": spec_hash()}


@app.post("/api/episode")
def run_episode(body: dict):
    """Run one whole episode server-side and return the per-week states (used by tests and offline replay)."""
    try:
        s = Session(body.get("policy", "rule_based"), body.get("scenario"), int(body.get("seed", 1000)))
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(400, str(e))
    states = [s.snapshot()]
    while not s.done:
        states.append(s.step())
    return {"states": states, "summary": s.env.episode_summary()}


@app.websocket("/ws")
async def ws(sock: WebSocket):
    """Protocol (JSON text frames). See docs/API.md.

    client -> server  start {policy, scenario, seed, speed} | step {action?} | pause | resume | reset | speed {value}
    server -> client  hello | state {data} | episode_end {summary, reason} | error {message}
    """
    await sock.accept()
    await sock.send_json({"type": "hello", "env_version": ENV_VERSION, "schema_version": SCHEMA_VERSION,
                          "spec_hash": spec_hash()})
    sess: Session | None = None
    playing, speed = False, 1.0
    send_lock = asyncio.Lock()

    async def send(msg):
        async with send_lock:
            await sock.send_json(msg)

    async def emit(snap):
        await send({"type": "state", "data": snap})
        if snap.get("done"):
            await send({"type": "episode_end", "summary": snap["metrics"], "reason": snap.get("termination_reason")})

    async def player():
        nonlocal playing
        while sess is not None and playing and not sess.done and sess.policy is not None:
            await emit(sess.step())
            await asyncio.sleep(max(0.05, 0.9 / max(0.1, speed)))
        playing = False

    task: asyncio.Task | None = None
    try:
        while True:
            msg = json.loads(await sock.receive_text())
            kind = msg.get("type")
            try:
                if kind == "start":
                    if task:
                        playing = False
                        task.cancel()
                    sess = Session(msg.get("policy", "rule_based"), msg.get("scenario"), int(msg.get("seed", 1000)))
                    speed = float(msg.get("speed", 1.0))
                    await emit(sess.snapshot())
                    playing = sess.policy is not None and msg.get("autoplay", True)
                    if playing:
                        task = asyncio.create_task(player())
                elif sess is None:
                    await send({"type": "error", "message": "send start first"})
                elif kind == "step":
                    await emit(sess.step(msg.get("action")))
                elif kind == "pause":
                    playing = False
                elif kind == "resume":
                    if sess.policy is not None and not sess.done and not playing:
                        playing = True
                        task = asyncio.create_task(player())
                elif kind == "reset":
                    playing = False
                    await emit(sess.reset())
                elif kind == "speed":
                    speed = float(msg["value"])
                else:
                    await send({"type": "error", "message": f"unknown message type {kind!r}"})
            except (ValueError, FileNotFoundError, KeyError) as e:
                await send({"type": "error", "message": str(e)})
    except WebSocketDisconnect:
        pass
    finally:
        playing = False
        if task:
            task.cancel()


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")


app.mount("/", StaticFiles(directory=FRONTEND), name="frontend")
