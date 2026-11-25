"""Train a single configuration. Used by the experiment runner and for quick manual runs."""
from __future__ import annotations

import importlib
import json
import time
import traceback
from pathlib import Path

from agents.common import RunLogger, write_json

ALGOS = {"dqn": "agents.dqn", "reinforce": "agents.reinforce", "ppo": "agents.ppo", "a2c": "agents.a2c"}


def train_one(cfg: dict, run_dir: Path) -> dict:
    import torch
    torch.set_num_threads(1)
    run_dir = Path(run_dir)
    log = RunLogger(run_dir)
    write_json(run_dir / "config.json", cfg)
    t0 = time.time()
    status, info = "ok", {}
    try:
        info = importlib.import_module(ALGOS[cfg["algo"]]).train(cfg, run_dir, log)
    except Exception as e:  # failures are recorded, never hidden
        status = "failed"
        info = {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]}
    log.flush()
    result = dict(cfg, status=status, train_seconds=round(time.time() - t0, 1), **info)
    write_json(run_dir / "result.json", result)
    return result
