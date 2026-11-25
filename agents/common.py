"""Pieces shared by all four learners: environment factory, network builder, seeding, run logging."""
from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchrl.envs import GymWrapper
from torchrl.modules import MLP

from environment import AgriResilienceEnv, N_ACTIONS, OBS_DIM

ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env(seed: int | None = None, scenario=None):
    """TorchRL view of AgriResilienceEnv. Actions are categorical integers."""
    env = GymWrapper(AgriResilienceEnv(scenario=scenario), categorical_action_encoding=True)
    if seed is not None:
        env.set_seed(seed)
    return env


def make_net(out_features: int, hidden: int, depth: int, activation: str, last_gain: float = 1.0) -> MLP:
    net = MLP(in_features=OBS_DIM, out_features=out_features, num_cells=[hidden] * depth,
              activation_class=ACTIVATIONS[activation])
    linears = [m for m in net.modules() if isinstance(m, nn.Linear)]
    for m in linears:
        nn.init.orthogonal_(m.weight, gain=float(np.sqrt(2)))
        nn.init.zeros_(m.bias)
    nn.init.orthogonal_(linears[-1].weight, gain=last_gain)
    return net


def make_value_net(hidden: int, depth: int, activation: str) -> MLP:
    return make_net(1, hidden, depth, activation, last_gain=1.0)


class RunLogger:
    """Collects rows per table and writes plain CSV files into the run directory."""

    def __init__(self, run_dir: Path):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.tables: dict[str, list[dict]] = {}
        self.t0 = time.time()

    def log(self, table: str, **row) -> None:
        row["wall_s"] = round(time.time() - self.t0, 2)
        self.tables.setdefault(table, []).append(row)

    def flush(self) -> None:
        for name, rows in self.tables.items():
            if not rows:
                continue
            keys = list(dict.fromkeys(k for r in rows for k in r))
            with open(self.dir / f"{name}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(rows)


def write_json(path: Path, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
