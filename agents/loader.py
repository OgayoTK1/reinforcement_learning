"""Checkpoint saving and loading. One code path serves play.py, the evaluator and the backend.

A checkpoint is <tag>.pt (network weights only) plus <tag>.json (architecture, training
config, observation-spec hash). Loading refuses a checkpoint whose observation spec does not
match the current environment.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from environment import N_ACTIONS, OBS_DIM, spec_hash
from environment.params import ENV_VERSION

from .common import make_net


class CheckpointPolicy:
    """Greedy (argmax) policy over network outputs. For DQN the outputs are Q-values, for the
    policy-gradient agents they are action logits. Evaluation is always deterministic."""

    def __init__(self, net: torch.nn.Module, algo: str = "", meta: dict | None = None):
        self.net = net.eval()
        self.name = algo
        self.meta = meta or {}

    def reset(self) -> None:
        pass

    @torch.no_grad()
    def scores(self, obs: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
        return self.net(x)[0].numpy()

    def act(self, obs: np.ndarray) -> int:
        return int(np.argmax(self.scores(obs)))


def save_checkpoint(run_dir: Path, tag: str, net: torch.nn.Module, meta: dict) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = dict(meta, obs_dim=OBS_DIM, n_actions=N_ACTIONS, spec_hash=spec_hash(), env_version=ENV_VERSION)
    torch.save({k: v.detach().cpu() for k, v in net.state_dict().items()}, run_dir / f"{tag}.pt")
    with open(run_dir / f"{tag}.json", "w") as f:
        json.dump(meta, f, indent=2, default=float)
    return run_dir / f"{tag}.pt"


def load_policy(path: str | Path) -> CheckpointPolicy:
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    if meta["spec_hash"] != spec_hash() or meta["obs_dim"] != OBS_DIM or meta["n_actions"] != N_ACTIONS:
        raise ValueError(f"{path.name} was trained on a different observation/action spec "
                         f"({meta['spec_hash']} vs {spec_hash()}); retrain or check out the matching code")
    net = make_net(N_ACTIONS, meta["hidden"], meta["depth"], meta["activation"])
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return CheckpointPolicy(net, meta.get("algo", ""), meta)
