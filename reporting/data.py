"""Loading experiment outputs. Everything the figures and the report show is read from here; nothing is typed in."""
from __future__ import annotations

import importlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ALGO_LABEL = {"dqn": "DQN", "reinforce": "REINFORCE", "ppo": "PPO", "a2c": "A2C", "rule_based": "Rule-based", "random": "Random"}
COLOR = {"dqn": "#1f5f8b", "reinforce": "#b5651d", "ppo": "#2f7d4f", "a2c": "#8a3b6a", "rule_based": "#4a4a4a", "random": "#9a9a9a"}
ALGOS = ["dqn", "reinforce", "ppo", "a2c"]


class Results:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.registry = pd.read_csv(self.root / "registry.csv")
        self.ok = self.registry[self.registry.status == "ok"].copy()
        p = self.root / "selected_configs.csv"
        self.selected = pd.read_csv(p) if p.exists() else None
        p = self.root / "test_episodes.csv"
        self.test = pd.read_csv(p) if p.exists() else None

    def runs(self, algo: str, name: str | None = None) -> pd.DataFrame:
        d = self.ok[self.ok.algo == algo]
        return d if name is None else d[d["name"] == name]

    def table(self, run_dir: str, which: str) -> pd.DataFrame:
        p = Path(run_dir) / f"{which}.csv"
        return pd.read_csv(p) if p.exists() else pd.DataFrame()

    def best_name(self, algo: str) -> str | None:
        if self.selected is None:
            g = self.ok[self.ok.algo == algo].groupby("name")["best_val_return"].mean()
            return g.idxmax() if len(g) else None
        r = self.selected[self.selected.algo == algo]
        return r["name"].iloc[0] if len(r) else None

    def n_seeds(self) -> int:
        return int(self.ok.groupby(["algo", "name"]).seed.nunique().max())

    def steps(self) -> int:
        return int(self.ok.total_steps.max())


def defaults(algo: str) -> dict:
    return dict(importlib.import_module(f"agents.{algo}").DEFAULTS)


def full_config(algo: str, name: str) -> dict:
    from experiments.matrix import _M
    ov = next(o for n, o, _ in _M[algo] if n == name)
    return {**defaults(algo), **ov}


def smooth(y: np.ndarray, w: int) -> np.ndarray:
    y = np.asarray(y, float)
    if len(y) < 2:
        return y
    w = max(1, min(w, len(y)))
    k = np.ones(w) / w
    pad = np.concatenate([np.full(w - 1, y[:w].mean()), y])
    return np.convolve(pad, k, mode="valid")


def on_grid(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.interp(grid, x, y, left=np.nan, right=np.nan)


def convergence_episode(returns: np.ndarray, window: int = 50, frac: float = 0.9) -> float:
    """First episode after which the smoothed training return stays above start + frac*(final-start).
    NaN when the run never improved by more than 0.05 return units."""
    s = smooth(returns, window)
    if len(s) < 3 * window:
        return float("nan")
    start, final = s[: max(5, len(s) // 20)].mean(), s[-max(5, len(s) // 10):].mean()
    if final - start < 0.05:
        return float("nan")
    thr = start + frac * (final - start)
    below = np.where(s < thr)[0]
    return float(below[-1] + 1) if len(below) else 0.0
