"""What the exported learners attempt when an action is impossible.

The test summary counts impossible actions but not which ones. This replays each exported checkpoint
(models/) on the test seasons and records, for every invalid attempt, the action, the reason the environment
gave, and whether the crop was in the field. The result is cached as a CSV next to the report.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from environment import ACTION_NAMES, AgriResilienceEnv
from training.splits import TEST_SEEDS

CHECKPOINTS = {"dqn": "models/dqn/best.pt", "reinforce": "models/pg/reinforce.pt", "ppo": "models/pg/ppo.pt", "a2c": "models/pg/a2c.pt"}


def _profile(algo: str, path: str, seeds) -> list[dict]:
    from agents.loader import load_policy

    pol, env, rows = load_policy(path), AgriResilienceEnv(), []
    for seed in seeds:
        obs, _ = env.reset(seed=seed, options={"episode_seed": seed})
        pol.reset()
        done = False
        while not done:
            a = int(pol.act(obs))
            s = env.state
            phase = "before sowing" if not s.sown else ("after harvest" if s.harvested_frac >= 0.99 else "crop in field")
            obs, _, done, _, info = env.step(a)
            rows.append({"algo": algo, "seed": seed, "action": ACTION_NAMES[a], "valid": bool(info["valid"]),
                         "reason": info.get("reason", ""), "phase": phase})
    return rows


def invalid_profile(out_dir: Path, rebuild: bool = False) -> pd.DataFrame:
    """One row per step of every exported learner on the test seasons. Cached in out_dir/action_attempts.csv."""
    p = out_dir / "action_attempts.csv"
    if p.exists() and not rebuild:
        return pd.read_csv(p)
    rows = []
    for algo, path in CHECKPOINTS.items():
        if Path(path).exists():
            rows += _profile(algo, path, TEST_SEEDS)
    df = pd.DataFrame(rows)
    df.to_csv(p, index=False)
    return df


def invalid_summary(df: pd.DataFrame) -> dict:
    """Per learner: invalid attempts and deliberate waits per season, the two most common invalid actions with
    their reasons, and the share of invalid attempts made while the crop is in the field."""
    out = {}
    for algo, g in df.groupby("algo"):
        n = g.seed.nunique()
        bad = g[~g.valid]
        top = bad.groupby(["action", "reason"]).size().sort_values(ascending=False).head(2) / n
        out[algo] = {"invalid": len(bad) / n, "wait": (g.action == "wait").sum() / n,
                     "top": [(a, r, v) for (a, r), v in top.items()],
                     "in_field": (bad.phase == "crop in field").mean() if len(bad) else 0.0}
    return out
