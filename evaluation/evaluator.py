"""Common evaluation interface. Any object with reset() and act(obs) -> int can be scored,
so learned checkpoints, the random policy and the rule policy share exactly one code path."""
from __future__ import annotations

import numpy as np
import pandas as pd

from environment import ACTION_NAMES, N_ACTIONS, AgriResilienceEnv


def run_episode(env: AgriResilienceEnv, policy, seed: int, scenario=None) -> dict:
    obs, _ = env.reset(seed=seed, options={"episode_seed": seed, "scenario": scenario})
    policy.reset()
    done, steps = False, 0
    counts = np.zeros(N_ACTIONS, dtype=int)
    while not done:
        a = int(policy.act(obs))
        obs, _, done, _, info = env.step(a)
        counts[info["executed_action"]] += 1
        steps += 1
    out = dict(env.episode_summary())
    out.update(seed=seed, steps=steps, scenario=env.rnd.scenario.name)
    for i, n in enumerate(ACTION_NAMES):
        out[f"n_{n}"] = int(counts[i])
    return out


def evaluate_policy(policy, seeds, scenario=None, env: AgriResilienceEnv | None = None) -> pd.DataFrame:
    env = env or AgriResilienceEnv()
    return pd.DataFrame([run_episode(env, policy, s, scenario) for s in seeds])
