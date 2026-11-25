"""The experiment matrix: 10 hyperparameter configurations per algorithm, each trained with several seeds.

Config 1 of every algorithm is the default. Configs 2 to 10 each change one factor (so the effect can be
read off against the default), and `factor` records which. Everything else is in agents/<algo>.py DEFAULTS.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

DEFAULT_STEPS = 100_000

_M = {
    "dqn": [
        ("base", {}, "default"),
        ("lr_1e-3", dict(lr=1e-3), "learning rate up"),
        ("lr_1e-4", dict(lr=1e-4), "learning rate down"),
        ("gamma_0.95", dict(gamma=0.95), "shorter horizon"),
        ("gamma_0.999", dict(gamma=0.999), "longer horizon"),
        ("buffer_5k", dict(buffer_size=5_000), "small replay buffer"),
        ("batch_256", dict(batch_size=256), "larger minibatch"),
        ("eps_fast", dict(eps_decay_frac=0.15), "fast exploration decay"),
        ("eps_slow", dict(eps_decay_frac=0.8, eps_end=0.1), "slow exploration decay, higher floor"),
        ("hard_noDouble", dict(target="hard", hard_every=500, double_dqn=False), "hard target copy, no double DQN"),
    ],
    "reinforce": [
        ("base", {}, "default"),
        ("lr_3e-3", dict(lr=3e-3), "learning rate up"),
        ("lr_3e-4", dict(lr=3e-4), "learning rate down"),
        ("gamma_0.95", dict(gamma=0.95), "shorter horizon"),
        ("gamma_0.999", dict(gamma=0.999), "longer horizon"),
        ("episodes_4", dict(episodes_per_update=4), "few episodes per update (noisy gradient)"),
        ("episodes_64", dict(episodes_per_update=64), "many episodes per update"),
        ("entropy_0.05", dict(ent_coef=0.05), "entropy bonus up"),
        ("no_baseline", dict(baseline="none", normalize=False), "raw returns, no baseline"),
        ("wide_256", dict(hidden=256), "wider network"),
    ],
    "ppo": [
        ("base", {}, "default"),
        ("lr_1e-3", dict(lr=1e-3), "learning rate up"),
        ("lr_1e-4", dict(lr=1e-4), "learning rate down"),
        ("gamma_0.95", dict(gamma=0.95), "shorter horizon"),
        ("clip_0.1", dict(clip_eps=0.1), "tighter clip"),
        ("clip_0.3", dict(clip_eps=0.3), "looser clip"),
        ("entropy_0.05", dict(ent_coef=0.05), "entropy bonus up"),
        ("epochs_15", dict(epochs=15), "more epochs per batch"),
        ("batch_512", dict(frames_per_batch=512, minibatch=64), "short rollouts, small minibatch"),
        ("lambda_0.8", dict(lmbda=0.8), "lower GAE lambda"),
    ],
    "a2c": [
        ("base", {}, "default"),
        ("lr_3e-3", dict(lr=3e-3), "learning rate up"),
        ("lr_1e-4", dict(lr=1e-4), "learning rate down"),
        ("gamma_0.95", dict(gamma=0.95), "shorter horizon"),
        ("gamma_0.999", dict(gamma=0.999), "longer horizon"),
        ("entropy_0.05", dict(ent_coef=0.05), "entropy bonus up"),
        ("entropy_0", dict(ent_coef=0.0), "no entropy bonus"),
        ("nstep_32", dict(frames_per_batch=32), "very short rollouts"),
        ("nstep_512", dict(frames_per_batch=512), "long rollouts"),
        ("lambda_0.8", dict(lmbda=0.8), "lower GAE lambda"),
    ],
}

ALGOS = list(_M)


def configs(algos=None, seeds=(0, 1, 2), steps: int = DEFAULT_STEPS) -> list[dict]:
    out = []
    for algo in algos or ALGOS:
        for name, ov, factor in _M[algo]:
            for seed in seeds:
                out.append(dict(algo=algo, name=name, factor=factor, seed=seed, total_steps=steps, **ov))
    return out


def write_yaml(dirpath: Path = Path("experiments/configs")) -> None:
    dirpath.mkdir(parents=True, exist_ok=True)
    for algo in ALGOS:
        rows = [dict(name=n, factor=f, **ov) for n, ov, f in _M[algo]]
        (dirpath / f"{algo}.yaml").write_text(yaml.safe_dump(rows, sort_keys=False))


if __name__ == "__main__":
    write_yaml()
    print(f"{sum(len(v) for v in _M.values())} configurations written to experiments/configs/ "
          f"({sys.argv[0]})")
