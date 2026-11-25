from __future__ import annotations

import argparse
import ast
import warnings
from pathlib import Path

from .train_one import train_one


def parse_overrides(items: list[str]) -> dict:
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            out[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k] = v
    return out


def run(algos: list[str], default_algo: str, description: str) -> None:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--algo", choices=algos, default=default_algo)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=100_000, help="environment steps")
    ap.add_argument("--name", default="manual")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="hyperparameter overrides, e.g. lr=1e-3 gamma=0.95")
    ap.add_argument("--out", default="logs")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    cfg = dict(algo=a.algo, name=a.name, seed=a.seed, total_steps=a.steps, **parse_overrides(a.set))
    run_dir = Path(a.out) / a.algo / f"{a.name}_seed{a.seed}"
    res = train_one(cfg, run_dir)
    print(f"{res['status']}: best validation return {res.get('best_val_return', float('nan')):.3f} "
          f"in {res['train_seconds']} s. Checkpoints and CSV logs in {run_dir}")
    if res["status"] != "ok":
        raise SystemExit(res.get("error"))
