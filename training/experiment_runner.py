"""Run the experiment matrix. Safe to interrupt: a finished run leaves result.json, and --resume skips it.

    uv run python -m training.experiment_runner --workers 4 --steps 100000 --seeds 0 1 2
    uv run python -m training.experiment_runner --algos ppo a2c --max-minutes 8     # chunked, resumable
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
import warnings
from pathlib import Path

import pandas as pd

from experiments.matrix import ALGOS, DEFAULT_STEPS, configs

RESULTS = Path("experiments/results")


def run_dir(cfg: dict, root: Path = RESULTS) -> Path:
    return root / "runs" / cfg["algo"] / cfg["name"] / f"seed{cfg['seed']}"


def _job(args):
    warnings.filterwarnings("ignore")
    import logging
    logging.getLogger("torchrl").setLevel(logging.ERROR)
    from training.train_one import train_one
    cfg, root = args
    return train_one(cfg, run_dir(cfg, Path(root)))


def is_done(cfg: dict, root: Path = RESULTS) -> bool:
    p = run_dir(cfg, root) / "result.json"
    if not p.exists():
        return False
    r = json.loads(p.read_text())
    return r.get("status") == "ok" and r.get("total_steps") == cfg["total_steps"]


def build_registry(root: Path = RESULTS) -> pd.DataFrame:
    rows = []
    for p in sorted((root / "runs").glob("*/*/seed*/result.json")):
        r = json.loads(p.read_text())
        r.pop("trace", None)
        r["run_dir"] = str(p.parent)
        rows.append(r)
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(root / "registry.csv", index=False)
        bad = df[df["status"] != "ok"]
        lines = ["# Failed runs", ""] + [f"- {r.algo}/{r['name']}/seed{r.seed}: {r.get('error', '')}" for _, r in bad.iterrows()]
        (root / "failures.md").write_text("\n".join(lines) + ("\n" if len(bad) else "\n(none)\n"))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algos", nargs="+", default=ALGOS, choices=ALGOS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) - 1))
    ap.add_argument("--max-minutes", type=float, default=None, help="stop starting new runs after this long")
    ap.add_argument("--out", default=str(RESULTS))
    ap.add_argument("--no-resume", action="store_true")
    a = ap.parse_args()
    root = Path(a.out)
    todo = [c for c in configs(a.algos, a.seeds, a.steps) if a.no_resume or not is_done(c, root)]
    total = len(configs(a.algos, a.seeds, a.steps))
    print(f"{total - len(todo)} of {total} runs already done, {len(todo)} to run on {a.workers} worker(s)", flush=True)
    t0 = time.time()
    # interleave algorithms and seeds so an interrupted batch leaves a balanced set
    todo.sort(key=lambda c: (c["seed"], c["name"] != "base", c["name"], c["algo"]))
    ctx = mp.get_context("spawn")
    done = 0
    with ctx.Pool(a.workers) as pool:
        it = pool.imap_unordered(_job, [(c, str(root)) for c in todo], chunksize=1)
        pending = len(todo)
        while pending:
            if a.max_minutes and time.time() - t0 > a.max_minutes * 60:
                print("time limit reached, stopping (rerun to resume)", flush=True)
                pool.terminate()
                break
            try:
                r = it.next(timeout=5)
            except mp.TimeoutError:
                continue
            except StopIteration:
                break
            pending -= 1; done += 1
            print(f"[{done}/{len(todo)}] {r['algo']}/{r['name']}/seed{r['seed']} {r['status']} "
                  f"val={r.get('best_val_return', float('nan')):.3f} {r['train_seconds']}s", flush=True)
            build_registry(root)
    df = build_registry(root)
    print(f"registry: {len(df)} runs, {int((df['status'] != 'ok').sum()) if len(df) else 0} failed", flush=True)


if __name__ == "__main__":
    main()
