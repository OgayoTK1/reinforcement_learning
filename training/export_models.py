"""Copy the best checkpoint of each algorithm into models/ for the viewer and play.py.

The configuration is chosen by mean validation return over seeds, then the seed with the highest validation
return supplies the weights. Run after experiment_runner (and before or after final_eval).

    uv run python -m training.export_models --results experiments/results
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from evaluation.final_eval import select_configs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="experiments/results")
    ap.add_argument("--models", default="models")
    a = ap.parse_args()
    reg = pd.read_csv(Path(a.results) / "registry.csv")
    sel = select_configs(reg)
    for _, row in sel.iterrows():
        runs = reg[(reg.algo == row.algo) & (reg["name"] == row["name"]) & (reg.status == "ok")]
        best = runs.sort_values("best_val_return", ascending=False).iloc[0]
        sub = "dqn" if row.algo == "dqn" else "pg"
        dst = Path(a.models) / sub
        dst.mkdir(parents=True, exist_ok=True)
        stem = "best" if row.algo == "dqn" else row.algo
        shutil.copy(Path(best.run_dir) / "best.pt", dst / f"{stem}.pt")
        meta = json.loads((Path(best.run_dir) / "best.json").read_text())
        meta["exported_from"] = f"{row.algo}/{row['name']}/seed{int(best.seed)}"
        (dst / f"{stem}.json").write_text(json.dumps(meta, indent=2))
        print(f"{row.algo:9s} -> {dst / (stem + '.pt')}  ({meta['exported_from']}, val {best.best_val_return:.3f})")


if __name__ == "__main__":
    main()
