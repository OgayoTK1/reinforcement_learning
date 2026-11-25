"""Held-out evaluation. Run once after the matrix has finished.

Model selection uses validation only: for each algorithm the configuration with the best mean validation
return (over seeds) is chosen, and only that configuration's best checkpoints touch the test episodes and the
out-of-distribution scenarios. Random and rule-based baselines are scored on exactly the same episodes.

    uv run python -m evaluation.final_eval --results experiments/results
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd

from agents.loader import load_policy
from environment import AgriResilienceEnv
from evaluation.baselines import ExtensionRulePolicy, RandomPolicy
from evaluation.evaluator import evaluate_policy
from training.splits import OOD_SCENARIOS, OOD_SEEDS, TEST_SEEDS

warnings.filterwarnings("ignore")


def select_configs(reg: pd.DataFrame) -> pd.DataFrame:
    ok = reg[reg["status"] == "ok"]
    g = ok.groupby(["algo", "name"])["best_val_return"].agg(["mean", "std", "count"]).reset_index()
    return g.sort_values("mean", ascending=False).groupby("algo").head(1).reset_index(drop=True)


def score(label: str, policy, meta: dict, env) -> pd.DataFrame:
    parts = []
    df = evaluate_policy(policy, TEST_SEEDS, env=env)
    df["split"] = "test"
    parts.append(df)
    for sc in OOD_SCENARIOS:
        d = evaluate_policy(policy, OOD_SEEDS, scenario=sc, env=env)
        d["split"] = f"ood:{sc}"
        parts.append(d)
    out = pd.concat(parts, ignore_index=True)
    for k, v in meta.items():
        out[k] = v
    out["policy"] = label
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="experiments/results")
    a = ap.parse_args()
    root = Path(a.results)
    reg = pd.read_csv(root / "registry.csv")
    sel = select_configs(reg)
    sel.to_csv(root / "selected_configs.csv", index=False)
    print(sel.to_string(index=False), flush=True)
    env = AgriResilienceEnv()
    frames = []
    for _, row in sel.iterrows():
        runs = reg[(reg.algo == row.algo) & (reg["name"] == row["name"]) & (reg.status == "ok")]
        for _, r in runs.iterrows():
            pol = load_policy(Path(r.run_dir) / "best.pt")
            print("scoring", row.algo, row["name"], "seed", r.seed, flush=True)
            frames.append(score(f"{row.algo}", pol, dict(algo=row.algo, config=row["name"], train_seed=int(r.seed)), env))
    for label, pol in [("random", RandomPolicy(0)), ("rule_based", ExtensionRulePolicy())]:
        print("scoring", label, flush=True)
        frames.append(score(label, pol, dict(algo=label, config=label, train_seed=0), env))
    allr = pd.concat(frames, ignore_index=True)
    allr.to_csv(root / "test_episodes.csv", index=False)
    cols = ["return", "harvested_kg", "marketable_bags", "net_return_kes", "water_used_m3", "water_use_efficiency_kg_m3",
            "post_harvest_loss_frac", "soil_change", "crop_failure", "food_reserve_met", "invalid_actions"]
    summ = allr.groupby(["policy", "split"])[cols].mean().reset_index()
    summ.to_csv(root / "test_summary.csv", index=False)
    print(summ[summ.split == "test"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
