"""Statistics on the held-out episodes. Comparisons are paired by episode seed (every policy sees the same
weather and prices), so they use the Wilcoxon signed-rank test on per-episode returns."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats

from .data import ALGOS


def bootstrap_ci(x: np.ndarray, n: int = 5000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def per_episode(test: pd.DataFrame, split: str, col: str = "return") -> pd.DataFrame:
    """policy x episode-seed matrix; learned policies are averaged over their training seeds (paired by episode)."""
    d = test[test.split == split]
    return d.groupby(["policy", "seed"])[col].mean().unstack(0)


def holm(p: list[float]) -> list[float]:
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return list(adj)


def paired_tests(test: pd.DataFrame, split: str = "test") -> pd.DataFrame:
    m = per_episode(test, split)
    pairs = [(a, "rule_based") for a in ALGOS if a in m] + list(itertools.combinations([a for a in ALGOS if a in m], 2))
    rows = []
    for a, b in pairs:
        d = (m[a] - m[b]).dropna()
        try:
            w = stats.wilcoxon(d) if np.any(d != 0) else None
        except ValueError:
            w = None
        rows.append(dict(a=a, b=b, mean_diff=float(d.mean()), ci_lo=bootstrap_ci(d.values)[0], ci_hi=bootstrap_ci(d.values)[1],
                         p=float(w.pvalue) if w else 1.0, effect_dz=float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else 0.0, n=len(d)))
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.p.tolist())
    return out


def summary_table(test: pd.DataFrame, split: str = "test") -> pd.DataFrame:
    cols = ["return", "harvested_kg", "marketable_bags", "net_return_kes", "water_used_m3", "water_use_efficiency_kg_m3",
            "post_harvest_loss_frac", "soil_change", "crop_failure", "food_reserve_met", "invalid_actions"]
    d = test[test.split == split]
    rows = []
    for pol, g in d.groupby("policy"):
        per = g.groupby("seed")["return"].mean()
        lo, hi = bootstrap_ci(per.values)
        row = {"policy": pol, "return_ci_lo": lo, "return_ci_hi": hi, "n_episodes": int(per.size)}
        row.update(g[cols].mean().to_dict())
        if "train_seed" in g:
            row["return_seed_std"] = float(g.groupby("train_seed")["return"].mean().std(ddof=1)) if g.train_seed.nunique() > 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)
