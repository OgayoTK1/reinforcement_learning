"""All report figures, built from experiment outputs only."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from environment import ACTION_NAMES

from .data import ALGO_LABEL, ALGOS, COLOR, Results, convergence_episode, on_grid, smooth

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6, "axes.titlesize": 10, "axes.titleweight": "bold",
    "figure.dpi": 100, "savefig.dpi": 200, "legend.frameon": False, "legend.fontsize": 8,
})


def _runs(R: Results, algo: str, name: str | None = None):
    name = name or R.best_name(algo)
    return R.runs(algo, name), name


def _band(ax, grid, curves, color, label, lw=1.6):
    A = np.vstack(curves)
    m = np.nanmean(A, axis=0)
    ax.plot(grid, m, color=color, lw=lw, label=label)
    if len(curves) > 1:
        s = np.nanstd(A, axis=0)
        ax.fill_between(grid, m - s, m + s, color=color, alpha=0.15, lw=0)


def _baseline_lines(R: Results, ax, col="return"):
    if R.test is None:
        return
    t = R.test[R.test.split == "test"].groupby("policy")[col].mean()
    for pol, ls in (("rule_based", "--"), ("random", ":")):
        if pol in t:
            ax.axhline(t[pol], color=COLOR[pol], ls=ls, lw=1.1, label=f"{ALGO_LABEL[pol]} (test)")


def learning_curves(R: Results, out: Path) -> Path:
    """Cumulative (episode) reward during training for each algorithm's selected configuration."""
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.6), sharey=True)
    for ax, algo in zip(axes.ravel(), ALGOS):
        runs, name = _runs(R, algo)
        n_ep = min((len(R.table(r.run_dir, "episodes")) for r in runs.itertuples()), default=0)
        if n_ep < 10:
            ax.set_title(f"{ALGO_LABEL[algo]}: no data"); continue
        grid = np.arange(1, n_ep + 1)
        curves, raw = [], []
        for r in runs.itertuples():
            e = R.table(r.run_dir, "episodes")["ret"].values[:n_ep]
            raw.append(e); curves.append(smooth(e, 40))
        ax.plot(grid, np.mean(raw, axis=0), color=COLOR[algo], alpha=0.12, lw=0.6)
        _band(ax, grid, curves, COLOR[algo], f"{ALGO_LABEL[algo]} ({name}), {len(curves)} seed(s)")
        _baseline_lines(R, ax)
        ax.set_title(ALGO_LABEL[algo]); ax.set_xlabel("training episode")
        ax.legend(loc="lower right")
    axes[0, 0].set_ylabel("episode return"); axes[1, 0].set_ylabel("episode return")
    fig.suptitle("Cumulative reward per training episode (smoothed over 40 episodes; band = std across seeds)", fontsize=9.5, y=0.995)
    fig.tight_layout(); p = out / "fig_learning_curves.png"; fig.savefig(p); plt.close(fig); return p


def evaluation_rewards(R: Results, out: Path) -> Path:
    """Greedy validation return during training (50 fixed validation episodes)."""
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for algo in ALGOS:
        runs, name = _runs(R, algo)
        tabs = [R.table(r.run_dir, "eval") for r in runs.itertuples()]
        tabs = [t for t in tabs if len(t)]
        if not tabs:
            continue
        grid = np.linspace(min(t.step.min() for t in tabs), min(t.step.max() for t in tabs), 30)
        _band(ax, grid, [on_grid(t.step.values, t.val_return.values, grid) for t in tabs], COLOR[algo], f"{ALGO_LABEL[algo]} ({name})")
    _baseline_lines(R, ax)
    ax.set_xlabel("environment steps"); ax.set_ylabel("validation return (greedy policy)")
    ax.set_title("Evaluation reward on held-out validation seasons"); ax.legend(ncol=2)
    fig.tight_layout(); p = out / "fig_eval_rewards.png"; fig.savefig(p); plt.close(fig); return p


def dqn_loss_and_entropy(R: Results, out: Path) -> Path:
    """Objective curves: DQN TD loss, policy entropy for the three policy-gradient methods."""
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.1))
    runs, name = _runs(R, "dqn")
    tabs = [R.table(r.run_dir, "updates") for r in runs.itertuples()]
    tabs = [t for t in tabs if len(t)]
    ax = axes[0]
    if tabs:
        grid = np.linspace(tabs[0].step.min(), min(t.step.max() for t in tabs), 120)
        _band(ax, grid, [on_grid(t.step.values, smooth(t.td_loss.values, 25), grid) for t in tabs], COLOR["dqn"], f"{name}")
        ax.set_yscale("log")
    ax.set_title("DQN TD loss (Huber)"); ax.set_xlabel("environment steps"); ax.legend()
    for ax, algo in zip(axes[1:], ["reinforce", "ppo", "a2c"]):
        runs, name = _runs(R, algo)
        tabs = [R.table(r.run_dir, "updates") for r in runs.itertuples()]
        tabs = [t for t in tabs if len(t) and "entropy" in t]
        if tabs:
            grid = np.linspace(tabs[0].step.min(), min(t.step.max() for t in tabs), 120)
            _band(ax, grid, [on_grid(t.step.values, smooth(t.entropy.values, 5), grid) for t in tabs], COLOR[algo], f"{name}")
        ax.axhline(np.log(20), color="#999", ls=":", lw=1); ax.text(0.02, 0.04, "uniform over 20 actions", transform=ax.transAxes, fontsize=7, color="#777")
        ax.set_title(f"{ALGO_LABEL[algo]} policy entropy"); ax.set_xlabel("environment steps"); ax.legend()
    fig.tight_layout(); p = out / "fig_objective_curves.png"; fig.savefig(p); plt.close(fig); return p


def convergence_and_stability(R: Results, out: Path) -> tuple[Path, pd.DataFrame]:
    rows = []
    for algo in ALGOS:
        runs, name = _runs(R, algo)
        for r in runs.itertuples():
            e = R.table(r.run_dir, "episodes")
            ev = R.table(r.run_dir, "eval")
            if not len(e):
                continue
            ret = e["ret"].values
            conv = convergence_episode(ret)
            last = ret[-max(20, len(ret) // 5):]
            rows.append(dict(algo=algo, name=name, seed=r.seed, episodes=len(ret), converge_episode=conv,
                             final_train=float(ret[-max(20, len(ret) // 10):].mean()), late_std=float(last.std()),
                             best_val=float(ev.val_return.max()) if len(ev) else np.nan,
                             last_val=float(ev.val_return.iloc[-1]) if len(ev) else np.nan,
                             val_drawdown=float(ev.val_return.max() - ev.val_return.iloc[-1]) if len(ev) else np.nan))
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2))
    x = np.arange(len(ALGOS))
    for j, (col, title, yl) in enumerate([("converge_episode", "Episodes to stable performance", "episodes"),
                                          ("late_std", "Return spread late in training", "std of episode return"),
                                          ("val_drawdown", "Validation drawdown (best minus last)", "return units")]):
        ax = axes[j]
        for i, algo in enumerate(ALGOS):
            v = df[df.algo == algo][col].dropna().values
            if col == "converge_episode" and len(v) < (df.algo == algo).sum():
                ax.text(i, 0, "no\nconv.", ha="center", va="bottom", fontsize=7, color="#777")
            ax.bar(i, v.mean() if len(v) else 0, color=COLOR[algo], alpha=0.85, width=0.6)
            ax.scatter(np.full(len(v), i), v, color="k", s=10, zorder=3)
        ax.set_xticks(x); ax.set_xticklabels([ALGO_LABEL[a] for a in ALGOS]); ax.set_title(title); ax.set_ylabel(yl)
    fig.tight_layout(); p = out / "fig_convergence_stability.png"; fig.savefig(p); plt.close(fig)
    df.to_csv(out.parent / "convergence_table.csv", index=False)
    return p, df


def hyperparameter_comparison(R: Results, out: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.4))
    for ax, algo in zip(axes.ravel(), ALGOS):
        g = R.runs(algo).groupby("name")["best_val_return"].agg(["mean", "std", "count"]).sort_values("mean")
        y = np.arange(len(g))
        ax.barh(y, g["mean"], xerr=g["std"].fillna(0), color=COLOR[algo], alpha=0.85, ecolor="#444", error_kw=dict(lw=0.8, capsize=2))
        ax.set_yticks(y); ax.set_yticklabels(g.index, fontsize=7.5); ax.set_title(f"{ALGO_LABEL[algo]}: validation return by configuration")
        if R.test is not None:
            rb = R.test[(R.test.split == "test") & (R.test.policy == "rule_based")]["return"].mean()
            ax.axvline(rb, color=COLOR["rule_based"], ls="--", lw=1)
        ax.set_xlabel("best validation return (mean over seeds)")
    fig.tight_layout(); p = out / "fig_hyperparameters.png"; fig.savefig(p); plt.close(fig); return p


def generalization(R: Results, out: Path) -> Path:
    t = R.test
    splits = ["test"] + sorted(s for s in t.split.unique() if s.startswith("ood:"))
    pols = [a for a in ALGOS + ["rule_based", "random"] if a in set(t.policy)]
    M = np.array([[t[(t.policy == p) & (t.split == s)]["return"].mean() for s in splits] for p in pols])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw=dict(width_ratios=[1.35, 1]))
    ax = axes[0]
    w = 0.8 / len(pols)
    for i, p in enumerate(pols):
        ax.bar(np.arange(len(splits)) + i * w - 0.4 + w / 2, M[i], w, color=COLOR[p], label=ALGO_LABEL[p])
    ax.set_xticks(np.arange(len(splits))); ax.set_xticklabels([s.replace("ood:", "").replace("_", "\n") if s != "test" else "unseen\nseasons" for s in splits], fontsize=7.5)
    ax.set_ylabel("mean return"); ax.set_title("Held-out seasons and out-of-distribution weather"); ax.legend(ncol=3, fontsize=7); ax.axhline(0, color="k", lw=0.6)
    ax = axes[1]
    fail = np.array([[t[(t.policy == p) & (t.split == s)]["crop_failure"].mean() for s in splits] for p in pols])
    im = ax.imshow(fail, cmap="Oranges", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(splits))); ax.set_xticklabels([s.replace("ood:", "").replace("_", "\n") if s != "test" else "unseen" for s in splits], fontsize=6.5)
    ax.set_yticks(np.arange(len(pols))); ax.set_yticklabels([ALGO_LABEL[p] for p in pols], fontsize=8); ax.grid(False)
    for i in range(len(pols)):
        for j in range(len(splits)):
            ax.text(j, i, f"{fail[i, j]:.2f}", ha="center", va="center", fontsize=7, color="k" if fail[i, j] < 0.6 else "w")
    ax.set_title("Crop failure rate (harvest < 500 kg)")
    fig.tight_layout(); p = out / "fig_generalization.png"; fig.savefig(p); plt.close(fig); return p


def agricultural_outcomes(R: Results, out: Path) -> Path:
    t = R.test[R.test.split == "test"]
    pols = [a for a in ALGOS + ["rule_based", "random"] if a in set(t.policy)]
    panels = [("marketable_bags", "Marketable grain (90 kg bags)"), ("net_return_kes", "Net result for the year (KES)"),
              ("soil_change", "Soil organic matter change (index)"), ("post_harvest_loss_frac", "Post-harvest loss (fraction)"),
              ("crop_failure", "Crop failure rate"), ("food_reserve_met", "Food reserve within 10% of 490 kg")]
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.2))
    for ax, (col, title) in zip(axes.ravel(), panels):
        m = [t[t.policy == p][col].mean() for p in pols]
        e = [t[t.policy == p].groupby("seed")[col].mean().sem() * 1.96 for p in pols]
        ax.bar(range(len(pols)), m, yerr=e, color=[COLOR[p] for p in pols], alpha=0.88, ecolor="#333", error_kw=dict(lw=0.8, capsize=2))
        ax.set_xticks(range(len(pols))); ax.set_xticklabels([ALGO_LABEL[p].replace("Rule-based", "Rule") for p in pols], fontsize=7.5, rotation=20)
        ax.set_title(title, fontsize=9)
    fig.suptitle("Agricultural outcomes on 200 unseen seasons (error bars: 95% CI over seasons)", fontsize=9.5)
    fig.tight_layout(); p = out / "fig_agri_outcomes.png"; fig.savefig(p); plt.close(fig); return p


def resource_efficiency(R: Results, out: Path) -> Path:
    t = R.test[R.test.split == "test"]
    pols = [a for a in ALGOS + ["rule_based", "random"] if a in set(t.policy)]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2))
    for ax, (col, title) in zip(axes, [("water_used_m3", "Pan water drawn (m³)"), ("water_use_efficiency_kg_m3", "Water use efficiency (kg grain per m³)"),
                                       ("invalid_actions", "Impossible actions attempted per season")]):
        m = [t[t.policy == p][col].mean() for p in pols]
        e = [t[t.policy == p].groupby("seed")[col].mean().sem() * 1.96 for p in pols]
        ax.bar(range(len(pols)), m, yerr=e, color=[COLOR[p] for p in pols], alpha=0.88, ecolor="#333", error_kw=dict(lw=0.8, capsize=2))
        ax.set_xticks(range(len(pols))); ax.set_xticklabels([ALGO_LABEL[p].replace("Rule-based", "Rule") for p in pols], fontsize=7.5, rotation=20); ax.set_title(title, fontsize=9)
    fig.tight_layout(); p = out / "fig_resource_efficiency.png"; fig.savefig(p); plt.close(fig); return p


def action_profile(R: Results, out: Path) -> Path:
    t = R.test[R.test.split == "test"]
    pols = [a for a in ALGOS + ["rule_based", "random"] if a in set(t.policy)]
    cols = [f"n_{n}" for n in ACTION_NAMES]
    M = np.array([t[t.policy == p][cols].mean().values for p in pols])
    fig, ax = plt.subplots(figsize=(9.6, 3.0))
    im = ax.imshow(M, cmap="Greens", aspect="auto"); ax.grid(False)
    ax.set_yticks(range(len(pols))); ax.set_yticklabels([ALGO_LABEL[p] for p in pols])
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([n.replace("_", " ") for n in ACTION_NAMES], rotation=60, ha="right", fontsize=7)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i, j] >= 0.5:
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=6.5, color="k" if M[i, j] < M.max() * 0.6 else "w")
    ax.set_title("Mean number of times each action is taken per season (unseen seasons)")
    fig.tight_layout(); p = out / "fig_action_profile.png"; fig.savefig(p); plt.close(fig); return p


def all_figures(R: Results, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    figs = {"learning": learning_curves(R, out), "eval": evaluation_rewards(R, out), "objective": dqn_loss_and_entropy(R, out)}
    figs["convergence"], conv = convergence_and_stability(R, out)
    figs["hyper"] = hyperparameter_comparison(R, out)
    if R.test is not None:
        figs["general"] = generalization(R, out); figs["agri"] = agricultural_outcomes(R, out)
        figs["resource"] = resource_efficiency(R, out); figs["actions"] = action_profile(R, out)
    return figs
