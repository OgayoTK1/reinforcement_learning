"""Architecture diagrams drawn with matplotlib so they regenerate with the repo."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK, PAPER = "#2b2a26", "#f3ecd9"
FILL = {"env": "#dfe8d0", "rl": "#d6e2ee", "io": "#efe2c4", "ui": "#e8d6e0", "data": "#e6e6e0"}


def _box(ax, x, y, w, h, text, kind="env", fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.012", fc=FILL[kind], ec=INK, lw=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, fontweight="bold" if bold else "normal", linespacing=1.25)


def _arrow(ax, a, b, text="", rad=0.0, fs=7, dy=0.012):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11, lw=1.0, color=INK, connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + dy, text, ha="center", va="bottom", fontsize=fs, color="#555", style="italic")


def _canvas(w=10, h=5.2, title=""):
    fig, ax = plt.subplots(figsize=(w, h)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    if title:
        ax.text(0.0, 0.985, title, fontsize=11, fontweight="bold", color=INK, va="top")
    return fig, ax


def system_architecture(out: Path) -> Path:
    fig, ax = _canvas(10.5, 5.4, "System architecture")
    _box(ax, 0.03, 0.62, 0.22, 0.22, "AgriResilienceEnv\n(Gymnasium)\nweather, crop, soil, pests,\nstorage, market, reward", "env", bold=True)
    _box(ax, 0.03, 0.30, 0.22, 0.20, "Baselines\nrandom, rule-based", "rl")
    _box(ax, 0.36, 0.62, 0.24, 0.22, "TorchRL training\nDQN  REINFORCE  PPO  A2C\nGymWrapper, collectors,\nlosses, GAE", "rl", bold=True)
    _box(ax, 0.36, 0.30, 0.24, 0.20, "Experiment runner\n40 configs x seeds\nresumable, failures logged", "rl")
    _box(ax, 0.70, 0.62, 0.26, 0.22, "Checkpoints\nmodels/dqn, models/pg\n.pt weights + .json spec hash", "data")
    _box(ax, 0.70, 0.30, 0.26, 0.20, "Results\nexperiments/results\nCSV logs, registry.csv", "data")
    _box(ax, 0.03, 0.04, 0.22, 0.16, "evaluation/\nvalidation, test, OOD,\nstatistics", "rl")
    _box(ax, 0.36, 0.04, 0.24, 0.16, "FastAPI backend\nREST + WebSocket\nSession(policy, env)", "io", bold=True)
    _box(ax, 0.70, 0.04, 0.26, 0.16, "Three.js viewer\nrenders serialize_state()\nno agronomy in JavaScript", "ui", bold=True)
    _arrow(ax, (0.25, 0.73), (0.36, 0.73), "obs, reward")
    _arrow(ax, (0.36, 0.78), (0.25, 0.78), "action", dy=0.0)
    _arrow(ax, (0.60, 0.73), (0.70, 0.73), "best / final")
    _arrow(ax, (0.48, 0.62), (0.48, 0.50), "")
    _arrow(ax, (0.60, 0.40), (0.70, 0.40), "logs")
    _arrow(ax, (0.14, 0.62), (0.14, 0.50), "")
    _arrow(ax, (0.14, 0.30), (0.14, 0.20), "")
    _arrow(ax, (0.80, 0.62), (0.55, 0.20), "", rad=-0.12)
    ax.text(0.70, 0.545, "load", fontsize=7, color="#555", style="italic")
    _arrow(ax, (0.25, 0.68), (0.40, 0.20), "", rad=0.25)
    ax.text(0.205, 0.46, "step()", fontsize=7, color="#555", style="italic")
    _arrow(ax, (0.60, 0.12), (0.70, 0.12), "JSON frames")
    p = out / "arch_system.png"; fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white"); plt.close(fig); return p


def environment_components(out: Path) -> Path:
    fig, ax = _canvas(10.5, 5.6, "Environment components (environment/)")
    _box(ax, 0.38, 0.84, 0.24, 0.10, "custom_env.py\nreset / step / terminal rules", "env", bold=True)
    items = [(0.02, 0.58, "weather.py\nWet/dry Markov rain, AR(1)\ntemperature, Hargreaves ET0"), (0.27, 0.58, "state.py\ndraw_episode(seed, scenario)\nall randomness pre-drawn"),
             (0.52, 0.58, "actions.py\n20 actions, validity masks,\nresource costs, labour"), (0.77, 0.58, "params.py\nparameter registry with\nprovenance L / C / S / E")]
    for x, y, t in items:
        _box(ax, x, y, 0.21, 0.15, t, "env")
    items2 = [(0.02, 0.30, "dynamics.py\nFAO-56 water balance, SCS runoff,\nthermal-time phenology, RUE growth"), (0.27, 0.30, "dynamics.py\nN and P pools, soil organic matter,\nweeds, fall armyworm, leaf blight"),
              (0.52, 0.30, "dynamics.py\nharvest, drying, mould, insects,\nmarket price and sales"), (0.77, 0.30, "rewards.py\nwealth change, soil, water,\nlabour, invalid, food reserve")]
    for x, y, t in items2:
        _box(ax, x, y, 0.21, 0.15, t, "env")
    _box(ax, 0.10, 0.04, 0.30, 0.14, "observation.py\n65 partial, noisy observations\nforecast and scouting estimates", "io", bold=True)
    _box(ax, 0.58, 0.04, 0.30, 0.14, "rendering.py\nserialize_state() schema 1.0\nfor the viewer", "ui", bold=True)
    for x in (0.12, 0.37, 0.62, 0.87):
        _arrow(ax, (0.5, 0.84), (x, 0.73), "")
        _arrow(ax, (x, 0.58), (x, 0.45), "")
    _arrow(ax, (0.30, 0.30), (0.25, 0.18), ""); _arrow(ax, (0.62, 0.30), (0.70, 0.18), "")
    p = out / "arch_environment.png"; fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white"); plt.close(fig); return p


def rl_workflow(out: Path) -> Path:
    fig, ax = _canvas(10.5, 4.2, "Reinforcement learning workflow")
    steps = [("1  Config\nexperiments/matrix.py\n10 per algorithm", "data"), ("2  Collect\nTorchRL collector or\nfull-episode rollouts", "rl"), ("3  Learn\nDQNLoss, ClipPPOLoss,\nA2CLoss, MC returns", "rl"),
             ("4  Validate\n50 fixed seasons,\ngreedy policy", "env"), ("5  Select\nbest.pt by validation\nreturn only", "data"), ("6  Test once\n200 seasons + 7 OOD\nscenarios, paired stats", "env"),
             ("7  Report\nfigures, tables,\nDOCX / PDF", "ui")]
    w, g = 0.122, 0.02
    for i, (t, k) in enumerate(steps):
        x = 0.02 + i * (w + g)
        _box(ax, x, 0.45, w, 0.28, t, k, fs=7.6)
        if i:
            _arrow(ax, (x - g, 0.59), (x, 0.59))
    ax.annotate("", xy=(0.02 + 1 * (w + g) + w / 2, 0.45), xytext=(0.02 + 3 * (w + g) + w / 2, 0.45), arrowprops=dict(arrowstyle="-|>", lw=1, color=INK, connectionstyle="arc3,rad=0.35"))
    ax.text(0.30, 0.22, "repeat until the step budget is used; evaluate every 10% of the budget", fontsize=7.5, style="italic", color="#555", ha="center")
    ax.text(0.02, 0.05, "Seeds: training episodes use seeds >= 10 000, validation 1 to 50, test 1000 to 1199, OOD 2000 to 2049. The ranges cannot overlap.", fontsize=7.5, color=INK)
    p = out / "arch_workflow.png"; fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white"); plt.close(fig); return p


def _poly(ax, pts, text="", fs=6.5):
    xs, ys = zip(*pts)
    ax.plot(xs[:-1], ys[:-1], color=INK, lw=1.0, solid_capstyle="round")
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=11, lw=1.0, color=INK))
    if text:
        ax.text((pts[0][0] + pts[-1][0]) / 2, pts[-2][1] - 0.035, text, ha="center", va="top", fontsize=fs, color="#555", style="italic")


def data_flow(out: Path) -> Path:
    fig, ax = _canvas(10.5, 4.4, "Data flow during playback")
    _box(ax, 0.02, 0.58, 0.17, 0.22, "Browser\nstart, pause or\nmanual action", "ui", bold=True)
    _box(ax, 0.25, 0.58, 0.17, 0.22, "WebSocket /ws\nJSON text frames", "io")
    _box(ax, 0.48, 0.58, 0.17, 0.22, "Session\npolicy.act(obs)\nenv.step(action)", "rl", bold=True)
    _box(ax, 0.71, 0.58, 0.17, 0.22, "AgriResilienceEnv\nauthoritative state", "env", bold=True)
    _box(ax, 0.48, 0.22, 0.17, 0.20, "Checkpoint\nload_policy()\nspec hash check", "data")
    _box(ax, 0.71, 0.22, 0.17, 0.20, "serialize_state()\nschema 1.0 dict", "env")
    _box(ax, 0.02, 0.22, 0.17, 0.20, "Three.js scene\nvalues become colour,\nheight, water level", "ui")
    _arrow(ax, (0.19, 0.69), (0.25, 0.69), "")
    _arrow(ax, (0.42, 0.69), (0.48, 0.69), "")
    _arrow(ax, (0.65, 0.69), (0.71, 0.69), "action", fs=6.5)
    ax.text(0.22, 0.715, "start{policy, seed}", fontsize=6.3, color="#555", style="italic", ha="center", va="bottom")
    _arrow(ax, (0.565, 0.42), (0.565, 0.58), "")
    ax.text(0.575, 0.50, "weights", fontsize=6.5, color="#555", style="italic")
    _arrow(ax, (0.795, 0.58), (0.795, 0.42), "")
    ax.text(0.805, 0.50, "read only", fontsize=6.5, color="#555", style="italic")
    _poly(ax, [(0.795, 0.22), (0.795, 0.10), (0.105, 0.10), (0.105, 0.22)], "state frame: Session, then WebSocket, then Browser")
    p = out / "arch_dataflow.png"; fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white"); plt.close(fig); return p


def all_diagrams(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    return {"system": system_architecture(out), "environment": environment_components(out), "workflow": rl_workflow(out), "dataflow": data_flow(out)}
