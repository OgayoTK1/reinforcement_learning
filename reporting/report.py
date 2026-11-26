"""Builds the report (DOCX, then PDF through LibreOffice) from experiment outputs. Every number in the text and tables is
read from the results directory at build time; the prose around the numbers is written here and phrased conditionally
where the claim depends on what the data show."""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from environment import ACTION_DESCRIPTIONS, ACTION_NAMES
from environment.observation import OBS_VARS
from environment.params import P
from experiments.matrix import _M

from . import stats as S
from .behaviour import invalid_profile, invalid_summary
from .data import ALGO_LABEL, ALGOS, Results, full_config
from .docx_helpers import figure, footer_page_numbers, new_document, para, table

L = ALGO_LABEL


def f3(x, d=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def sgn(x, d=3):
    return f"{x:+.{d}f}"


def pm(m, s, n, d=3):
    return f"{m:.{d}f}" if n < 2 or np.isnan(s) else f"{m:.{d}f} ± {s:.{d}f}"


def env_composite(assets: Path, out: Path) -> Path:
    names = ["env_vegetative.png", "env_grainfill.png", "env_drought.png"]
    ims = [Image.open(assets / n).convert("RGB").crop((0, 0, 1070, 900)) for n in names if (assets / n).exists()]
    if not ims:
        return None
    w = 640
    ims = [im.resize((w, int(im.height * w / im.width))) for im in ims]
    canvas = Image.new("RGB", (w * len(ims) + 8 * (len(ims) - 1), ims[0].height), "white")
    for i, im in enumerate(ims):
        canvas.paste(im, (i * (w + 8), 0))
    canvas.save(out, quality=90)
    return out


def obs_rows() -> list[list[str]]:
    rows, seen = [], set()
    for v in OBS_VARS:
        key = next((k for k in ("stage_", "store_", "prev_action_") if v.name.startswith(k)), None)
        if key:
            if key in seen:
                continue
            seen.add(key)
            n = sum(1 for x in OBS_VARS if x.name.startswith(key))
            what = {"stage_": "Crop stage (6 values)", "store_": "Storage type in use (3 values)", "prev_action_": "Action taken last week (20 values)"}[key]
            rows.append([f"{key}* ({n})", what, v.source, "binary, one-hot", v.rng, "weekly"])
        else:
            rows.append([v.name, v.meaning, v.source, f"{v.dtype}; {v.norm}", v.rng, v.update + (", estimate" if v.kind == "estimated" else "")])
    return rows


def hyper_table(d, R: Results, algo: str, cols: list[tuple[str, str]], widths: list[float]):
    reg = R.runs(algo)
    g = reg.groupby("name")["best_val_return"].agg(["mean", "std", "count"])
    sel = R.best_name(algo)
    header = ["#", "Config"] + [c[0] for c in cols] + ["Validation return"]
    rows, bold = [], []
    for i, (name, ov, factor) in enumerate(_M[algo]):
        cfg = full_config(algo, name)
        vals = []
        for _, key in cols:
            v = cfg.get(key) if "|" not in key else " / ".join(str(cfg.get(k)) for k in key.split("|"))
            vals.append(str(v))
        r = g.loc[name] if name in g.index else None
        rows.append([str(i + 1), name] + vals + [pm(r["mean"], r["std"], int(r["count"])) if r is not None else "failed"])
        if name == sel:
            bold.append(i)
    table(d, header, rows, widths, size=7.2, bold_rows=bold, align_right_from=len(header) - 1)


_FLOOR = {}


def wait_floor() -> float:
    """Validation return of a farmer who never acts (never sows). Runs whose best validation return equals this never left that policy."""
    if "v" not in _FLOOR:
        from evaluation.evaluator import evaluate_policy
        from training.splits import VAL_SEEDS

        class Wait:
            def reset(self): pass
            def act(self, obs): return 0
        _FLOOR["v"] = float(evaluate_policy(Wait(), VAL_SEEDS)["return"].mean())
    return _FLOOR["v"]


def why(algo: str, name: str, gain: bool, steps: int) -> str:
    """The mechanism behind a configuration's effect, worded for the direction it actually went."""
    dflt = full_config(algo, "base")
    eps_d = int(dflt.get("eps_decay_frac", 0.4) * steps)
    eps_f = int(0.15 * steps)
    harvest_w = 32
    W = {
        "lr_up": ("with a short budget, bigger steps reach a useful policy sooner", "bigger steps overshot and made updates erratic"),
        "lr_down": ("smaller steps kept updates stable", "smaller steps learned too little within the budget"),
        "gamma_0.95": ("a shorter horizon lowered the variance of the returns",
                       f"at γ = 0.95 harvest and sale income about {harvest_w} weeks after sowing is weighted by {0.95 ** harvest_w:.2f}, so it barely reaches early decisions"),
        "gamma_0.999": ("the late harvest and sale income keeps almost full weight in early decisions", "almost undiscounted 52-week returns are noisier to learn from"),
        "buffer_5k": ("a small buffer holds only recent transitions, which match the current policy more closely", "a small buffer gives less varied, more correlated updates"),
        "batch_256": ("larger minibatches gave lower-variance gradient steps", "larger minibatches with the same update ratio gave fewer effective updates per sample"),
        "eps_fast": (f"ε reaches its floor after about {eps_f:,} steps instead of {eps_d:,}, leaving more of the budget to refine a policy",
                     "exploration stopped before the agent had found the harvest-and-sell sequence"),
        "eps_slow": ("longer exploration found more of the 20 actions' effects", "acting randomly for 80 percent of training with a 0.1 floor left little budget to exploit what was learned"),
        "hard_noDouble": ("hard target copies passed value changes on faster", "without double DQN the max over 20 noisy Q-values overestimates, and hard copies make the targets jump"),
        "episodes_4": ("four times as many updates in the same budget outweighed the noisier gradient", "four-episode batches gave too noisy a gradient"),
        "episodes_64": ("64-episode batches gave a much less noisy gradient", f"64-episode batches allow only about {max(1, int(steps / 52 / 64))} updates in the whole budget"),
        "entropy_up": ("a larger entropy bonus kept the policy exploring rather than settling early", "a larger entropy bonus kept the policy too random to commit"),
        "entropy_0": ("without an entropy bonus the policy committed sooner", "without an entropy bonus the policy committed early to a poor action pattern"),
        "no_baseline": ("raw returns happened to work here", "raw returns without a baseline give a much noisier gradient"),
        "wide_256": ("the wider network fitted the stage-dependent decisions better", "the wider network needed more data than the budget gave"),
        "clip_0.1": ("a tighter clip gave smaller, safer policy steps", "a tighter clip slowed learning within the budget"),
        "clip_0.3": ("a looser clip let each batch move the policy further", "a looser clip let single batches push the policy too far"),
        "epochs_15": ("more epochs squeezed more learning out of each batch", "15 epochs over-fitted each batch and moved the policy too far from the data"),
        "batch_512": ("short rollouts give four times as many updates in the same budget", "short rollouts gave noisy advantage estimates"),
        "lambda_0.8": ("lower λ cut the variance of the advantage estimates", "lower λ leans on a critic that is still inaccurate, so the advantages are biased"),
        "nstep_32": ("very short rollouts give many more updates", "32-step rollouts gave noisy updates from less than one season of data"),
        "nstep_512": ("long rollouts gave steadier advantage estimates", "long rollouts give far fewer updates in the budget"),
    }
    key = {"lr_1e-3": "lr_up", "lr_3e-3": "lr_up", "lr_1e-4": "lr_down", "lr_3e-4": "lr_down", "entropy_0.05": "entropy_up"}.get(name, name)
    return W[key][0 if gain else 1] if key in W else ""


def effects_text(R: Results, algo: str) -> str:
    reg = R.runs(algo)
    g = reg.groupby("name")["best_val_return"].mean()
    if "base" not in g.index or len(g) < 3:
        return ""
    base = g["base"]
    steps = R.steps()
    diff = (g - base).drop("base").sort_values(ascending=False)

    def item(n, v):
        w = why(algo, n, v > 0, steps)
        return f"{n} ({sgn(v)}{'; ' + w if w else ''})"
    best = [item(n, v) for n, v in diff.head(2).items() if v > 0]
    worst = [item(n, v) for n, v in diff.tail(2)[::-1].items() if v < 0]
    stuck = int((g <= wait_floor() + 0.005).sum())
    s = (f"{stuck} of {len(g)} configurations never did better than a farmer who never sows (validation return {wait_floor():.3f}) within the budget. " if stuck else "")
    s += f"Against the default ({base:.3f}), "
    if best:
        s += "the largest gains were " + " and ".join(best) + ". "
    if worst:
        s += "The largest losses were " + " and ".join(worst) + "."
    return s


def sweep_summary(R: Results) -> str:
    """Patterns across the four sweeps, and how much a single seed can be trusted."""
    lr_down = {"dqn": "lr_1e-4", "reinforce": "lr_3e-4", "ppo": "lr_1e-4", "a2c": "lr_1e-4"}
    lr_up = {"dqn": "lr_1e-3", "reinforce": "lr_3e-3", "ppo": "lr_1e-3", "a2c": "lr_3e-3"}
    worst2, best2 = [], []
    for a in ALGOS:
        g = R.runs(a).groupby("name")["best_val_return"].mean()
        if "base" in g.index:
            dd = (g - g["base"]).drop("base").sort_values()
            worst2 += [a for n in dd.head(2).index if n == lr_down[a]]
            best2 += [a for n in dd.tail(2).index if n == lr_up[a]]
    s = ""
    if len(worst2) >= 3:
        s += (f"Lowering the learning rate was one of the two worst changes for {len(worst2)} of 4 algorithms"
              + (f", and raising it one of the two best for {len(best2)}" if len(best2) >= 2 else "")
              + ". That points to the step budget, not instability, as the main limit on these runs. ")
    n_seeds = R.n_seeds()
    if n_seeds < 2:
        s += "Each configuration has one seed, so I cannot separate an effect from seed-to-seed noise; I read single differences below about 0.1 as leads, not findings."
    else:
        sd = R.runs("dqn", "base")["best_val_return"].std()
        s += f"Across seeds the DQN default itself varies by {sd:.3f} (sd), so differences of that size are not evidence of an effect."
    return s


def entropy_stats(R: Results, sel: dict) -> dict:
    out = {}
    for a in ["reinforce", "ppo", "a2c"]:
        runs = R.runs(a, sel.get(a))
        if not len(runs):
            continue
        u = R.table(runs.iloc[0].run_dir, "updates")
        if "entropy" not in u or len(u) < 4:
            continue
        e = u.entropy.to_numpy()
        i = int(np.argmin(e))
        out[a] = dict(first=float(e[:3].mean()), last=float(e[-3:].mean()), lo=float(e[i]), lo_step=int(u.step.iloc[i]), hi=float(e.max()))
    return out


def entropy_text(R: Results, sel: dict) -> str:
    parts = []
    for a, s in entropy_stats(R, sel).items():
        if s["lo"] < s["first"] - 0.5 and s["last"] > s["lo"] + 0.5:
            parts.append(f"{L[a]}'s entropy collapsed from {s['first']:.2f} to {s['lo']:.2f} by step {s['lo_step']:,} and then climbed back to {s['last']:.2f}: the policy committed hard, "
                         "the whole-episode returns that followed did not support that commitment, and it spread out again, which is the instability expected from a critic-free gradient")
        elif s["last"] < s["first"] - 0.3:
            parts.append(f"{L[a]}'s entropy fell steadily from {s['first']:.2f} to {s['last']:.2f}, a gradual move from exploring to a committed policy")
        else:
            parts.append(f"{L[a]}'s entropy stayed between {s['lo']:.2f} and {s['hi']:.2f} (ending at {s['last']:.2f}), as spread out as a uniform choice among about {np.exp(s['last']):.0f} actions, so the sampled policy never committed even though its greedy action did well")
    return ". ".join(parts) + "." if parts else ""


def convergence_text(conv: pd.DataFrame) -> str:
    c = conv.groupby("algo")[["episodes", "converge_episode", "val_drawdown", "best_val", "last_val"]].mean().reindex([a for a in ALGOS if a in set(conv.algo)])
    frac = (c.converge_episode / c.episodes).dropna()
    s = ""
    if len(frac) and frac.min() > 0.75:
        s += (f"Every method reached this point only at {frac.min() * 100:.0f} to {frac.max() * 100:.0f} percent of its training episodes, so none had levelled off: the runs were still improving "
              "when the budget ran out, and longer training would probably change the returns. ")
    unstable = c[c.val_drawdown > 0.1].sort_values("val_drawdown", ascending=False)
    if len(unstable):
        s += ("The validation drawdown separates stable from unstable learners: "
              + ", ".join(f"{L[a]} ended {r.val_drawdown:.3f} below its best ({r.best_val:.3f} to {r.last_val:.3f})" for a, r in unstable.iterrows()))
        steady = [L[a] for a in c.index if c.loc[a, "val_drawdown"] < 0.01]
        s += (f", while {' and '.join(steady)} finished at {'their' if len(steady) > 1 else 'its'} best. " if steady else ". ")
        s += "Because the checkpoint used for testing is the best one on validation (`best.pt`), a late collapse does not reach the test results, but it shows how fragile those runs were."
    return s


def observed_text(R: Results, sel: dict, conv: pd.DataFrame, summ: pd.DataFrame) -> str:
    """How far this run bore out those expectations, one clause per method, from the logged numbers."""
    c = conv.groupby("algo")[["val_drawdown", "late_std"]].mean()
    ent = entropy_stats(R, sel)
    algos = [a for a in ALGOS if a in summ.index and a in c.index]
    if not algos:
        return ""
    ret = summ.loc[algos, "return"]; dd = c.loc[algos, "val_drawdown"]; fail = summ.loc[algos, "crop_failure"]
    bits = []
    for a in algos:
        r, e = ret[a], ent.get(a)
        b = f"{L[a]} scored {r:.3f} on test"
        if r == ret.max():
            b += ", the best"
        elif r == ret.min():
            b += ", the weakest"
        if dd[a] < 0.01:
            b += ", and finished training at its best validation score"
        else:
            b += f", and its validation return fell {dd[a]:.3f} below its best by the end" + (", the least stable of the four" if dd[a] == dd.max() else "")
        if fail[a] == fail.max() and fail[a] > fail.min() + 0.05:
            b += f"; it had the most crop failures ({fail[a] * 100:.0f} percent)"
        if e and e["lo"] < e["first"] - 0.5 and e["last"] > e["lo"] + 0.5:
            b += f"; its entropy collapse and rebound ({e['first']:.2f} to {e['lo']:.2f} to {e['last']:.2f}) is the high-variance gradient showing up directly"
        elif e and e["last"] < e["first"] - 0.3:
            b += "; its entropy fell smoothly"
        bits.append(b)
    return "**In this run** " + "; ".join(bits) + ". With one seed per configuration these are observations about this run, not general rankings."


def gap_text(summ, top: str) -> str:
    """How much of the top learner's margin over the rule-based policy comes from the soil and water terms of the reward."""
    a, b = summ.loc[top], summ.loc["rule_based"]
    gap = a["return"] - b["return"]
    soil = P.som_value_kes * (a["soil_change"] - b["soil_change"]) / P.r0
    water = -P.water_shadow_kes_m3 * (a["water_used_m3"] - b["water_used_m3"]) / P.r0
    return (f"Of the {gap:+.3f} return difference between {L[top]} and the rule-based policy, about {soil:+.3f} comes from the soil term "
            f"(soil organic matter change {a['soil_change']:+.3f} against {b['soil_change']:+.3f}) and {water:+.3f} from the water term; "
            f"the rest is {(gap - soil - water):+.3f}, mainly wealth, labour and the end-of-year reserve. The reward weights are my choice, so a different soil weight would change this split.")


def build(R: Results, figs: dict, out_dir: Path, assets: Path, student="Ogayo Andrew", video="https://www.youtube.com/watch?v=vv_giK4jY10", repo="https://github.com/OgayoTK1/reinforcement_learning") -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sel = {a: R.best_name(a) for a in ALGOS}
    test = R.test
    summ = S.summary_table(test, "test").set_index("policy")
    tests = S.paired_tests(test, "test")
    conv = pd.read_csv(out_dir / "convergence_table.csv")
    n_seeds, steps = R.n_seeds(), R.steps()
    ep_total = int(conv.episodes.median()) if len(conv) else 0
    ood = test[test.split.str.startswith("ood")].groupby(["policy", "split"])["return"].mean().unstack()
    ranking = summ.loc[[a for a in ALGOS if a in summ.index], "return"].sort_values(ascending=False)
    top = ranking.index[0]
    rb = summ.loc["rule_based", "return"]; rnd = summ.loc["random", "return"]
    vs_rb = tests[tests.b == "rule_based"].set_index("a")

    d = new_document(9.5)
    footer_page_numbers(d, "AgriResilience RL: Reinforcement Learning Summative")
    para(d, "AFRICAN LEADERSHIP UNIVERSITY   |   BSE   |   ML TECHNIQUES II   |   SUMMATIVE ASSIGNMENT", bold=True, size=8, after=1)
    para(d, "Reinforcement Learning Summative Assignment Report", bold=True, size=15, after=1)
    para(d, f"**Student Name:** {student}      **Video Recording:** {video}      **GitHub Repository:** {repo}", size=8.5, after=2)

    # ------------------------------------------------------------------ 1
    d.add_heading("1. Project Overview", 1)
    para(d, f"I built AgriResilience RL, a simulator of one year on a one-acre maize holding at Njoro in Nakuru County, Kenya. Each week the "
            f"farmer picks one of 20 actions (sow, irrigate from a lined pan, fertilise, scout, spray, weed, harvest, dry, store, sell) and sees only noisy "
            f"farm observations and a weather forecast, never the true state. The reward is the change in household wealth, with penalties for soil mining, "
            f"water use, unpaid family labour and ending the year below a 490 kg food reserve. I trained DQN, REINFORCE, PPO and A2C with TorchRL on the same "
            f"Gymnasium environment, tuned each with 10 configurations, picked configurations on validation seasons only, and tested them on {int(summ['n_episodes'].max())} unseen seasons "
            f"and seven out-of-distribution weather scenarios. The weather and many agronomic parameters are modelled, not measured on a farm, so the results compare "
            f"algorithms inside my model and do not show that any policy would raise yields on a real farm.")
    para(d, f"**Scope of this run.** Each of the 40 configurations was trained for {steps:,} environment steps (about {ep_total:,} episodes) with {n_seeds} seed{'s' if n_seeds != 1 else ''}. "
            + ("This is below the 3 seeds and 100,000 steps the pipeline is designed for, so rankings between close configurations are tentative." if (n_seeds < 3 or steps < 100_000) else ""), size=9)

    # ------------------------------------------------------------------ 2
    d.add_heading("2. Environment Description", 1)
    d.add_heading("a. Agent(s)", 2)
    para(d, f"One agent: the household member who manages the plot. It stands for a smallholder who decides once per week, has about {P.labour_cap_h:g} hours of family labour per week "
            f"(extra hours are hired at KES {P.wage_h:g}), limited cash, a cow whose manure accumulates, and a lined water pan filled by rain. It cannot see soil nutrients, true "
            "pest numbers or next week's rain. It sees what a farmer can sense: a rain gauge, a thermometer, finger-test soil moisture, a visual canopy score, "
            "market prices, and a forecast whose skill varies by season. Scouting the field for one week of labour reduces the noise on pest and disease estimates.")
    d.add_heading("b. Action Space", 2)
    para(d, "Discrete(20), one action per week for 52 weeks. An action that is not possible (no water, no cash, no stock, wrong stage) is executed as a wait and costs a small "
            "penalty; the validity mask is returned in `info` for monitoring only and is not given to any learner.", after=2)
    half = 10
    rows = [[str(i), ACTION_NAMES[i], ACTION_DESCRIPTIONS[i], str(i + half), ACTION_NAMES[i + half], ACTION_DESCRIPTIONS[i + half]] for i in range(half)]
    table(d, ["#", "Action", "Real-world meaning", "#", "Action", "Real-world meaning"], rows, [0.6, 2.6, 5.4, 0.6, 2.8, 5.4], size=7)
    d.add_heading("c. Observation Space", 2)
    para(d, f"The agent receives a {len(OBS_VARS)}-element float32 vector, a partial and noisy view of the hidden farm state. Estimates (soil moisture, canopy, pest and disease "
            "pressure, grain moisture, spoilage) carry Gaussian reading error; the forecast has a skill that changes by season. Each row says how the number "
            "would be gathered on a real farm. In the simulator every row is produced by `environment/observation.py` every week and enters the network as one input "
            "to the first linear layer. The same table is in `docs/OBSERVATION_SPACE.md`.", after=2)
    table(d, ["Observation", "Description", "Source (sensor/camera/API/dataset)", "Encoding / data type", "Range", "Update"], obs_rows(), [3.1, 4.4, 3.5, 3.1, 1.6, 1.7], size=6.6)
    d.add_heading("d. Reward Structure", 2)
    para(d, f"With W the farm's wealth (cash plus {P.liquid_disc * 100:.0f} percent of the quality-adjusted value of stored grain) and R0 = KES {P.r0:,.0f} the reference gross revenue, the reward each week is "
            f"**r_t = ΔW/R0 + {P.som_value_kes:,.0f}·ΔSOM/R0 − {P.water_shadow_kes_m3:g}·water_m³/R0 − {P.family_opp_h:g}·family_hours/R0 − {P.invalid_penalty:g}·[invalid]**, "
            f"and at the end of the episode a food-reserve term **−(shortfall_kg/{P.bag_kg:g})·price·{P.shortfall_mult:g}/R0** is added for any shortfall below the {P.reserve_kg:g} kg reserve. "
            f"SOM is the mean soil organic matter index; water is costed at KES {P.water_shadow_kes_m3:g} per m³ (refilling the pan) and family labour at KES {P.family_opp_h:g} per hour. "
            "Selling grain, paying for inputs and spoilage all show up through W. There is no shaping term on biomass, greenness or growth stage, so the agent is never "
            f"paid for looking busy. Episodes end at week 52, if the crop is not sown by week {P.sow_deadline_week}, if every plot is dead, or if cash falls below −KES {abs(P.cash_floor):,.0f}.", after=2)
    comp = env_composite(assets, out_dir / "env_visual.png")
    if comp:
        figure(d, comp, 17.2, "Figure 1. The Three.js viewer reading real Gymnasium states (rule-based policy, seed 1001): vegetative stage, grain filling, and grain filling under the severe-drought scenario. "
                              "Colour, plant height, the water in the pan and the sacks by the granary come from `serialize_state()`.")
    para(d, "**Why RL.** The outcomes I care about arrive months after the decisions: fertiliser timing shows up at harvest, a pesticide spray changes next week's pest load, "
            "drying decides whether stored grain moulds, and a sale now removes the food reserve later. Water, cash and labour limits couple these choices, and the information is partial. "
            "There is no labelled 'correct action' to learn from, which rules out supervised learning, and the horizon is too coupled for a fixed calendar to be best in every season.", size=9)

    # ------------------------------------------------------------------ 3
    d.add_heading("3. System Analysis And Design", 1)
    d.add_heading("a. Deep Q-Network (DQN)", 2)
    para(d, "A two-layer MLP (128 units, ReLU) maps the 65 observations to 20 Q-values (`QValueActor`). Exploration is epsilon-greedy with a linear decay from 1.0 to a floor (`EGreedyModule`). "
            "Transitions go into a replay buffer (`LazyTensorStorage`); `DQNLoss` uses the Huber loss, a delayed target network and optional Double DQN, and the target is updated by "
            "Polyak averaging (`SoftUpdate`) or periodic copies (`HardUpdate`). I added a configurable updates-per-step ratio (0.25 by default), gradient clipping and a learning-start threshold. "
            "Episode ends are treated as terminal, which is correct here because the week number is part of the observation.")
    d.add_heading("b. Policy Gradient Methods (REINFORCE, PPO, A2C)", 2)
    para(d, "All three use the same actor, a 2 × 128 tanh MLP producing logits for a `Categorical` policy (`ProbabilisticActor`), with a small initial output scale so the first policy is near uniform. "
            "**REINFORCE** is a genuine Monte-Carlo method: it rolls out whole episodes with `env.rollout`, computes discounted returns to go from the real rewards, subtracts the batch mean "
            "(a baseline that does not depend on the action), normalises, and adds an entropy bonus; it has no critic. "
            "**PPO** adds a separate value network, GAE advantages and `ClipPPOLoss` with several epochs of minibatch updates per 2,048-step batch. "
            "**A2C** uses the same critic and GAE but takes one gradient step per short on-policy rollout (`A2CLoss`). Evaluation always uses the greedy action (argmax of Q-values or logits).")
    d.add_heading("c. System Integration", 2)
    para(d, "Each checkpoint is a weights file plus a JSON sidecar (algorithm, network sizes, observation-spec hash, seed, training metrics); `agents/loader.py` refuses a checkpoint whose "
            "observation-spec hash does not match the live environment. One policy interface (`reset()`, `act(obs)`) serves three front ends: the evaluator, `play.py` in the terminal "
            "(one line per week), and a FastAPI backend. The backend exposes REST endpoints (`/api/policies`, `/api/catalogue`, `/api/observation_spec`, and `POST /api/episode`, "
            "which runs a whole season server-side) and a WebSocket (`/ws`) that streams one state per week with start, step, pause, resume, reset and speed messages. "
            "Every state is produced by `serialize_state()` in `environment/rendering.py`: per-plot soil moisture, nutrients, canopy, pest and disease levels, weather, stock, cash, "
            "the action taken and its reward components. The Three.js viewer only draws these JSON states and contains no farm equations, so what it shows is what the "
            "environment computed. The protocol is documented in `docs/API.md`.")

    # ------------------------------------------------------------------ 4
    d.add_heading("4. Implementation", 1)
    para(d, f"Every configuration was trained for {steps:,} environment steps on the same environment, with {n_seeds} seed{'s' if n_seeds != 1 else ''}. Config 1 is the default; each of "
            "configs 2 to 10 changes the factor named in `experiments/matrix.py`. The last column is the best validation return reached during training (50 fixed seasons, greedy policy), "
            "mean ± sd over seeds. The selected configuration for each algorithm is shaded.", after=2)
    d.add_heading("a. DQN", 2)
    hyper_table(d, R, "dqn", [("Learning rate", "lr"), ("Gamma", "gamma"), ("Replay buffer", "buffer_size"), ("Batch", "batch_size"), ("Exploration (decay frac / floor)", "eps_decay_frac|eps_end"), ("Target / double", "target|double_dqn")],
                [0.6, 2.3, 1.4, 1.1, 1.5, 1.2, 3.2, 2.3, 3.3])
    d.add_heading("b. REINFORCE", 2)
    hyper_table(d, R, "reinforce", [("Learning rate", "lr"), ("Gamma", "gamma"), ("Episodes / update", "episodes_per_update"), ("Baseline", "baseline"), ("Entropy coef", "ent_coef"), ("Hidden", "hidden")],
                [0.6, 2.3, 1.5, 1.1, 1.7, 2.1, 1.5, 1.2, 3.5])
    d.add_heading("c. PPO", 2)
    hyper_table(d, R, "ppo", [("Learning rate", "lr"), ("Gamma", "gamma"), ("Clip eps", "clip_eps"), ("Entropy", "ent_coef"), ("Epochs", "epochs"), ("Batch / minibatch", "frames_per_batch|minibatch"), ("GAE lambda", "lmbda")],
                [0.6, 2.2, 1.4, 1.0, 1.1, 1.1, 1.0, 2.3, 1.3, 3.1])
    d.add_heading("d. A2C", 2)
    para(d, "The template has no A2C table, so I added one in the same form.", size=8.5, after=1)
    hyper_table(d, R, "a2c", [("Learning rate", "lr"), ("Gamma", "gamma"), ("Rollout steps", "frames_per_batch"), ("Entropy", "ent_coef"), ("GAE lambda", "lmbda"), ("Critic coef", "critic_coef")],
                [0.6, 2.3, 1.5, 1.1, 1.7, 1.2, 1.5, 1.5, 3.3])
    figure(d, figs["hyper"], 15.5, "Figure 2. Best validation return for every configuration (bars: mean over seeds, whiskers: sd). The dashed line is the rule-based policy's mean return on the test seasons.")
    for a in ALGOS:
        t = effects_text(R, a)
        if t:
            para(d, f"**{L[a]}.** {t}", size=9)
    para(d, f"**Across the sweeps.** {sweep_summary(R)}", size=9)

    # ------------------------------------------------------------------ 5
    d.add_heading("5. Results Discussion", 1)
    d.add_heading("a. Cumulative Rewards and Training Stability", 2)
    figure(d, figs["learning"], 15.8, "Figure 3. Cumulative reward per training episode for the selected configuration of each method (band: sd over seeds; dashed line: rule-based test return; dotted: random).")
    sel_txt = ", ".join(f"{L[a]} ({sel[a]})" for a in ALGOS)
    para(d, f"Selected configurations: {sel_txt}. On the {int(summ['n_episodes'].max())} unseen test seasons the mean returns were "
            + ", ".join(f"{L[a]} {summ.loc[a, 'return']:.3f}" for a in ranking.index) + f", against {rb:.3f} for the rule-based policy and {rnd:.3f} for random actions. "
            + (f"{L[top]} scored highest. " if True else "")
            + ", ".join(f"{L[a]} {'beat' if vs_rb.loc[a, 'mean_diff'] > 0 else 'trailed'} the rule-based policy by {abs(vs_rb.loc[a, 'mean_diff']):.3f} (95% CI {vs_rb.loc[a, 'ci_lo']:+.3f} to {vs_rb.loc[a, 'ci_hi']:+.3f}, Holm-adjusted p = {vs_rb.loc[a, 'p_holm']:.3g})" for a in ALGOS if a in vs_rb.index) + ". "
            "Seasons are paired: every policy faced the same weather and prices, so these are paired Wilcoxon tests on per-season returns.")
    figure(d, figs["objective"], 17.4, "Figure 4. Objective curves: DQN temporal-difference loss (log scale) and policy entropy for REINFORCE, PPO and A2C. The dotted line is the entropy of a uniform policy over 20 actions (ln 20 = 3.0).")
    dq = R.table(R.runs("dqn", sel["dqn"]).iloc[0].run_dir, "updates")
    cq = conv.set_index("algo")
    txt = "**Reading the objective curves.** "
    if len(dq):
        txt += (f"The DQN TD loss fell from {dq.td_loss.iloc[:5].mean():.4f} in the first updates to {dq.td_loss.iloc[-5:].mean():.4f} at the end, yet its validation return peaked at "
                f"{cq.loc['dqn', 'best_val']:.3f} and finished at {cq.loc['dqn', 'last_val']:.3f} (Figure 5). A small TD loss means the Q-values agree with their own targets, not that the greedy "
                "policy is good, because the targets move with the policy and the replay data. ")
    txt += f"Entropy shows how far each policy method has committed (uniform over 20 actions is {np.log(20):.2f}). " + entropy_text(R, sel) + " "
    para(d, txt, size=9)
    figure(d, figs["eval"], 11.2, "Figure 5. Greedy validation return during training (50 fixed seasons, mean ± sd over seeds).")

    d.add_heading("b. Episodes To Converge", 2)
    figure(d, figs["convergence"], 16.5, "Figure 6. Episodes until the smoothed training return stays above 90% of its final improvement; late-training spread of episode returns; and the gap between the best and last validation return.")
    crow = []
    for a in ALGOS:
        c = conv[conv.algo == a]
        v = c.converge_episode.dropna()
        crow.append([L[a], f"{int(c.episodes.median()):,}", f"{v.median():,.0f}" if len(v) else "no stable improvement", f"{c.final_train.mean():.3f}", f"{c.late_std.mean():.3f}", f"{c.best_val.mean():.3f}", f"{c.val_drawdown.mean():.3f}"])
    table(d, ["Method", "Training episodes", "Episodes to converge", "Final train return", "Late sd of return", "Best val return", "Val drawdown"], crow, [2.4, 2.4, 3.2, 2.4, 2.4, 2.4, 2.2], size=7.5, align_right_from=1)
    para(d, "I define convergence as the first episode after which the 50-episode moving average of the training return never falls below the start value plus 90 percent of the "
            "gain between the start and the end of training. A run that gains less than 0.05 return units is reported as having no stable improvement. This definition describes the "
            "training curve of the run, so a method can 'converge' to a mediocre policy; I therefore also report the validation drawdown, which is how far the last evaluation sat below the best one. "
            + convergence_text(conv), size=9)

    d.add_heading("c. Generalization", 2)
    figure(d, figs["general"], 17.4, "Figure 7. Test return on 200 unseen in-distribution seasons and 50 seasons in each of seven out-of-distribution scenarios, and the crop-failure rate in each.")
    cols = ["test"] + list(ood.columns.str.replace("ood:", "", regex=False).str.replace("_", " "))
    rows = []
    for p in ALGOS + ["rule_based", "random"]:
        if p in summ.index:
            rows.append([L[p], f"{summ.loc[p, 'return']:.3f}"] + [f"{ood.loc[p].iloc[j]:.3f}" for j in range(ood.shape[1])])
    table(d, ["Policy", "unseen"] + cols[1:], rows, [2.3] + [1.9] * (len(cols)), size=7, align_right_from=1)
    drop = {a: summ.loc[a, "return"] - ood.loc[a].mean() for a in ALGOS if a in ood.index}
    worst_ood = {a: ood.loc[a].idxmin().replace("ood:", "").replace("_", " ") for a in ALGOS if a in ood.index}
    para(d, "The unseen seasons come from the same weather generator as training, with different seeds. The seven scenarios (severe drought, late onset, wet year, warming by 1.5 and 3 °C, a poor forecast, and drought with warming) lie "
            "outside the training draw. Mean return fell by "
            + ", ".join(f"{drop[a]:.3f} for {L[a]}" for a in drop) + " from the unseen seasons to the average OOD scenario, and the hardest scenario for each was "
            + ", ".join(f"{worst_ood[a]} ({L[a]})" for a in worst_ood) + ". The rule-based policy fell by " + f"{rb - ood.loc['rule_based'].mean():.3f}" + ". "
            "All four learners were trained only on mild year-to-year variation, so a drop in the OOD scenarios is expected; what matters for comparison is whether it is larger than the rule-based policy's.", size=9)
    figure(d, figs["agri"], 15.4, "Figure 8. Agricultural outcomes on the 200 unseen seasons.")
    figure(d, figs["resource"], 15.4, "Figure 9. Resource efficiency: pan water drawn, grain per m³ of water (rain plus irrigation) and impossible actions attempted per season.")
    ag = summ
    para(d, "**What the policies actually did.** " + "; ".join(f"{L[p]}: {ag.loc[p, 'marketable_bags']:.1f} marketable bags, net KES {ag.loc[p, 'net_return_kes']:,.0f}, {ag.loc[p, 'crop_failure'] * 100:.0f}% crop failures, {ag.loc[p, 'water_used_m3']:.0f} m³ water, {ag.loc[p, 'invalid_actions']:.1f} impossible actions per season, food reserve kept in {ag.loc[p, 'food_reserve_met'] * 100:.0f}% of seasons" for p in ALGOS + ["rule_based", "random"] if p in ag.index)
            + ". " + gap_text(summ, top) + " The per-action counts behind this are in `experiments/results/test_summary.csv` and Figure 10 of the repository (`fig_action_profile.png`).", size=9)
    inv = invalid_summary(invalid_profile(out_dir))
    if inv:
        def top2(a):
            return " and ".join(f"{act.replace('_', ' ')} ({reason}, {v:.1f})" for act, reason, v in inv[a]["top"])
        para(d, "**Why the learners attempt so many impossible actions.** Replaying the exported checkpoints on the same test seasons shows that the learners almost never choose "
                "wait (" + ", ".join(f"{L[a]} {inv[a]['wait']:.1f}" for a in ALGOS if a in inv) + " times per season). Their idle weeks are spent on an action that is impossible "
                "at that moment and therefore runs as a wait: " + "; ".join(f"{L[a]} mostly {top2(a)}" for a in ALGOS if a in inv) + " per season. "
                f"Each costs only {P.invalid_penalty:g}, so {inv[top]['invalid']:.0f} of them cost {L[top]} about {inv[top]['invalid'] * P.invalid_penalty:.2f} per season, small next to its return. "
                "The learners have in effect learned a second 'wait' action, and because the mask is not given to them they cannot tell the two apart. This comes from the reward design "
                "and costs little on its own. "
                + (lambda good, bad: (f"{', '.join(L[a] for a in good)} still take the actions that matter at the right stage: they sow, harvest, dry and store, losing at most "
                                      f"{max(summ.loc[a, 'post_harvest_loss_frac'] for a in good) * 100:.0f} percent of the grain after harvest. " if good else "")
                   + "".join(f"{L[a]} harvests but almost never dries or stores the grain, so it loses {summ.loc[a, 'post_harvest_loss_frac'] * 100:.0f} percent of its grain after harvest and sells only {summ.loc[a, 'marketable_bags']:.1f} bags, a large part of why its return is low. " for a in bad))(
                    [a for a in ALGOS if a in summ.index and summ.loc[a, "post_harvest_loss_frac"] < 0.1],
                    [a for a in ALGOS if a in summ.index and summ.loc[a, "post_harvest_loss_frac"] >= 0.1])
                + "A larger invalid-action penalty or action masking would remove the habit.", size=9)

    # ------------------------------------------------------------------ 6
    d.add_heading("6. Conclusion and Discussion", 1)
    beat = [a for a in ALGOS if a in vs_rb.index and vs_rb.loc[a, "mean_diff"] > 0 and vs_rb.loc[a, "p_holm"] < 0.05]
    lose = [a for a in ALGOS if a in vs_rb.index and vs_rb.loc[a, "mean_diff"] < 0 and vs_rb.loc[a, "p_holm"] < 0.05]
    pair = tests[(tests.a.isin(ALGOS)) & (tests.b.isin(ALGOS))]
    sig_pairs = pair[pair.p_holm < 0.05]
    concl = f"{L[top]} had the highest mean test return ({ranking.iloc[0]:.3f}). "
    concl += (f"{', '.join(L[a] for a in beat)} beat the rule-based policy with Holm-adjusted p below 0.05. " if beat else "No learner beat the rule-based extension calendar by a margin that survived the multiple-comparison correction. ")
    if lose:
        concl += f"{', '.join(L[a] for a in lose)} did significantly worse than it. "
    concl += (f"Between learners, {len(sig_pairs)} of {len(pair)} pairwise differences were significant after correction. " if len(pair) else "")
    concl += "These comparisons are on paired seasons from my simulator, with a fixed training budget and few training seeds, so they show how the methods behave in this model and under this budget, not which algorithm is better in general."
    para(d, concl)
    para(d, "**Strengths and weaknesses for this problem.** DQN reuses every transition, which matters when only 52 decisions make an episode, but it must learn values for 20 actions across stage-dependent decisions "
            "and its performance depends on the exploration schedule and buffer. REINFORCE is the simplest and the only method with no critic, but its gradient uses whole-episode returns, so it is noisy and needs many episodes per update. "
            "PPO's clipped updates and GAE usually give steadier progress per batch at the price of more hyperparameters. A2C updates after every short rollout, which is quick but noisier than PPO. "
            + observed_text(R, sel, conv, summ))
    para(d, "**Limits.** The weather is synthetic, calibrated to published normals, because no daily station record for Njoro was available; parameters are tagged literature, calibrated, synthetic or convenience in `environment/params.py`. "
            "Maize is a sole crop with no intercropping, one farm size and one soil family. Four plots share one decision. Reward weights for soil and water are my modelling choices, and I found while building it that an early soil weight let a policy "
            "earn most of its return from repeated manure application, so I reduced the soil gain per application. Training budgets are small by RL standards. "
            "**With more time** I would add more seeds and longer runs, a recurrent policy to handle partial observability, action masking in the learners, intercropping, and validation of the water and pest modules against farm records.")
    docx_path = out_dir / "AgriResilience_RL_Report.docx"
    d.save(docx_path)
    pdf = to_pdf(docx_path)
    return docx_path, pdf


def to_pdf(docx_path: Path) -> Path:
    """LibreOffice if it is installed; otherwise Microsoft Word through PowerShell (Windows)."""
    pdf = docx_path.with_suffix(".pdf")
    try:
        subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(docx_path.parent), str(docx_path)], check=True, capture_output=True, timeout=180)
    except FileNotFoundError:
        src, dst = str(docx_path.resolve()), str(pdf.resolve())
        ps = (f"$w = New-Object -ComObject Word.Application; $w.DisplayAlerts = 0; $w.Visible = $false; "
              f"$d = $w.Documents.Open('{src}', $false, $true, $false); $d.ExportAsFixedFormat('{dst}', 17); $d.Close($false); $w.Quit()")
        subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True, capture_output=True, timeout=180)
    return pdf


def page_count(pdf: Path) -> int:
    out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
    return int([l for l in out.splitlines() if l.startswith("Pages:")][0].split()[1])
