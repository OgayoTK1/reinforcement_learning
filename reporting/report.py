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


def effects_text(R: Results, algo: str) -> str:
    reg = R.runs(algo)
    g = reg.groupby("name")["best_val_return"].mean()
    if "base" not in g.index or len(g) < 3:
        return ""
    base = g["base"]
    fac = {n: f for n, _, f in _M[algo]}
    diff = (g - base).drop("base").sort_values(ascending=False)
    seed_sd = reg[reg["name"] == "base"]["best_val_return"].std()
    noise = f" Across seeds the default itself varies by {seed_sd:.3f} (sd), so differences of that size are not evidence of an effect." if not np.isnan(seed_sd) else \
        " With one seed per configuration I cannot separate a real effect from seed-to-seed noise, so I read these as leads, not findings."
    best = [f"{n} ({fac[n]}; {sgn(v)})" for n, v in diff.head(2).items() if v > 0]
    worst = [f"{n} ({fac[n]}; {sgn(v)})" for n, v in diff.tail(2).items() if v < 0]
    stuck = int((g <= wait_floor() + 0.005).sum())
    s = (f"{stuck} of {len(g)} configurations never did better than a farmer who never sows (validation return {wait_floor():.3f}) within the budget. " if stuck else "")
    s += f"Against the default ({base:.3f}), "
    if best:
        s += "the largest gains were " + " and ".join(best) + ". "
    if worst:
        s += "The largest losses were " + " and ".join(worst) + "."
    return s + noise


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
    para(d, "One agent: the household member who manages the plot. It stands for a smallholder who decides once per week, has about 56 hours of family labour per week "
            "(extra hours are hired at KES 75), limited cash, a cow whose manure accumulates, and a lined water pan filled by rain. It cannot see soil nutrients, true "
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
    para(d, "With W the farm's wealth (cash plus 85 percent of the quality-adjusted value of stored grain) and R0 = KES 88,000 the reference gross revenue, the reward each week is "
            "**r_t = ΔW/R0 + 150,000·ΔSOM/R0 − 5·water_m³/R0 − 40·family_hours/R0 − 0.01·[invalid]**, "
            "and at the end of the episode a food-reserve term **−(shortfall_kg/90)·price·1.5/R0** is added, where SOM is the mean soil organic matter index and the weights are those in `environment/params.py`. "
            "Selling grain, paying for inputs and spoilage all show up through W. There is no shaping term on biomass, greenness or growth stage, so the agent is never "
            "paid for looking busy. Episodes end at week 52, if the crop is not sown by week 8, if every plot is dead, or if cash falls below −KES 20,000.", after=2)
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
    para(d, "**Reading the objective curves.** " + (f"The DQN TD loss moved from {dq.td_loss.iloc[:5].mean():.4f} in the first updates to {dq.td_loss.iloc[-5:].mean():.4f} at the end. A falling TD loss alone does not mean the policy improved, because the targets move as the policy and the replay data change; I therefore read it next to the evaluation curve (Figure 5). " if len(dq) else "")
            + "Entropy shows how fast each policy method commits to a behaviour: a curve that collapses early leaves little exploration, and one that stays near ln 20 means the policy is still close to random. "
            + "REINFORCE, which waits for whole-episode returns and has the noisiest gradient, is the one I would expect to drift the most between updates.", size=9)
    figure(d, figs["eval"], 11.2, "Figure 5. Greedy validation return during training (50 fixed seasons, mean ± sd over seeds).")

    d.add_heading("b. Episodes To Converge", 2)
    figure(d, figs["convergence"], 16.5, "Figure 6. Episodes until the smoothed training return stays above 90% of its final improvement; late-training spread of episode returns; and the gap between the best and last validation return.")
    crow = []
    for a in ALGOS:
        c = conv[conv.algo == a]
        v = c.converge_episode.dropna()
        crow.append([L[a], f"{int(c.episodes.median()):,}", f"{v.median():,.0f}" if len(v) else "no stable improvement", f"{c.final_train.mean():.3f}", f"{c.late_std.mean():.3f}", f"{c.best_val.mean():.3f}", f"{c.val_drawdown.mean():.3f}"])
    table(d, ["Method", "Training episodes", "Episodes to converge", "Final train return", "Late sd of return", "Best val return", "Val drawdown"], crow, [2.4, 2.4, 3.2, 2.4, 2.4, 2.4, 2.2], size=7.5, align_right_from=1)
    para(d, "I define convergence as the first episode after which the 40-episode moving average of the training return never falls below the start value plus 90 percent of the "
            "gain between the start and the end of training. A run that gains less than 0.05 return units is reported as having no stable improvement. This definition describes the "
            "training curve of the run, so a method can 'converge' to a mediocre policy; I therefore also report the validation drawdown, which is how far the last evaluation sat below the best one.", size=9)

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
            "Whether each of these tendencies showed up is what Sections 4 and 5 report; where a result goes against the usual expectation I treat it as a property of this run and not as a finding.")
    para(d, "**Limits.** The weather is synthetic, calibrated to published normals, because the build sandbox could not reach station data; parameters are tagged literature, calibrated, synthetic or convenience in `environment/params.py`. "
            "Maize is a sole crop with no intercropping, one farm size and one soil family. Four plots share one decision. Reward weights for soil and water are my modelling choices, and I found while building it that an early soil weight let a policy "
            "earn most of its return from repeated manure application, so I reduced the soil gain per application. Training budgets are small by RL standards. "
            "**With more time** I would add more seeds and longer runs, a recurrent policy to handle partial observability, action masking in the learners, intercropping, and validation of the water and pest modules against farm records.")
    docx_path = out_dir / "AgriResilience_RL_Report.docx"
    d.save(docx_path)
    pdf = to_pdf(docx_path)
    return docx_path, pdf


def to_pdf(docx_path: Path) -> Path:
    subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(docx_path.parent), str(docx_path)], check=True, capture_output=True, timeout=180)
    return docx_path.with_suffix(".pdf")


def page_count(pdf: Path) -> int:
    out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
    return int([l for l in out.splitlines() if l.startswith("Pages:")][0].split()[1])
