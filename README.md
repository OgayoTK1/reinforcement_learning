# AgriResilience RL

A reinforcement-learning testbed for smallholder maize farming at **Njoro, Nakuru County, Kenya**. One agent manages a one-acre
holding for 52 weeks (Gymnasium environment `AgriResilienceEnv`), and four TorchRL agents (DQN, REINFORCE, PPO, A2C) are trained,
tuned (40 configurations), compared against a random and a rule-based policy, and replayed in a Three.js farm viewer that
reads real environment states. Built for the ALU *Machine Learning Techniques II* summative.

* **Video walkthrough:** https://www.youtube.com/watch?v=vv_giK4jY10
* **Repository:** https://github.com/OgayoTK1/reinforcement_learning

**Read this first.** This is a simulation and an algorithm comparison. The weather is synthetic (calibrated to published normals),
and many agronomic parameters are literature-informed assumptions. Results show how the four algorithms behave inside this model,
not that any policy would raise yields on a real farm. Every parameter is tagged literature / calibrated / synthetic / convenience in
`environment/params.py`.

## Quick start (fresh clone)

```bash
git clone https://github.com/OgayoTK1/reinforcement_learning.git
cd reinforcement_learning
uv sync                 # creates .venv and installs everything from uv.lock (PyTorch is large, allow a few minutes)
uv run main.py          # starts the viewer at http://127.0.0.1:8000 and opens your browser
```

In the viewer choose *Who is farming* (a trained checkpoint, the rule-based policy, random actions, or *Manual*), a season type and an
episode seed, then press Start. The right-hand "farm record book" shows the numbers; the left shows the farm. Drag to rotate, wheel to
zoom, right-drag or Shift-drag to pan. No internet is needed: Three.js is vendored in `frontend/vendor/`.

Terminal playback with no browser, printing one line per week:

```bash
uv run play.py --policy models/pg/ppo.pt --seed 1000
uv run play.py --policy rule_based --scenario severe_drought --seed 1001
uv run play.py --list
```

`models/` holds the checkpoints selected for the report (see "Results provenance"). The commands below retrain them from scratch.

## Commands

| Command | What it does |
|---|---|
| `uv run main.py` | start the viewer (alias for `uv run main.py serve [--port N] [--no-browser]`) |
| `uv run main.py train --workers 4 --seeds 0 1 2 --steps 100000` | run the whole experiment matrix; safe to stop and rerun, finished runs are skipped |
| `uv run main.py train --algos ppo --max-minutes 8` | run in time-limited chunks |
| `uv run main.py evaluate` | score the validation-selected configuration of each algorithm, plus the baselines, on 200 test seasons and 7 OOD scenarios |
| `uv run main.py export` | copy the best checkpoints into `models/dqn` and `models/pg` |
| `uv run main.py report` | rebuild figures, tables, diagrams, `docs/API.md` and the report (`docs/report/*.docx` and `.pdf`) |
| `uv run python -m training.dqn_training --steps 100000 --set lr=1e-3` | train one DQN run (logs in `logs/`) |
| `uv run python -m training.pg_training --algo ppo --set clip_eps=0.1` | train one REINFORCE / PPO / A2C run |
| `uv run pytest` | run the tests |

Full pipeline after cloning:

```bash
uv sync
uv run main.py train --workers 4 --seeds 0 1 2 --steps 100000   # resumable, results in experiments/results/
uv run main.py evaluate
uv run main.py export
uv run main.py report
uv run main.py                                                  # view the exported models
```

**Compute.** Measured on 2 shared CPU cores without a GPU, one 100,000-step run takes about 4 to 6 minutes (DQN is the slowest). The default
matrix is 40 configurations x 3 seeds = 120 runs, about 8 to 10 CPU-hours; `--workers 6` brings that near 1.5 hours.
`--steps 40000 --seeds 0` runs a reduced version in under an hour on 2 cores. The runner records failed runs in
`experiments/results/failures.md` instead of dropping them.

## Repository layout

```
main.py  play.py  pyproject.toml  uv.lock  README.md
environment/   custom_env.py (Gymnasium env)  rendering.py (serialize_state)  params.py  weather.py  state.py
               actions.py  dynamics.py  rewards.py  observation.py
agents/        dqn.py  reinforce.py  ppo.py  a2c.py  common.py  loader.py          (TorchRL learners, checkpoint I/O)
training/      dqn_training.py  pg_training.py  experiment_runner.py  train_one.py  splits.py  export_models.py
evaluation/    evaluator.py  baselines.py  final_eval.py
experiments/   matrix.py (the 40 configurations)  configs/*.yaml  results/ (CSV logs, registry, test episodes)
backend/       app.py (FastAPI REST + WebSocket)  session.py
frontend/      index.html  style.css  src/*.js  vendor/three.module.min.js
models/        dqn/  pg/        logs/        assets/ (viewer screenshots)        tests/
reporting/     figures.py  stats.py  report.py  diagrams.py  api_doc.py
docs/          DESIGN.md  ARCHITECTURE.md  API.md  OBSERVATION_SPACE.md  RUBRIC_MATRIX.md  DEMO_SCRIPT.md  report/
```

## The environment in brief

* **Horizon.** 52 weekly decisions starting 1 March. Seven daily sub-steps per week drive the crop, water, pest and storage models.
* **Actions.** `Discrete(20)`. An impossible action (no cash, no water, wrong stage) runs as a wait and costs a small penalty. The learners are not given the mask.

| id | action | real-world meaning |
|---|---|---|
| 0 | `wait` | No intervention this week |
| 1 | `sow` | Sow certified hybrid maize at rain onset |
| 2 | `irrigate_10mm` | Irrigate 10 mm from the farm pan |
| 3 | `irrigate_25mm` | Irrigate 25 mm from the farm pan |
| 4 | `irrigate_40mm` | Irrigate 40 mm from the farm pan |
| 5 | `basal_dap_50kg` | Basal DAP, 50 kg per acre, at or soon after planting |
| 6 | `topdress_can_25kg` | Top-dress CAN, 25 kg per acre |
| 7 | `topdress_can_50kg` | Top-dress CAN, 50 kg per acre |
| 8 | `apply_manure` | Apply cattle manure, 500 kg |
| 9 | `scout_field` | Walk the rows and inspect whorls, leaves and soil |
| 10 | `spray_insecticide` | Spray insecticide against fall armyworm |
| 11 | `spray_fungicide` | Spray fungicide against leaf blight |
| 12 | `hand_weed` | Hand weeding with a hoe |
| 13 | `mulch` | Spread crop residue or grass mulch |
| 14 | `harvest` | Harvest a labour-limited share of the standing crop |
| 15 | `sun_dry` | Sun-dry harvested grain on a drying sheet |
| 16 | `store_pp_bags` | Store the grain in polypropylene bags |
| 17 | `store_hermetic_bags` | Store the grain in hermetic bags |
| 18 | `sell_half_surplus` | Sell half of the stock above the household food reserve |
| 19 | `sell_all_surplus` | Sell all of the stock above the household food reserve |

* **Observations.** 65 float32 values: calendar, crop stage, canopy and health estimates, soil moisture estimates, rain gauge, thermometer, forecast,
  nutrient and pest estimates, resources, market price, storage state, last action. Each has a stated real-world source in
  [docs/OBSERVATION_SPACE.md](docs/OBSERVATION_SPACE.md), generated from the code.
* **Reward.** `r = dW/R0 + soil value - water shadow price - family labour cost - invalid-action penalty`, plus a food-reserve shortfall at the end,
  with W = cash + 85% of quality-adjusted stored grain and R0 = KES 88,000. No shaping on biomass or greenness.
* **Termination.** Week 52; not sown by week 8; all plots dead; cash below -KES 20,000.
* **Randomness.** All of an episode's randomness (weather, pest flights, prices, labour shocks, observation noise) is drawn from the seed before the first
  step and does not depend on actions, so every policy faces the same season for the same seed and runs are reproducible.
* **Scenarios.** `training` (mild variation), `baseline`, `severe_drought`, `late_onset`, `wet_year`, `warm_1p5`, `warm_3`, `poor_forecast`, `drought_warm`.

## Training and evaluation protocol

* The same environment and a shared discrete action space for all four algorithms; DQN, PPO and A2C use TorchRL's collectors, buffers and losses. REINFORCE is
  Monte-Carlo (whole-episode returns, no critic) with a hand-written loss, because TorchRL has none without a value function.
* 40 configurations (10 per algorithm), each with several seeds, equal environment-step budget. Config 1 is the default, configs 2 to 10 change one factor each.
* Seed ranges cannot overlap: training >= 10,000, validation 1 to 50, test 1000 to 1199, out-of-distribution 2000 to 2049.
* Validation picks `best.pt` inside a run and the configuration for each algorithm. The test seasons and OOD scenarios are scored only for those selected configurations.
* Statistics use paired comparisons (same seasons for every policy): Wilcoxon signed-rank, bootstrap confidence intervals, Holm correction.

## Results provenance

Numbers in the report come from `experiments/results/` and are rebuilt by `uv run main.py report`. The submitted results, the checkpoints in `models/` and the
figures come from 1 seed and 40,000 steps per configuration (40 runs, none failed), which is below the 3 seeds and 100,000 steps the pipeline supports;
the report states this and treats close rankings as tentative.
Nothing in the report is typed by hand except the prose around the numbers.

## Tests

`uv run pytest` runs 53 tests: Gymnasium API conformance, determinism, no leakage of future weather into observations, persistence of action effects
(irrigation, fertiliser, spraying, manure, storage), resource limits, terminal conditions, reward accounting, scripted reward-hacking policies, observation-spec
consistency, agent training and checkpoint round-trips, WebSocket and REST behaviour. They check plumbing and model behaviour; they do not check that a trained agent is good.

## Limitations and troubleshooting

* Weather is synthetic. Maize is a sole crop (no intercropping). Four plots share one decision. The agronomic modules are simplified (single-Kc water balance, radiation-use-efficiency growth).
* The viewer needs a browser with WebGL. If the port is busy use `uv run main.py serve --port 8001`.
* If `uv sync` is slow the cause is the PyTorch download. CPU-only wheels are fine; no GPU is used or needed.
* On Windows run the same commands in PowerShell or the VS Code terminal. Multiprocessing uses the `spawn` start method, so run `main.py train` from the project root.
