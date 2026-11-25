# Rubric and requirement compliance matrix

Status key: **Done** means implemented and exercised in this repository by tests or a run; **Needs you** means something only you can do;
**Reduced** means done at a smaller scale than the final plan and clearly labelled.

## Assignment tasks

| Requirement | Where | Status |
|---|---|---|
| 2.1.1 Action space exhaustive, relevant, mappable to the real world | `environment/actions.py`; report 2b; README table | Done |
| 2.1.2 Observation space with provenance | `environment/observation.py`; `docs/OBSERVATION_SPACE.md`; report 2c | Done |
| 2.1.3 Rewards tied to states and actions | `environment/rewards.py`; report 2d; `tests/test_environment.py::test_reward_*` | Done |
| 2.1.4 Start state | `environment/state.py::initial_state` (randomised cash, vouchers, manure, soil per plot) | Done |
| 2.1.5 Terminal conditions | `custom_env.py::_terminal`; four tests in `test_environment.py` | Done |
| 2.1.6 Advanced visualisation | Three.js scene in `frontend/` fed by `backend/` over WebSocket | Done |
| 2.1.7 Visual in the report | Figure 1 from real viewer screenshots in `assets/` | Done |
| 2.2 Four algorithms with TorchRL on the same environment | `agents/dqn.py`, `reinforce.py`, `ppo.py`, `a2c.py` | Done |
| 2.3 At least 10 meaningfully different runs per algorithm | `experiments/matrix.py` (10 each, one-factor changes from a default); `tests/test_agents.py::test_matrix_has_ten_distinct_configs_per_algorithm` | Done (design); results **Reduced** until you run the full matrix |
| 2.4 5-minute video | script and execution plan in `docs/DEMO_SCRIPT.md` | **Needs you** (I cannot record your screen and camera) |
| 2.5 Report, 7 to 10 pages, template, PDF | `docs/report/AgriResilience_RL_Report.pdf` and `.docx`; page count checked at build time | Done for the build I ran; **rebuild after your full run** |
| 2.6 Repository `hamed_rl_summative`, structure, uv, `uv.lock` | this tree; `pyproject.toml`, `uv.lock` | Local repo done; **Needs you** to push it to GitHub under that name |
| Runs after clone with `uv sync` then `uv run main.py` | checked from a clean copy, see below | Done |
| Submit a doc or docx to Canvas | `docs/report/AgriResilience_RL_Report.docx` | **Needs you** |

## Rubric rows

| Criterion | Evidence |
|---|---|
| Environment validity and complexity | Action-dependent persistent state (water, nitrogen, soil organic matter, pest and disease, storage, cash, labour); delayed trade-offs (fertiliser at week 3 affects the week 30 harvest, drying decides spoilage months later); documented start, terminal conditions and reward; every observation mapped to meaning, range, real-world source, update cadence (`docs/OBSERVATION_SPACE.md`); a "why RL" paragraph in report section 2. Persistence and delay are tested in `test_environment.py` |
| Hyperparameter experiments and analysis | 10 configurations per algorithm covering shared controls (learning rate, gamma) and method-specific ones (DQN: exploration, buffer, batch, target update, double DQN; REINFORCE: episodes per update, baseline, entropy, width; PPO: clip, epochs, rollout, GAE lambda, entropy; A2C: rollout length, entropy, GAE lambda). Tables in report section 4, effects text generated from the data with seed-noise caveats |
| System implementation and agent behaviour | `play.py` and the viewer replay the same checkpoints; behaviour summarised per action in `fig_action_profile.png` and described against the objective in report 5c; API and serialisation in `docs/API.md`; `tests/test_backend.py` compares the HTTP episode with a direct rollout |
| Discussion and analysis | Required figures: comparative rewards (Fig 3, 5), DQN TD loss and PG entropy (Fig 4), convergence (Fig 6 and table), generalisation (Fig 7), agricultural outcomes (Fig 8), resource efficiency (Fig 9). Paired statistics with Holm correction. Conclusions are phrased from the measured ordering and stated as limited to this simulator, seed count and budget |
| Video | `docs/DEMO_SCRIPT.md` gives the narration and a timed execution plan (about 3 minutes of agent execution). The video itself does not exist yet |

## Standing rules from the brief

| Rule | How it is met |
|---|---|
| Do not claim anything executes unless tested | Test suite plus a clean-copy run of `uv sync` and `uv run main.py`; the only unverified items are listed under "Needs you" |
| Separate model assumptions from evidence | Parameter provenance classes L / C / S / E in `environment/params.py`; scope paragraph in the report |
| Failures documented, nothing fabricated | `experiments/results/failures.md` is generated from `result.json` of every run; report tables read from the registry. One real failure found and fixed during development: A2C and PPO crashed with an entropy coefficient of 0 (TorchRL omits `loss_entropy`); fixed and covered by the config `a2c/entropy_0` |
| JavaScript does not simulate agronomy | The frontend only maps `serialize_state()` values to visuals; see `docs/ARCHITECTURE.md` |
