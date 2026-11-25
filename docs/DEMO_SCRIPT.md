# Demo narration and execution plan (5 minutes, about 3 minutes of agent execution)

Assignment 2.4 asks for 5 minutes with the file execution inside three quarters of that time. The report template says "3 minutes max".
If your instructor wants the shorter one, keep sections 1, 3 and 5 and cut section 2 and the second scenario.

Set up before recording: share the **entire screen**, camera on, VS Code open on the repository, a terminal at the project root, a browser tab
ready at `http://127.0.0.1:8000`, `uv sync` already finished. Do one dry run so the browser is not seeing WebGL for the first time.

## Timeline

| Time | On screen | Say (first person, adapt to your own words) |
|---|---|---|
| 0:00 to 0:35 | VS Code, repo tree (`environment/`, `agents/`, `training/`, `models/`) | "I am Ogayo Andrew. My project is AgriResilience RL. Smallholder maize farmers around Njoro in Nakuru County decide every week whether to plant, irrigate, fertilise, spray, harvest, dry, store or sell, with limited cash, labour and water and with weather they cannot predict. Mistakes show up months later. That is why I model it as sequential decision making." |
| 0:35 to 1:10 | `environment/rewards.py`, then `docs/OBSERVATION_SPACE.md` | "The agent is the farm household. It picks one of 20 discrete actions each week for 52 weeks. It sees noisy observations such as rain gauge, soil moisture estimate and market price, not the true state. The reward is the change in household wealth, with penalties for mining the soil, using water, using unpaid family labour, trying impossible actions, and ending the year without a 490 kg food reserve. The objective is to maximise the season's return." |
| 1:10 to 1:25 | Terminal: `uv run pytest -q` (let it start, cut or let it run) | "The environment has tests for determinism, persistence of action effects, and reward hacking." |
| 1:25 to 2:25 | Terminal: `uv run play.py --policy models/pg/ppo.pt --seed 1000` (replace with the best agent) and the browser at the same time with the same policy and seed | "This is my best agent, `<name>`, replaying an unseen season. The terminal prints one line per week; the browser shows the same season from the environment's state. Here it sows after the rains start, applies fertiliser, scouts, sprays when armyworm pressure rises, harvests, dries the grain, and sells while keeping the food reserve." (Describe what you actually see; do not read this line if the agent does something else.) |
| 2:25 to 3:25 | Browser: switch scenario to `severe_drought`, same seed; then the rule-based policy on the same season | "Same season, severe drought. I compare the agent with the extension-calendar rule policy. Both face identical weather." |
| 3:25 to 4:10 | `docs/report/figures/fig_learning_curves.png`, `fig_generalization.png` | "Across 200 unseen seasons the agent scored `<value>`; the rule policy `<value>`; random `<value>`. In out-of-distribution scenarios it `<held up / dropped>`. I tuned each algorithm with 10 configurations and picked on validation seasons only." |
| 4:10 to 5:00 | Report conclusion page | "Findings are for my simulator only: the weather is synthetic and several parameters are assumptions. With more time I would run more seeds and longer training, add a recurrent policy, and validate the water and pest modules against farm records." |

## Checklist while recording

* Whole screen visible, camera on, no notifications, terminal font large enough to read.
* Show yourself navigating directories and running the Python file (`ls`, `uv run play.py ...`), not a pasted result.
* Run the agent for roughly three minutes in total. At speed "normal" one season takes about 47 seconds in the browser, so two seasons plus the comparison fits.
* Say the objective, the reward structure and the agent's behaviour explicitly; the rubric scores those.
* Fill the `<...>` values from `docs/report/test_summary_ci.csv` for the checkpoint you are demonstrating.
