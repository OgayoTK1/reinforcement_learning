# AgriResilience RL: Design (Njoro, Nakuru County reference farm)

*Ogayo Andrew, ALU Machine Learning Techniques II summative.*

I wrote this design before building the environment and kept it as the record of what I set out to model and why. Where the code ended up
different, the code is authoritative: `docs/OBSERVATION_SPACE.md` and `docs/API.md` are generated from it, and `environment/params.py`
holds every parameter. What changed while I implemented and calibrated it:

- The observation vector has 65 elements, not about 57 (stage, last action and storage type are one-hot).
- Storage can be chosen before harvest (a hermetic-bag purchase has to be possible while the crop is still standing). A harvest action
  takes 50 percent of the standing grain, so a full harvest is two actions.
- The mould penalty applies above a 0.1 mould index, and mould and insect rates were relaxed so that careful post-harvest handling pays.
- Soil organic matter gain per manure application was cut from 0.018 to 0.004 and mineralisation loss per kg N from 0.0012 to 0.0004.
  With the first values a PPO policy earned most of its return by spreading manure ten times a year, which no real soil would reward.
- Training episodes use seeds from 10 000 up; validation 1 to 50; test 1000 to 1199; out-of-distribution 2000 to 2049.
- Weather is synthetic and calibrated to published normals (class S in the parameter registry). I did not have a daily station record
  for Njoro, so none is used.
- The reward is written as the weekly change in wealth (cash plus 85 percent of the quality-adjusted value of stored grain) instead of
  separate cash-flow and end-of-season stock terms. It comes to the same thing, and unsold grain no longer needs a special terminal term.
- Parameters live in `environment/params.py`, not a YAML file, so each one sits next to its class tag and description.

## 0. What this document covers

1. Which real place and farming system the simulation represents.
2. The MDP: horizon, state, observations, actions, transitions, reward, termination.
3. Where every parameter comes from (literature, calibrated, synthetic, convenience).
4. How the visualization is kept faithful to a real Nakuru farm instead of looking like a generic dashboard.
5. Architecture, experiment protocol, compute plan and acceptance criteria.

Scope statement, repeated in the README and the report: this is a simulation and an algorithm comparison. The results show how four RL algorithms behave in my model, not that any policy would raise yields on a real farm.

## 1. Site and farming system

**Reference farm.** One smallholder maize farm in Njoro Sub-County, Nakuru County, Kenya. Njoro falls in the Kenya Meteorological Department (KMD) "medium ground" zone, 2000 to 2300 m above sea level, shared with Subukia, Kabazi, Bahati and Naivasha. Molo, Kuresoi and Mau Narok are the "high ground" zone above 2300 m. I chose Njoro because it is a maize and wheat area with volcanic loamy soils, it sits against the Mau escarpment (a recognisable backdrop for the 3D scene), and Egerton and KALRO-Njoro publish a lot of local agronomy.

**Crop system.** Highland hybrid maize as a sole crop on 1 acre (0.405 ha), sown in the long rains. Maize-bean intercropping is the dominant practice on Nakuru smallholdings, so leaving it out is a stated simplification, not a claim about typical practice.

**Climate facts used (verified against sources, see section 11).**
- Rainfall is bimodal. The county gives a range of about 500 to 1800 mm, with long rains March to June and short rains October to November.
- KMD describes October to December as not a major season for Nakuru and advises quick-maturing crops.
- KMD weekly forecasts for the medium zone show daily maxima in the low 20s °C and minima around 13 to 15 °C.
- Nakuru town station sits at 1901 m.

**Agronomic facts used.**
- Nakuru's main crops are maize, beans, Irish potato and wheat. The county reports maize output has been falling for about five years because of pests, disease, climate risk and value-chain problems.
- A 1998 survey in Nakuru found inorganic and organic fertilizer behave as substitutes once cropping pattern, transport cost and labour are accounted for. This is why manure is an action alongside DAP and CAN.
- Fall armyworm (FAW) is established in Kenya. Farmer-reported losses reach 47 percent, and a 2017 assessment gave 24 percent (long rains) and 26 percent (short rains).
- Northern leaf blight was found at 100 percent prevalence in a 2012 survey of Kiambu, Embu and Nakuru.

**Economic facts used (2026 prices, checked when I froze the parameters).**
- Nakuru maize rose from about KES 3,700 to KES 4,500 to 4,600 per 90 kg bag by July 2026. The NCPB floor price was KES 4,000.
- Subsidised fertilizer is KES 2,000 per 50 kg bag from September 2026. A Vihiga spot check put open-market DAP at KES 6,700 and CAN at KES 4,950, and the source warns this is a reference point, not a national price.
- Subsidised certified seed is KES 150 per kg, down from KES 300.
- Typical yields are quoted at 20 to 25 bags per acre in the press. I treat that as an upper reference for well-managed fields, not an average.
- Per-capita maize consumption is about 98 kg per year, which sets the household food reserve (5 people is roughly 490 kg, about 5.4 bags).

## 2. Why sequential RL and not a prediction model

A yield or weather predictor answers "what will happen". The farm problem is "what should I do each week given a limited tank, cash and labour". The same irrigation litre is worth more at flowering than at emergence. A top-dress applied before heavy rain leaches. A pesticide sprayed after the larvae are inside the whorl does little. Selling at harvest sacrifices later price recovery but avoids storage loss. These are delayed, resource-coupled, state-dependent decisions, so the target is a policy, not a forecast. A static model also cannot choose between conflicting objectives (yield, cash, water, soil) without an explicit decision rule.

## 3. MDP formulation

The environment is a finite-horizon **partially observable** MDP. The agent sees noisy estimates of soil moisture, nutrient status, pest, disease and weed pressure, and a probabilistic forecast, never the true values or future weather. I keep the POMDP framing honest by giving every algorithm the same observation vector, which includes the previous action and time-since-last-scout features as cheap memory. I do not use recurrent policies.

**Timing.**
- One step is one week. Inside each step the engine runs 7 daily sub-steps for weather, evapotranspiration and soil water. The daily detail matters for the water balance; the weekly decision keeps episodes short enough for 40 configurations on a CPU.
- Horizon T = 52 weeks, from the first week of March to the end of February. A normal year has sowing in weeks 2 to 6, maturity around weeks 34 to 40, harvest and drying afterwards, and sales through to week 52.
- The week index is in the observation, so the finite horizon is Markov and the last step is `terminated`, not `truncated`.

**Initial state distribution.** Fixed farm layout, 4 plots with different slope and water-holding capacity. Randomised each episode: initial soil moisture, initial soil N and P, weather seed (including a seasonal rainfall multiplier for wet and dry years), price path seed, pest immigration schedule, labour availability schedule, and the opening cash drawn from a narrow range.

**State (hidden, authoritative in Python).** Per plot: soil moisture, N pool, P pool, soil organic matter index, mulch cover, weed biomass, FAW pressure, leaf disease severity, crop thermal time, LAI, biomass, grain mass, crop health. Farm level: cash, water in pan and tank, subsidised fertilizer vouchers remaining, manure stock, labour hours available this week, harvested fraction, grain stock with moisture, mould and insect damage, storage type, market price and fertilizer price, true weather for the week.

Actions apply to the whole farm. The 4 plots differ in soil and slope, so the 3D view shows real plot-to-plot variation, but the agent cannot target a plot. That is a documented limitation.

**Observation.** 65 floats, generated from one `ObservationSpec` object in code. That object emits the documentation table (name, agricultural meaning, dtype, range and encoding, source rule, update frequency, normalisation, observable or estimated, entry into the network), so documentation cannot drift from the code. All variables enter the network the same way, as one concatenated float32 vector through the first linear layer of a shared-style MLP trunk (each algorithm sets its own widths). Groups:

| Group | Variables | Observable or estimated | Update |
|---|---|---|---|
| Calendar and stage | week/52, sown flag, weeks since sowing, stage one-hot (unsown, vegetative, flowering, grain fill, dry-down, harvested) | Directly observable (farmer sees stage) | Weekly |
| Crop condition | canopy greenness, crop health | Estimated, noise sd 0.08 | Weekly |
| Soil and water | mean and driest-plot soil moisture | Estimated, noise sd 0.07, halves if monitored | Weekly |
| Weather | rain last week, rain last 4 weeks, max temperature last week, next-week rain forecast | Gauge reading (exact) and forecast (noise set by forecast reliability) | Weekly |
| Nutrients | N-deficiency score, cumulative N and P applied, manure applied flag | Score estimated (leaf colour), inputs exact | Weekly |
| Pests and disease | FAW estimate, disease estimate, weed estimate, weeks since scouting, weeks since insecticide, weeks since fungicide | Estimated, sd 0.25 normally, 0.06 in a scouting week | Weekly |
| Resources | pan and tank level, cash/R0, labour hours available, voucher bags left, fertilizer price | Exact | Weekly |
| Market | maize price, 4-week price change | Exact (price known at the market) | Weekly |
| Harvest and storage | harvested fraction, grain stock, grain moisture estimate, storage type one-hot, weeks in storage, spoilage-sign estimate | Moisture and spoilage estimated, rest exact | Weekly |
| History | previous action one-hot | Exact | Weekly |

No future weather, no true pest or disease level, no true soil moisture, no future price and no yield-to-date oracle enters the vector. `test_observation_has_no_future_weather` checks this: two environments that differ only in hidden future weather give identical observations.

## 4. Action space (Discrete(20), shared by all four algorithms)

Invalid actions (precondition not met, or unaffordable) are executed as no-op and incur a small fixed penalty. The mask is returned in `info` for diagnostics and is not used by any learner, which keeps the comparison fair.

| ID | Action | Real practice | Precondition | Main cost | Main effect |
|---|---|---|---|---|---|
| 0 | Wait | No intervention | None | None | Time passes |
| 1 | Sow | Plant maize at rain onset | Not sown, week 0 to 8 | Seed (about 10 kg per acre), labour | Starts the crop clock; if never sown by week 8 the season is lost |
| 2 | Irrigate light (10 mm) | Bucket or hose from pan | Sown, water available | About 40 m3 water, labour | Raises soil moisture, no runoff if soil is dry |
| 3 | Irrigate medium (25 mm) | Same | Same | About 100 m3 | Same |
| 4 | Irrigate heavy (40 mm) | Same | Same | About 160 m3 | Same, runoff and leaching risk on wet soil |
| 5 | Basal DAP (50 kg) | DAP at planting | Sown, before flowering | Voucher bag, else market price | Adds N and P near the root zone |
| 6 | Top-dress CAN (25 kg) | Split nitrogen | Sown, before flowering | Voucher or market | Adds N, leaching if heavy rain follows |
| 7 | Top-dress CAN (50 kg) | Full top-dress | Same | Voucher or market | Larger N pulse |
| 8 | Apply manure | Cattle manure | Manure stock available | Labour | Slow N and P release, raises soil organic matter |
| 9 | Scout the field | Walk the rows, check whorls and leaves | Sown | 4 labour hours | Cuts observation noise on pest, disease, weeds and moisture for the week |
| 10 | Spray insecticide | FAW control | Sown, before dry-down | Chemical cost, labour | Kills larvae, efficacy fades with repeated use and late timing |
| 11 | Spray fungicide | Leaf blight and rust control | Sown, before dry-down | Chemical cost, labour | Protects leaves for about 2 weeks |
| 12 | Hand weed | Weeding hoe | Sown | Heavy labour | Removes most weed biomass, with soil disturbance |
| 13 | Mulch | Crop residue or grass | Sown | Labour | Cuts soil evaporation, decays over weeks |
| 14 | Harvest (this week's share) | Cut and carry cobs | Crop at or near maturity | Labour, hired labour if short | Moves a labour-limited fraction of standing crop to grain stock |
| 15 | Sun-dry | Drying sheet or tarpaulin | Harvested grain not dry | Labour | Lowers grain moisture, no effect if it rains |
| 16 | Store in polypropylene bags | Common practice | Dry grain | Bag cost | Cheap, exposed to weevils and moulds |
| 17 | Store in hermetic bags | Hermetic (PICS-type) bags | Dry grain | Higher bag cost | Strongly cuts insect loss and mould growth |
| 18 | Sell half the stock | Market sale | Stock above food reserve | None | Cash in at current price, quality discount if spoiled |
| 19 | Sell all surplus | Market sale | Stock above food reserve | None | Same, keeps the household reserve |

A household food reserve of about 490 kg stays in storage. Selling below it is blocked, and ending the year below it is penalised at the market purchase price plus a margin.

## 5. Transition dynamics (what is modelled, in order)

1. **Weather (daily).** A two-state wet/dry Markov chain with gamma-distributed wet-day amounts, monthly parameters tuned to a bimodal climatology, a seasonal rainfall multiplier per year, and Tmax/Tmin with seasonal and day-to-day variation. Forecast for the next week mixes true rainfall with climatology according to a forecast-reliability parameter.
2. **Reference evapotranspiration.** Hargreaves-Samani from Tmax, Tmin and latitude (the generator produces no humidity or wind, so Penman-Monteith is not available).
3. **Soil water (per plot, daily).** FAO-56 single crop coefficient water balance with root-zone depletion, stress factor Ks, SCS curve-number runoff depending on slope and cover, deep percolation above field capacity, mulch reducing soil evaporation.
4. **Crop development.** Thermal time with a cool-highland base temperature gives stage progress. Biomass accumulates through radiation-use efficiency times intercepted light, scaled by water, N, P, weed, pest and disease factors. Grain fill uses a harvest index reduced by stress at flowering and grain fill (the FAO guidance is that water deficit around flowering and yield formation hurts yield most, while deficit at ripening barely matters).
5. **Nutrients.** N and P pools with mineralisation (temperature and moisture dependent), crop uptake, leaching on heavy drainage, and fertilizer and manure inputs. Soil organic matter drifts down with nutrient mining and erosion and up with manure and mulch.
6. **Pests and disease.** FAW pressure grows with warm days and susceptible stages, with random immigration events, natural decline, and insecticide kill. Leaf disease follows a logistic infection driven by cool, wet, humid spells, with a protective fungicide window. Both reduce leaf area and grain yield up to a capped loss.
7. **Weeds.** Biomass growth with rain and warmth, competition strongest in the early critical period, removed by weeding.
8. **Harvest and grain.** Standing crop incurs field losses while it waits (lodging, ear rot after rain, birds). Harvested grain starts at high moisture and dries only on dry days. Storing above about 13.5 percent moisture raises mould and aflatoxin risk with storage time and humidity. PP bags allow weevil build-up that grows with temperature and time. Hermetic bags strongly limit it.
9. **Market.** Mean-reverting log-price with a seasonal pattern, low around harvest and recovering afterwards, with a shock term. Fertilizer open-market price has its own path.
10. **Labour.** Weekly family labour hours vary by a random availability factor. Actions that exceed it use hired labour at a daily wage.

All of these are in `environment/dynamics.py` as pure functions of (state, action, weather, rng), so each can be unit-tested alone.

## 6. Reward function

Everything is expressed in KES and divided by a reference gross revenue R0 = KES 88,000 (22 bags times KES 4,000), so rewards are dimensionless and comparable across seeds. Per step:

`r_t = (net cash flow_t)/R0 + w_s * (change in soil health index) - (shadow price of water drawn)/R0 - invalid-action penalty`

and at the final step:

`r_T += value of unsold stock net of quality discount/R0 - food-reserve shortfall cost/R0 - failure penalty`

Reasoning for each term:
- **Net cash flow** carries marketable yield, price, input cost, labour, storage and post-harvest loss, because spoiled grain sells at a discount or not at all. This keeps the main signal tied to the real economic outcome and makes yield, price and loss interact.
- **Soil health change** (organic matter plus nutrient balance) is the only way a short season can register long-term degradation. Weight `w_s` is set so mining the soil for one season cannot pay for itself in cash terms. This is a synthetic choice and is reported as such.
- **Water shadow price** puts drawn water in the same unit as everything else, using the cost of refilling the pan, so there is no arbitrary water weight.
- **Food-reserve shortfall** reflects that a household cannot sell its whole crop.
- **Crop failure** is not given a separate bonus or penalty beyond the lost revenue and the reserve shortfall. I do not add shaping on biomass, greenness or intermediate growth, because those invite the agent to farm the proxy.

**Reward hacking tests.** A scripted-policy suite (`tests/test_environment.py`) checks expected orderings: never-act, always-irrigate-heavy, spray-every-week, harvest-on-first-possible-day, sell-everything-at-harvest, hoard-and-never-sell, and a sensible agronomic schedule. The tests assert that (a) the sensible schedule beats all extremes in expectation, (b) pesticide spam loses to scouting-driven spraying, (c) over-irrigating loses to timed irrigation in a drought year, (d) hoarding loses once spoilage and the horizon are counted, (e) reward sums match recomputed cash flow exactly.

## 7. Termination

- `terminated = True` at week 52 (end of marketing year).
- `terminated = True` early on bankruptcy (cash below minus KES 20,000, because farms cannot borrow without limit), on total crop loss before harvest (crop health zero), or if the crop is never sown by week 8.
- No `truncated`, because the horizon is observable and part of the MDP.

## 8. Parameter provenance register

Each parameter in `environment/params.py` carries a class tag: **L** literature-informed, **C** calibrated to published Nakuru/Kenya numbers, **S** synthetic, **E** chosen for experimental convenience. The status at submission:

| Parameter group | Class | Basis | Remaining gap |
|---|---|---|---|
| Kc stage values (mid 1.05 to 1.2, late 0.8 to 0.9, harvest 0.55 to 0.6) | L | FAO maize guidance | None; FAO-56 cited in the report |
| Hargreaves-Samani ET0, SCS curve number, thermal-time form | L | Standard methods | Standard published coefficients used |
| Season length 30 to 34 weeks, thermal-time thresholds | C | Highland hybrids at 2000 m+ mature later than lowland maize; FAO notes cool temperatures lengthen maturity | A KALRO variety description would give a primary figure |
| Monthly rainfall and temperature normals for Njoro | C | KMD forecasts and county pages give ranges only | Calibrated to published normals; station or CHIRPS data would be better |
| FAW and disease loss caps (up to 47 percent) | L | Farmer-reported Kenyan losses | Use as an upper cap, not a mean |
| Post-harvest loss ranges, moisture threshold 13.5 percent, aflatoxin threshold 10 ppb | L | Kenyan storage studies, KEBS limit cited in the Nakuru hygiene study | Exact PP versus hermetic loss figures would tighten the ranges |
| Maize price level and fertilizer prices | C | 2026 press and government figures | Fixed at 2026 levels |
| Seed rate, fertilizer rates per application | L/S | Common recommendations (1 bag DAP, 1 or 2 top-dress bags per acre) | Not yet checked against KALRO guidelines |
| Pest immigration rates, disease infection rates, weed growth | S | No usable weekly Njoro data | Documented as synthetic, tested for plausible ranges |
| Labour hours, wage, chemical cost, bag cost, pan size | S | Reasonable local values, unverified | Documented as synthetic |
| Reward weights `w_s`, shadow price, failure penalty | E | Chosen to rule out degenerate policies | Documented as convenience choices |

No S or E value is described as empirical anywhere in the project.

## 9. Visualization: keeping it looking like a real Nakuru farm

I wanted the viewer to look like a real smallholding and a farmer's record book, not a generic dashboard, so I set myself concrete rules.

**Rules against the generic look.**
- No glowing neon accents, glass-morphism cards, gradient headers, floating emoji icons, or a perfect grid of identical cubes.
- No flat green plane with cartoon lollipop trees.
- No pie-chart-heavy dashboard layout.

**Landscape and objects (all procedural, no downloaded models or textures, so the viewer runs offline).**
- A gently sloping farm with an irregular parcel outline, not a rectangle. The Mau escarpment forest edge on the horizon. A eucalyptus woodlot and a cypress windbreak on the boundary, Napier grass along the contour strip, Grevillea trees, kikuyu grass at the margins.
- A homestead with a corrugated-iron (mabati) roof on timber and stone walls, a separate cow shed (manure comes from here), a plastic water tank on a stand, a lined farm pan, a timber maize store, a drying sheet on the yard, stacked 90 kg bags. Hermetic bags look different from polypropylene bags so storage choice is legible.
- Maize drawn as instanced plants with per-plant variation in height, lean and leaf droop, planted at standard row spacing, and with stage-correct shapes (whorl, tassel, silk, cob, senescence). Stress is shown from true state: leaf rolling under drought, yellowing from N deficiency, ragged leaf windows from FAW, tan lesions from leaf blight, weeds between rows, mulch layer on the soil.
- Soil colour darkens with true moisture per plot. The 4 plots look different because their simulated soil and slope differ.
- Sun angle follows the day of year at the equator. Cloud cover and rain particles follow simulated radiation and rainfall. Palette is muted with a soft Rift Valley haze.

**Interface.**
- Panels use a field-record-book look: paper tones, dark ink text, thin rules, tabular numerals, no card shadows. Units are KES and 90 kg bags, dates are shown as calendar weeks ("Wk 6, 7 Apr"), with the crop calendar as a timeline strip.
- The action log reads like an extension officer's notes ("Wk 6: top-dressed 25 kg CAN, KES 1,240"), with the agent's top alternatives beside it.
- Controls required by the assignment: algorithm picker, checkpoint loader, play and pause, reset, speed, current action, state inspector, cumulative metrics.

**Fidelity rule.** The browser only draws serialized backend state. It has no agronomic equations and no random values for crop or weather.

**Verification.** I checked the scene in the browser against these rules. The screenshots in `assets/` (vegetative stage, grain fill, and grain fill under severe drought) are taken from real environment states and appear as Figure 1 of the report.

## 10. Architecture, stack, protocol

**Pinned stack.** Python 3.11 with uv, locked in `uv.lock`: torch 2.15.0, torchrl 0.15.0, tensordict 0.15.6, gymnasium 1.4.0, numpy 2.4.6, scipy 1.17.1, pandas 3.1.1, matplotlib 3.11.2, fastapi 0.165.0, uvicorn 0.54.0, websockets 17.4, pytest 9.2.0. Development and training ran on 2 CPU cores with 7 GB RAM and no GPU. A fresh clone installs with `uv sync`.

**Layers.** (1) `environment/` pure Python Gymnasium env, no browser dependency. (2) `agents/` and `training/` TorchRL and PyTorch only. (3) `evaluation/` and `experiments/` registry, pandas, matplotlib. (4) `backend/` FastAPI plus WebSocket. (5) `frontend/` Three.js. Dependencies point downward only, so rendering and API code cannot touch training.

**Gymnasium and TorchRL.** `AgriResilienceEnv` is a plain Gymnasium environment and the single source of truth. TorchRL wraps it with `GymWrapper`, which gives TensorDict specs, collectors and replay buffers. Losses: `DQNLoss` with a target updater for DQN, `ClipPPOLoss` plus `GAE` for PPO, `A2CLoss` plus `GAE` for A2C. REINFORCE is my own Monte-Carlo loss on a TorchRL `ProbabilisticActor` with a categorical head and full-episode discounted returns. TorchRL's own REINFORCE loss uses a learned critic, which would make it an actor-critic method, so I did not use it. An optional non-learned running-mean baseline is a hyperparameter and is documented as variance reduction.

**Checkpoints and playback.** Each checkpoint is a weights file plus a JSON sidecar with algorithm, network configuration, observation-spec hash, environment version, seed and training metrics. The loader refuses a checkpoint whose observation-spec hash does not match the live environment. Playback loads it in eval mode, runs the greedy action for every algorithm (sampled actions are an optional ablation), and the same code path is used by `play.py`, the evaluator and the FastAPI backend. Training never imports the backend.

**Experiment protocol.**
- 10 configurations per algorithm, 40 in total: a default, then nine configurations that each change one factor from it, covering every hyperparameter the assignment lists, so each difference can be traced to one change.
- The pipeline is built for 3 training seeds per configuration (120 runs) with an equal environment-step budget per run. I measured environment throughput first and set the budget from that.
- Three disjoint scenario sets: training (random seeds), validation (50 episodes, used only to pick the best configuration), test (200 episodes plus the shifted-weather families, touched once per model). The shifted families are delayed onset, severe drought, wet year, +1.5 and +3 °C, and degraded forecast reliability.
- Baselines: random policy and a transparent rule-based policy that only uses the same observation vector.
- Metrics: mean and sd of reward, marketable yield, survival rate, net return, water use and water-use efficiency, soil health at the end, post-harvest loss, crop-failure rate, training stability across seeds, and out-of-distribution drop. Bootstrap confidence intervals and a rank-based test across seeds. Failed or diverged runs stay in the registry with their status.
- Every run gets a unique ID, configuration file, seed, history CSV, checkpoint, evaluation JSON and execution metadata.

**Compute.** On 2 CPU cores one 100,000-step run takes 4 to 6 minutes, so the full 120-run matrix is 8 to 10 CPU-hours. The experiment runner is resumable and skips finished runs, so it can be run in chunks. The results in the report come from 1 seed per configuration at 40,000 steps (40 runs, none failed), and the report says so and treats close rankings as tentative. Every number in it comes from a run I actually completed.

## 11. Sources checked

- Nakuru County Government, agricultural potential and rainfall: https://nakuru.go.ke/?p=2238
- Nakuru County agriculture at a glance: https://nakuru.go.ke/?p=13638
- KMD Nakuru OND 2025 seasonal forecast: https://meteo.go.ke/documents/1360/Nakuru_County_OND_2025_Seasonal_forecast.pdf
- KMD Nakuru weekly forecasts, climatic zones: https://meteo.go.ke/documents/1853/Nakuru_County_weekly_weather_forecast_valid_18th_to_24th_November_2025.pdf and https://meteo.go.ke/documents/2982/Nakuru_County_weekly_weather_forecast_valid_17th_to_23rd_March_2026.pdf
- Nakuru soil fertility management survey (ILRI): https://www.ilri.org/knowledge/publications/soil-fertility-management-small-farms-africa-evidence-nakuru-district-kenya
- Maize diseases in Kiambu, Embu and Nakuru: https://www.sciencegate.app/document/10.5539/jas.v11n4p47
- Fall armyworm in Kenya: https://horizon.documentation.ird.fr/exl-doc/pleins_textes/2022-05/010084572.pdf , https://core.ac.uk/download/340226344.pdf , https://www.downtoearth.org.in/news/fall-armyworm-attack-desperation-pushes-kenya-farmers-to-danger-63525
- Post-harvest losses and hermetic storage: https://cimmyt.org/?p=149786 , https://pmc.ncbi.nlm.nih.gov/articles/PMC7001978 , https://experts.arizona.edu/en/publications/comparative-effects-of-hermetic-and-traditional-storage-devices-o/
- Maize storage hygiene in Nakuru County (Makinya 2022): https://agris.fao.org/search/zh/records/68010f6ddd564a9a8282fccc
- FAO maize crop information and Kc: https://www.fao.org/land-water/databases-and-software/crop-information/maize/ru/ and https://www.fao.org/land-water/databases-and-software/crop-information/maize/fr/
- Fertilizer and seed prices: https://www.standardmedia.co.ke/national/article/2001556037/ruto-cuts-fertiliser-price-to-sh2000-in-new-relief-for-farmers , https://businessradar.co.ke/fertilizer-prices-in-kenya/
- Nakuru maize price, July 2026: https://nation.africa/kenya/business/maize-prices-jump-as-poor-harvest-shrinks-supply-5535220
- Per-capita maize consumption: https://core.ac.uk/download/234655507.pdf

## 12. Decisions taken

1. **Site.** Njoro with sole-crop maize as the reference farm, rather than Subukia or Bahati, for the reasons in section 1.
2. **Time and actions.** Weekly steps, a 52-week horizon, farm-level actions and the 20-action space above.
3. **Weather.** Synthetic, calibrated to published normals, because I had no daily Njoro station record. This is the biggest limitation of the environment.
4. **Training budget.** The pipeline supports the full 3-seed, 100,000-step matrix; the submitted results use 1 seed at 40,000 steps to fit the compute I had.
5. **Visual reference.** The farm scene is built from the published descriptions of Nakuru smallholdings in section 9, not from photos of a specific farm.
