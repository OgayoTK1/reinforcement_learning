# API and WebSocket specification

Everything the viewer shows is produced by `environment.rendering.serialize_state(env)`, schema version
**1.0**. The browser never computes agronomy. It only maps these numbers to colours, heights and water levels.

## REST

| Method and path | Purpose |
|---|---|
| `GET /api/health` | `status`, `env_version`, `schema_version`, `spec_hash` of the observation spec |
| `GET /api/catalogue` | the 20 actions (id, name, description) and the scenario names |
| `GET /api/policies` | `manual`, `rule_based`, `random`, and every checkpoint found under `models/` |
| `GET /api/observation_spec` | the 65 observation names and the spec hash |
| `POST /api/episode` | body `{"policy", "scenario", "seed"}`; runs a whole episode server side and returns `{"states": [...], "summary": {...}}` |

Errors: `400` with a text reason for an unknown scenario, an unknown policy, or a checkpoint path outside the project.

## WebSocket `/ws`

Text frames carrying JSON, one connection per viewer. The server owns the environment; the client only sends intent.

Client to server:

| `type` | fields | effect |
|---|---|---|
| `start` | `policy`, `scenario`, `seed`, `speed`, `autoplay` | create a session, reset the environment with `episode_seed = seed`, send the first state, then step on a timer if the policy is not manual |
| `step` | `action` (integer, manual mode only) | apply one action and send the new state |
| `pause`, `resume` | none | stop or continue automatic stepping |
| `reset` | none | restart the same seed and scenario |
| `speed` | `value` | playback speed multiplier (about 0.9 s per week at 1.0) |

Server to client:

| `type` | fields |
|---|---|
| `hello` | `env_version`, `schema_version`, `spec_hash` |
| `state` | `data`: the state object below |
| `episode_end` | `summary` (episode metrics), `reason` (`end_of_year`, `never_sown`, `crop_failure`, `bankrupt`) |
| `error` | `message` |

Policies and seeds are reproducible: `seed` becomes the environment's `episode_seed`, so a replay shows exactly the season that
`evaluation/` scored for that seed.

## State object (`data`)

Valid scenarios: `training`, `baseline`, `severe_drought`, `late_onset`, `wet_year`, `warm_1p5`, `warm_3`, `poor_forecast`, `drought_warm`.

| Field | Type | Example |
|---|---|---|
| `schema_version` | string | 1.0 |
| `env_version` | string | 1.0 |
| `time.week` | integer | 14 |
| `time.horizon` | integer | 52 |
| `time.date` | string | 2027-06-07 |
| `time.label` | string | 07 Jun |
| `time.doy` | integer | 157 |
| `scenario.name` | string | train_draw |
| `scenario.rain_mult` | number | 0.944 |
| `scenario.temp_shift` | number | -0.243 |
| `scenario.onset_delay_weeks` | integer | 0 |
| `scenario.forecast_skill` | number | 0.481 |
| `weather.rain_week_mm` | number | 88.28 |
| `weather.rain_days` | integer | 4 |
| `weather.tmax_week` | number | 21.036 |
| `weather.daily_rain_mm` | array | [7 items] |
| `weather.daily_tmax` | array | [7 items] |
| `weather.daily_tmin` | array | [7 items] |
| `weather.forecast_next_week_mm` | number | 18.266 |
| `crop.sown` | boolean | True |
| `crop.sow_week` | integer | 2 |
| `crop.gdd` | number | 804.411 |
| `crop.stage` | string | vegetative |
| `crop.stage_index` | integer | 1 |
| `crop.harvested_fraction` | number | 0.0 |
| `crop.standing_grain_kg` | number | 0.0 |
| `plots[i].id` | integer | 0 |
| `plots[i].soil_moisture` | number | 1.0 |
| `plots[i].mineral_n` | number | 28.463 |
| `plots[i].available_p` | number | 11.674 |
| `plots[i].organic_matter` | number | 0.651 |
| `plots[i].mulch` | number | 0.0 |
| `plots[i].weeds` | number | 0.05 |
| `plots[i].faw_pressure` | number | 0.155 |
| `plots[i].disease_severity` | number | 0.095 |
| `plots[i].canopy_retained` | number | 0.841 |
| `plots[i].lai` | number | 2.706 |
| `plots[i].biomass_gm2` | number | 320.953 |
| `plots[i].grain_gm2` | number | 0.0 |
| `plots[i].stress` | number | 1.0 |
| `plots[i].alive` | boolean | True |
| `plots[i].water_capacity_mm` | number | 140.0 |
| `plots[i].slope_class` | integer | 0 |
| `resources.cash` | number | 13681.099 |
| `resources.water_m3` | number | 190.396 |
| `resources.water_capacity_m3` | number | 220.0 |
| `resources.vouchers` | number | 0.0 |
| `resources.manure_kg` | number | 1722.673 |
| `resources.labour_hours` | number | 47.924 |
| `resources.labour_capacity` | number | 56.0 |
| `storage.stock_kg` | number | 0.0 |
| `storage.bags` | number | 0.0 |
| `storage.moisture` | number | 20.0 |
| `storage.insect_damage` | number | 0.0 |
| `storage.mould_index` | number | 0.0 |
| `storage.store_type` | integer | 0 |
| `storage.store_type_name` | string | none |
| `storage.weeks_stored` | integer | 0 |
| `storage.quality` | number | 0.74 |
| `storage.reserve_kg` | number | 490.0 |
| `market.maize_bag_kes` | number | 4803.292 |
| `market.dap_bag_kes` | number | 7009.48 |
| `market.can_bag_kes` | number | 5178.646 |
| `protection.faw_protect_days` | integer | 0 |
| `protection.fung_protect_days` | integer | 0 |
| `action.requested` | string | hand_weed |
| `action.executed` | string | hand_weed |
| `action.valid` | boolean | True |
| `action.reason` | string |  |
| `action.detail` | string |  |
| `action.cash_cost` | number | 0.0 |
| `action.water_m3` | number | 0.0 |
| `action.labour_hours` | number | 40.0 |
| `action.sold_kg` | number | 0.0 |
| `action.revenue` | number | 0.0 |
| `action.harvested_kg` | number | 0.0 |
| `reward.last_components.wealth` | number | 0.0 |
| `reward.last_components.soil` | number | -0.001 |
| `reward.last_components.water` | number | -0.0 |
| `reward.last_components.labour` | number | -0.011 |
| `reward.last_components.invalid` | number | 0.0 |
| `reward.last_components.shortfall` | number | 0.0 |
| `reward.episode_return` | number | -0.184 |
| `metrics.return` | number | -0.184 |
| `metrics.harvested_kg` | number | 0.0 |
| `metrics.marketable_kg` | number | 0.0 |
| `metrics.marketable_bags` | number | 0.0 |
| `metrics.yield_kg_acre` | number | 0.0 |
| `metrics.crop_survival` | number | 1.0 |
| `metrics.crop_failure` | number | 1.0 |
| `metrics.net_return_kes` | number | -11273.546 |
| `metrics.water_used_m3` | number | 0.0 |
| `metrics.water_use_efficiency_kg_m3` | number | 0.0 |
| `metrics.soil_health` | number | 0.574 |
| `metrics.soil_change` | number | -0.002 |
| `metrics.post_harvest_loss_frac` | number | 0.0 |
| `metrics.spoilage_mould` | number | 0.0 |
| `metrics.final_cash` | number | 13681.099 |
| `metrics.stock_kg` | number | 0.0 |
| `metrics.food_reserve_met` | number | 0.0 |
| `metrics.reserve_shortfall_kg` | number | 490.0 |
| `metrics.invalid_actions` | integer | 0 |
| `metrics.reason` | string |  |
| `metrics.sown_week` | integer | 2 |
| `valid_actions` | array | [20 items] |
| `done` | boolean | False |
| `termination_reason` | null | None |

`valid_actions` is a list of 20 booleans (which actions are possible now); `done` marks the final frame. Per-plot values use the
index `i = 0..3`. `plots[i].soil_moisture` is available water as a fraction of capacity, `canopy_retained` the fraction of leaf
area not lost to stress, pests and disease, and `stress` the accumulated water and nitrogen stress.

## Actions

| id | name | description |
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
