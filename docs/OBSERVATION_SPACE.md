# Observation space (generated from environment/observation.py)

Each row is one element of the float32 observation vector, in order. The agent sees these estimates, never the hidden state.

| Observation | Meaning | Source | Encoding / type | Range | Update | Normalisation | Observed or estimated |
|---|---|---|---|---|---|---|---|
| week_frac | Week of the marketing year | Farm calendar | float32 | [0, 1] | weekly | week / 52 | observable |
| sown | Maize has been sown | Farmer's own record | binary | {0, 1} | weekly | none | observable |
| weeks_since_sowing | Crop age | Farmer's own record | float32 | [0, 1.2] | weekly | weeks / 40, clipped | observable |
| stage_unsown | Crop stage is unsown | Visual inspection of the field | binary | {0, 1} | weekly | one-hot | observable |
| stage_vegetative | Crop stage is vegetative | Visual inspection of the field | binary | {0, 1} | weekly | one-hot | observable |
| stage_flowering | Crop stage is flowering | Visual inspection of the field | binary | {0, 1} | weekly | one-hot | observable |
| stage_grain_fill | Crop stage is grain fill | Visual inspection of the field | binary | {0, 1} | weekly | one-hot | observable |
| stage_dry_down | Crop stage is dry down | Visual inspection of the field | binary | {0, 1} | weekly | one-hot | observable |
| stage_harvested | Crop stage is harvested | Visual inspection of the field | binary | {0, 1} | weekly | one-hot | observable |
| canopy_greenness | Retained green canopy, mean of plots | Visual scoring of leaves | float32 | [0, 1.2] | weekly | fraction + N(0, 0.06) | estimated |
| crop_health | Overall plant health (leaf and stress) | Visual scoring | float32 | [0, 1.2] | weekly | fraction + N(0, 0.06) | estimated |
| soil_moisture_mean | Relative available soil water, mean of plots | Finger test or low-cost soil moisture probe | float32 | [-0.2, 1.3] | weekly | fraction + N(0, 0.10), sd x0.3 if scouted | estimated |
| soil_moisture_min | Relative available soil water, driest plot | Finger test or low-cost soil moisture probe | float32 | [-0.2, 1.3] | weekly | fraction + N(0, 0.10), sd x0.3 if scouted | estimated |
| rain_last_week | Rain gauge total, last week | Farm rain gauge | float32 | [0, 1.5] | weekly | mm / 60, clipped | observable |
| rain_last_4_weeks | Rain gauge total, last four weeks | Farm rain gauge | float32 | [0, 1.5] | weekly | mm / 200, clipped | observable |
| tmax_last_week | Mean daily maximum, last week | Max-min thermometer | float32 | [-1, 1.5] | weekly | (C - 20) / 10 | observable |
| rain_forecast_next_week | Forecast rainfall, next week | County forecast (KMD) | float32 | [0, 1.5] | weekly | mm / 60, clipped; error grows as forecast skill falls | estimated |
| n_deficiency | Nitrogen deficiency score | Leaf colour chart | float32 | [0, 1] | weekly | 1 - N factor + N(0, 0.08) | estimated |
| n_applied | Cumulative fertilizer N applied | Farmer's input record | float32 | [0, 1.5] | weekly | kg N / 60 | observable |
| p_applied | Cumulative fertilizer P applied | Farmer's input record | float32 | [0, 1.5] | weekly | kg P / 20 | observable |
| manure_applications | Manure applications so far | Farmer's input record | float32 | [0, 1.5] | weekly | count / 3 | observable |
| faw_estimate | Fall armyworm pressure estimate | Scouting (whorl damage counts) | float32 | [0, 1] | weekly | fraction + N(0, 0.15), sd x0.3 if scouted | estimated |
| disease_estimate | Leaf blight severity estimate | Scouting (leaf lesion scoring) | float32 | [0, 1] | weekly | fraction + N(0, 0.15), sd x0.3 if scouted | estimated |
| weed_estimate | Weed cover estimate | Scouting (weed cover scoring) | float32 | [0, 1] | weekly | fraction + N(0, 0.15), sd x0.3 if scouted | estimated |
| weeks_since_scout | Weeks since the last scouting walk | Farmer's own record | float32 | [0, 1] | weekly | weeks / 8, clipped | observable |
| weeks_since_insecticide | Weeks since the last insecticide spray | Farmer's input record | float32 | [0, 1] | weekly | weeks / 8, clipped | observable |
| weeks_since_fungicide | Weeks since the last fungicide spray | Farmer's input record | float32 | [0, 1] | weekly | weeks / 8, clipped | observable |
| water_level | Farm pan water level | Staff gauge in the pan | float32 | [0, 1] | weekly | m3 / 220 | observable |
| cash | Cash on hand | Cash book / mobile money balance | float32 | [-1, 2] | weekly | KES / 88,000, clipped | observable |
| labour_available | Family labour hours available this week | Household roster | float32 | [0, 1] | weekly | hours / 56 | observable |
| vouchers | Subsidised fertilizer bags left on the e-voucher | NCPB voucher record | float32 | [0, 1] | weekly | bags / 3 | observable |
| fertilizer_price | Open-market DAP price | Agro-dealer quote | float32 | [0.5, 1.5] | weekly | KES per bag / 6,700 | observable |
| maize_price | Maize price per 90 kg bag | Market price enquiry (trader or SMS) | float32 | [0.5, 1.5] | weekly | KES / 4,200 | observable |
| price_change_4wk | Four-week change in maize price | Market price enquiry | float32 | [-1, 1] | weekly | log ratio / 0.3, clipped | observable |
| harvest_readiness | Maturity indicator (husk and kernel colour) | Visual inspection of cobs | float32 | [0, 1.2] | weekly | GDD / 2000 | estimated |
| harvested_fraction | Share of the crop already cut | Farmer's own record | float32 | [0, 1] | weekly | fraction | observable |
| grain_stock | Grain in store or on the drying yard | Bag count | float32 | [0, 1.5] | weekly | kg / 2500, clipped | observable |
| grain_moisture | Grain moisture | Hand-held grain moisture meter | float32 | [0, 2] | weekly | (percent - 10) / 20 + N(0, 0.075) | estimated |
| store_none | Grain kept in none storage | Farmer's own record | binary | {0, 1} | weekly | one-hot | observable |
| store_polypropylene | Grain kept in polypropylene storage | Farmer's own record | binary | {0, 1} | weekly | one-hot | observable |
| store_hermetic | Grain kept in hermetic storage | Farmer's own record | binary | {0, 1} | weekly | one-hot | observable |
| weeks_in_store | Weeks the grain has been stored | Farmer's own record | float32 | [0, 1.5] | weekly | weeks / 20 | observable |
| spoilage_signs | Visible mould and insect damage | Visual inspection of grain | float32 | [0, 1] | weekly | mean(insect, mould) + N(0, 0.08) | estimated |
| mulch_cover | Mulch cover on the soil | Visual inspection | float32 | [0, 1] | weekly | fraction | observable |
| scouted_last_week | The farmer scouted last week | Farmer's own record | binary | {0, 1} | weekly | none | observable |
| prev_action_wait | Last executed action was wait | Farmer's own record | binary | {0, 1} | weekly | one-hot | observable |
| prev_action_sow | Last executed action was sow | Farmer's own record | binary | {0, 1} | weekly | one-hot | observable |
| prev_action_irrigate_10mm | Last executed action was irrigate_10mm | Farmer's own record | binary | {0, 1} | weekly | one-hot | observable |
| prev_action_irrigate_25mm | Last executed action was irrigate_25mm | Farmer's own record | binary | {0, 1} | weekly | one-hot | observable |
| prev_action_irrigate_40mm | Last executed action was irrigate_40mm | Farmer's own record | binary | {0, 1} | weekly | one-hot | observable |
| prev_action_basal_dap_50kg | Last executed action was basal_dap_50kg | Farmer's own record | binary | {0, 1} | weekly | one-hot | observable |
| prev_action_topdress_can_25kg | Last executed action was topdress_can_25kg | Farmer's own record | binary | {0, 1} | weekly | one-hot | observable |
| prev_action_topdress_can_50kg | Last executed action was topdress_can_50kg | Farmer's own record | binary | {0, 1} | weekly | one-hot | observable |
| prev_action_apply_manure | Last executed action was apply_manure | Farmer's own record | binary | {0, 1} | weekly | one-hot | observable |
| prev_action_scout_field | Last executed action was scout_field | Farmer's own record | binary | {0, 1} | weekly | one-hot | observable |
| prev_action_spray_insecticide | Last executed action was spray_insecticide | Farmer's own record | binary | {0, 1} | weekly | one-hot | observable |
| prev_action_spray_fungicide | Last executed action was spray_fungicide | Farmer's own record | binary | {0, 1} | weekly | one-hot | observable |
| prev_action_hand_weed | Last executed action was hand_weed | Farmer's own record | binary | {0, 1} | weekly | one-hot | observable |
| prev_action_mulch | Last executed action was mulch | Farmer's own record | binary | {0, 1} | weekly | one-hot | observable |
| prev_action_harvest | Last executed action was harvest | Farmer's own record | binary | {0, 1} | weekly | one-hot | observable |
| prev_action_sun_dry | Last executed action was sun_dry | Farmer's own record | binary | {0, 1} | weekly | one-hot | observable |
| prev_action_store_pp_bags | Last executed action was store_pp_bags | Farmer's own record | binary | {0, 1} | weekly | one-hot | observable |
| prev_action_store_hermetic_bags | Last executed action was store_hermetic_bags | Farmer's own record | binary | {0, 1} | weekly | one-hot | observable |
| prev_action_sell_half_surplus | Last executed action was sell_half_surplus | Farmer's own record | binary | {0, 1} | weekly | one-hot | observable |
| prev_action_sell_all_surplus | Last executed action was sell_all_surplus | Farmer's own record | binary | {0, 1} | weekly | one-hot | observable |
