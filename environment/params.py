"""Simulation parameters with provenance.

Every parameter is registered with a class tag:
  L  literature-informed (standard method or published figure)
  C  calibrated to published Kenyan / Nakuru figures or to a stated target
  S  synthetic (plausible value chosen by the author, no empirical source)
  E  chosen for experimental convenience

The tags are printed into the README and the report. No S or E value is
presented as an empirical fact.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

ENV_VERSION = "1.0"

PROVENANCE: dict[str, tuple[str, str]] = {}
_vals: dict[str, object] = {}


def _r(name: str, value: object, cls: str, note: str = "") -> None:
    assert cls in {"L", "C", "S", "E"}, cls
    _vals[name] = value
    PROVENANCE[name] = (cls, note)


# ---------------------------------------------------------------- time
_r("horizon_weeks", 52, "E", "One marketing year, first week of March to end of February")
_r("sow_deadline_week", 8, "C", "Long rains sowing window, March to April (county rainfall calendar)")
_r("start_doy", 59, "C", "1 March (0-based day of year)")
_r("plots", 4, "E", "Four plots with different soil and slope")
_r("acre_m2", 4046.86, "L", "Definition of an acre")
_r("area_plot_m2", 4046.86 / 4, "L", "Equal plot areas")

# ---------------------------------------------------------------- climate (Njoro, 2000 to 2300 m)
_r("rain_month_mm", np.array([40, 40, 85, 140, 120, 65, 70, 90, 65, 80, 100, 55], float), "S",
   "Annual about 950 mm, bimodal. Consistent with county range (500 to 1800 mm) and season timing; "
   "no station series was available, replace with station normals when supplied")
_r("wet_day_mm", 9.0, "S", "Mean wet-day depth")
_r("wet_shape", 0.8, "S", "Gamma shape of wet-day depth")
_r("rain_persist", 0.30, "S", "Wet-day persistence in the Markov chain")
_r("month_anom_sd", 0.25, "S", "Log-sd of monthly rainfall anomaly")
_r("tmax_month", np.array([24.0, 24.5, 24.0, 22.5, 21.5, 20.5, 19.8, 20.3, 21.8, 22.8, 22.5, 23.0], float), "C",
   "Medium-ground zone daily maxima in the low 20s (KMD weekly forecasts)")
_r("tmin_month", np.array([12.0, 12.3, 13.0, 13.5, 13.0, 11.8, 11.0, 11.2, 11.5, 12.5, 13.0, 12.5], float), "C",
   "Daily minima around 13 to 15 in forecasts; slightly lower mean used")
_r("temp_noise_sd", 1.6, "S", "Day-to-day temperature noise")
_r("temp_ar", 0.6, "S", "Day-to-day temperature persistence")
_r("rain_day_tmax_drop", 2.0, "S", "Cloud cooling on rain days")
_r("ra_mj", 36.5, "L", "Equatorial extraterrestrial radiation, MJ m-2 d-1")
_r("ra_amp", 1.2, "L", "Equinox maxima amplitude")
_r("hargreaves_k", 0.0023, "L", "Hargreaves and Samani coefficient")
_r("rs_k", 0.16, "L", "Hargreaves radiation coefficient, interior sites")
_r("rs_rain", 0.65, "S", "Radiation reduction on rain days")

# ---------------------------------------------------------------- crop (highland hybrid maize)
_r("tbase", 8.0, "S", "Cool-highland base temperature")
_r("tcap", 30.0, "L", "Upper cut-off in growing degree days")
_r("gdd_emerge", 90.0, "S", "Emergence")
_r("gdd_silk", 1050.0, "C", "Silking, tuned so sowing to maturity is about 30 to 33 weeks at Njoro temperatures")
_r("gdd_mat", 2000.0, "C", "Physiological maturity")
_r("gdd_end", 2300.0, "S", "End of growth, field dry-down complete")
_r("gdd_harvest_ok", 1700.0, "S", "Earliest harvest the farmer may attempt")
_r("lai_gdd", np.array([0, 90, 300, 700, 1050, 1500, 2000, 2300], float), "S", "Potential LAI curve nodes")
_r("lai_val", np.array([0.0, 0.05, 0.6, 2.8, 4.2, 3.6, 1.8, 0.8], float), "S", "Potential LAI curve values")
_r("k_ext", 0.65, "L", "Light extinction coefficient for maize")
_r("rue", 0.58, "C", "Radiation-use efficiency g DM per MJ; calibrated so full management gives about 5 t/ha")
_r("g2kgac", 4.047, "L", "g m-2 to kg per acre")
_r("n_cont", 0.011, "S", "N per unit biomass")
_r("p_cont", 0.0018, "S", "P per unit biomass")
_r("n_crit", 9.0, "C", "Mineral N (kg/acre) giving no N stress; calibrated against unfertilised yield")
_r("p_crit", 3.0, "S", "Available P (kg/acre) giving no P stress")
_r("stress_floor", 0.12, "S", "Minimum nutrient factor")
_r("partition", 0.70, "S", "Share of post-silking growth sent to grain")
_r("remob", 0.25, "S", "Share of vegetative biomass remobilisable to grain")
_r("sink_max", 520.0, "C", "Maximum grain sink g/m2 (calibration)")
_r("kernel_win", 150.0, "S", "Half-width of the kernel-set window in GDD")
_r("harvest_index_cap", 0.55, "L", "Upper bound on grain share of total biomass")
_r("weed_comp", 0.45, "S", "Maximum growth loss from weeds in the critical period")
_r("gdd_weed_crit", 700.0, "S", "End of weed-critical period")
_r("p_depl", 0.55, "L", "FAO-56 depletion fraction for maize")
_r("kc_gdd", np.array([0, 250, 800, 1500, 2000, 2300], float), "L", "Kc nodes (FAO maize stages)")
_r("kc_val", np.array([0.40, 0.40, 1.15, 1.15, 0.80, 0.55], float), "L", "Kc 0.4 initial, 1.05-1.2 mid, 0.8 late, 0.55 harvest (FAO)")
_r("kc_bare", 0.35, "S", "Bare-soil coefficient before sowing")
_r("mulch_kc", 0.20, "S", "Mulch reduction of soil evaporation")
_r("mulch_decay", 0.015, "S", "Daily mulch decay")

# ---------------------------------------------------------------- soil
_r("taw", np.array([140.0, 120.0, 100.0, 85.0]), "S", "Total available water in root zone, mm, loam on volcanic parent material")
_r("cn", np.array([72.0, 75.0, 78.0, 82.0]), "L", "SCS curve numbers by slope position")
_r("cn_cover", 8.0, "S", "Curve-number reduction at full cover")
_r("slope_k", np.array([0.6, 0.9, 1.2, 1.5]), "S", "Erosion multiplier by slope")
_r("kmin", 0.17, "C", "Mineralisation kg N / acre / day per unit SOM; calibrated with n_crit")
_r("som_per_n", 0.0004, "S", "SOM index lost per kg N mineralised")
_r("erosion_k", 0.00020, "S", "SOM index lost per mm runoff on bare soil")
_r("leach_k", 0.0022, "S", "Leaching fraction per mm drainage")
_r("p_release", 0.004, "S", "Daily release of available P")

# ---------------------------------------------------------------- pests, disease, weeds
_r("faw_g", 0.10, "S", "FAW daily growth factor")
_r("faw_mort", 0.045, "S", "Natural enemy mortality")
_r("faw_leaf", 0.030, "S", "Leaf loss per unit pressure per day")
_r("faw_imm_p", 0.30, "S", "Weekly probability of a moth flight")
_r("faw_imm_amt", (0.03, 0.12), "S", "Pressure added by a flight")
_r("faw_loss_cap", 0.47, "L", "Observed upper range of farmer-reported FAW yield loss in Kenya")
_r("dis_r", 0.14, "S", "Leaf blight infection rate")
_r("dis_seed", 0.0010, "S", "Background spore load; blight is present on all sampled fields")
_r("dis_topt", 21.0, "S", "Favourable mean temperature")
_r("dis_width", 7.0, "S", "Width of temperature response")
_r("dis_leaf", 0.016, "S", "Leaf loss per unit severity per day")
_r("weed_r", 0.060, "S", "Weed growth rate")
_r("damage_floor", 0.45, "L", "Pest and disease cannot remove more than 55 percent of the canopy (FAW loss reports up to 47 percent)")
_r("leaf_regrow", 0.012, "S", "Leaf recovery rate in vegetative phase")
_r("drought_leaf", 0.06, "S", "Leaf senescence under drought stress")
_r("ks_dead", 0.15, "S", "Ks below this for 14 days kills the plot")

# ---------------------------------------------------------------- water
_r("pan_cap_m3", 220.0, "S", "Lined farm pan")
_r("pan_start_m3", (60.0, 140.0), "S", "Initial water range")
_r("catch_m2", 360.0, "S", "Effective catchment (roof, pan surface, runoff area)")
_r("pan_area_m2", 120.0, "S", "Open water area")
_r("pan_evap_k", 1.1, "S", "Pan coefficient on ET0")
_r("irr_mm", (10.0, 25.0, 40.0), "E", "Irrigation depths for the three actions")
_r("irr_eff", 0.85, "S", "Application efficiency")
_r("water_shadow_kes_m3", 20.0, "S", "Cost of refilling the pan")

# ---------------------------------------------------------------- inputs and prices (KES)
_r("seed_kg", 10.0, "S", "Seed per acre")
_r("seed_price", 300.0, "C", "Certified seed KES/kg before subsidy (2026 press)")
_r("voucher_price_bag", 2000.0, "C", "Subsidised 50 kg fertilizer bag from Sept 2026 (government)")
_r("dap_market_bag", 6700.0, "C", "Open-market DAP, single county spot check, Sept 2026")
_r("can_market_bag", 4950.0, "C", "Open-market CAN, same source")
_r("vouchers_start", (1, 2, 3), "S", "Subsidised bags available at start")
_r("dap_n", 9.0, "L", "kg N in 50 kg DAP (18 percent)")
_r("dap_p", 7.0, "L", "kg plant-available P from 50 kg DAP")
_r("can_n_per_kg", 0.26, "L", "CAN is 26 percent N")
_r("manure_n", 2.5, "S", "Plant-available N from one manure application in season")
_r("manure_p", 0.8, "S", "Plant-available P from one manure application")
_r("manure_som", 0.004, "S", "SOM gain per manure application")
_r("manure_apply_kg", 500.0, "S", "Manure per application")
_r("manure_start_kg", (1000.0, 3000.0), "S", "Initial manure stock")
_r("manure_weekly_kg", 70.0, "S", "Weekly manure accumulation from the cow shed")
_r("insecticide_cost", 1200.0, "S", "Per spray")
_r("fungicide_cost", 1800.0, "S", "Per spray")
_r("pp_bag_cost", 120.0, "S", "Polypropylene bag, per 90 kg")
_r("herm_bag_cost", 260.0, "S", "Hermetic bag, per 90 kg")
_r("transport_bag", 100.0, "S", "Transport to market per 90 kg bag")
_r("wage_h", 75.0, "S", "Hired labour KES/hour")
_r("family_opp_h", 25.0, "S", "Opportunity cost of family labour KES/hour")
_r("labour_cap_h", 56.0, "S", "Family labour hours per week at full availability")
_r("cash_start", (15000.0, 25000.0), "S", "Opening cash")
_r("cash_floor", -20000.0, "S", "Bankruptcy line")
_r("maize_base_bag", 4200.0, "C", "Nakuru price range KES 3,700 to 4,600 per 90 kg bag in 2026")
_r("price_seasonal_amp", 0.14, "S", "Seasonal amplitude: low at harvest, high in lean season")
_r("price_ar", 0.85, "S", "Price shock persistence")
_r("price_sd", 0.045, "S", "Price shock sd")
_r("fert_ar", 0.9, "S", "Fertilizer price persistence")
_r("fert_sd", 0.02, "S", "Fertilizer price shock sd")
_r("liquid_disc", 0.85, "S", "Unsold stock is valued at this share of market value")

# ---------------------------------------------------------------- harvest, storage
_r("reserve_kg", 490.0, "C", "5 people at 98 kg per capita per year")
_r("bag_kg", 90.0, "L", "Standard Kenyan maize bag")
_r("dm_to_std", 1 / 0.865, "L", "Dry matter to grain at 13.5 percent moisture")
_r("harvest_share", 0.50, "S", "Share of the crop harvested per action")
_r("safe_moist", 13.5, "L", "Recommended storage moisture")
_r("moist_nodes_gdd", np.array([1500.0, 2000.0, 2300.0]), "S", "Field grain moisture nodes")
_r("moist_nodes_val", np.array([42.0, 32.0, 19.0]), "S", "Field grain moisture percent")
_r("field_loss_base", 0.008, "S", "Weekly loss of mature standing crop (lodging, birds)")
_r("field_loss_rain", 0.004, "S", "Extra loss per rain day on mature standing crop")
_r("dry_gain_dry", 6.0, "S", "Moisture points removed by sun-drying in a dry week")
_r("dry_gain_wet", 1.5, "S", "Moisture points removed in a wet week")
_r("insect_rate", (0.010, 0.004, 0.0003), "S", "Weekly insect damage by storage: none, polypropylene, hermetic")
_r("insect_age", 0.0004, "S", "Insect damage growth with weeks stored")
_r("mould_rate", (0.010, 0.006, 0.0010), "S", "Mould index per moisture point above safe, per week")
_r("none_loss", 0.02, "S", "Weekly loss of grain left unstored (rodents, damp)")

# ---------------------------------------------------------------- labour
_r("labour_ar", 0.7, "S", "Labour availability persistence")
_r("labour_shock_p", 0.08, "S", "Weekly chance of a labour shock (illness, funeral)")

# ---------------------------------------------------------------- reward
_r("r0", 88000.0, "C", "Reference gross revenue, 22 bags x KES 4,000")
_r("som_value_kes", 150000.0, "E", "Value of one unit of soil organic matter index; set so mining cannot pay")
_r("invalid_penalty", 0.002, "E", "Penalty for an invalid action")
_r("shortfall_mult", 1.3, "S", "Purchase premium on food reserve shortfall")

# ---------------------------------------------------------------- observation noise
_r("noise_moist", 0.10, "S", "Moisture estimate sd")
_r("noise_pest", 0.15, "S", "Pest, disease, weed estimate sd")
_r("noise_scout_factor", 0.3, "S", "Noise multiplier in a scouting week")
_r("noise_crop", 0.06, "S", "Canopy and health estimate sd")
_r("noise_grain_moist", 1.5, "S", "Grain moisture estimate sd (points)")
_r("noise_spoil", 0.08, "S", "Spoilage sign estimate sd")

P = SimpleNamespace(**_vals)

# Optional constants derived for convenience
HORIZON = int(P.horizon_weeks)
N_DAYS = HORIZON * 7 + 7
