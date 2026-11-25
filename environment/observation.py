"""Observation specification and construction.

The observation is a partial, noisy view of the hidden FarmState. One spec
list drives the vector builder, the README table and the report table, so the
documentation cannot drift from the code. Every variable enters the networks
the same way: as one element of a float32 vector fed to the first linear layer.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from .actions import ACTION_NAMES, N_ACTIONS
from .dynamics import forecast_next_week, stage_of
from .params import ENV_VERSION, HORIZON, P
from .state import EpisodeRandomness, FarmState

STAGES = ["unsown", "vegetative", "flowering", "grain_fill", "dry_down", "harvested"]
STORES = ["none", "polypropylene", "hermetic"]


@dataclass(frozen=True)
class ObsVar:
    name: str
    meaning: str
    dtype: str
    rng: str
    source: str
    update: str
    norm: str
    kind: str  # "observable" or "estimated"


def _v(name, meaning, rng, source, norm, kind="observable", dtype="float32", update="weekly"):
    return ObsVar(name, meaning, dtype, rng, source, update, norm, kind)


OBS_VARS: list[ObsVar] = [
    _v("week_frac", "Week of the marketing year", "[0, 1]", "Farm calendar", "week / 52"),
    _v("sown", "Maize has been sown", "{0, 1}", "Farmer's own record", "none", dtype="binary"),
    _v("weeks_since_sowing", "Crop age", "[0, 1.2]", "Farmer's own record", "weeks / 40, clipped"),
]
_v_stage = [_v(f"stage_{n}", f"Crop stage is {n.replace('_', ' ')}", "{0, 1}", "Visual inspection of the field",
               "one-hot", dtype="binary") for n in STAGES]
OBS_VARS += _v_stage
OBS_VARS += [
    _v("canopy_greenness", "Retained green canopy, mean of plots", "[0, 1.2]", "Visual scoring of leaves",
       "fraction + N(0, 0.06)", "estimated"),
    _v("crop_health", "Overall plant health (leaf and stress)", "[0, 1.2]", "Visual scoring", "fraction + N(0, 0.06)",
       "estimated"),
    _v("soil_moisture_mean", "Relative available soil water, mean of plots", "[-0.2, 1.3]",
       "Finger test or low-cost soil moisture probe", "fraction + N(0, 0.10), sd x0.3 if scouted", "estimated"),
    _v("soil_moisture_min", "Relative available soil water, driest plot", "[-0.2, 1.3]",
       "Finger test or low-cost soil moisture probe", "fraction + N(0, 0.10), sd x0.3 if scouted", "estimated"),
    _v("rain_last_week", "Rain gauge total, last week", "[0, 1.5]", "Farm rain gauge", "mm / 60, clipped"),
    _v("rain_last_4_weeks", "Rain gauge total, last four weeks", "[0, 1.5]", "Farm rain gauge", "mm / 200, clipped"),
    _v("tmax_last_week", "Mean daily maximum, last week", "[-1, 1.5]", "Max-min thermometer", "(C - 20) / 10"),
    _v("rain_forecast_next_week", "Forecast rainfall, next week", "[0, 1.5]", "County forecast (KMD)",
       "mm / 60, clipped; error grows as forecast skill falls", "estimated"),
    _v("n_deficiency", "Nitrogen deficiency score", "[0, 1]", "Leaf colour chart", "1 - N factor + N(0, 0.08)",
       "estimated"),
    _v("n_applied", "Cumulative fertilizer N applied", "[0, 1.5]", "Farmer's input record", "kg N / 60"),
    _v("p_applied", "Cumulative fertilizer P applied", "[0, 1.5]", "Farmer's input record", "kg P / 20"),
    _v("manure_applications", "Manure applications so far", "[0, 1.5]", "Farmer's input record", "count / 3"),
    _v("faw_estimate", "Fall armyworm pressure estimate", "[0, 1]", "Scouting (whorl damage counts)",
       "fraction + N(0, 0.15), sd x0.3 if scouted", "estimated"),
    _v("disease_estimate", "Leaf blight severity estimate", "[0, 1]", "Scouting (leaf lesion scoring)",
       "fraction + N(0, 0.15), sd x0.3 if scouted", "estimated"),
    _v("weed_estimate", "Weed cover estimate", "[0, 1]", "Scouting (weed cover scoring)",
       "fraction + N(0, 0.15), sd x0.3 if scouted", "estimated"),
    _v("weeks_since_scout", "Weeks since the last scouting walk", "[0, 1]", "Farmer's own record", "weeks / 8, clipped"),
    _v("weeks_since_insecticide", "Weeks since the last insecticide spray", "[0, 1]", "Farmer's input record",
       "weeks / 8, clipped"),
    _v("weeks_since_fungicide", "Weeks since the last fungicide spray", "[0, 1]", "Farmer's input record",
       "weeks / 8, clipped"),
    _v("water_level", "Farm pan water level", "[0, 1]", "Staff gauge in the pan", "m3 / 220"),
    _v("cash", "Cash on hand", "[-1, 2]", "Cash book / mobile money balance", "KES / 88,000, clipped"),
    _v("labour_available", "Family labour hours available this week", "[0, 1]", "Household roster", "hours / 56"),
    _v("vouchers", "Subsidised fertilizer bags left on the e-voucher", "[0, 1]", "NCPB voucher record", "bags / 3"),
    _v("fertilizer_price", "Open-market DAP price", "[0.5, 1.5]", "Agro-dealer quote", "KES per bag / 6,700"),
    _v("maize_price", "Maize price per 90 kg bag", "[0.5, 1.5]", "Market price enquiry (trader or SMS)",
       "KES / 4,200"),
    _v("price_change_4wk", "Four-week change in maize price", "[-1, 1]", "Market price enquiry",
       "log ratio / 0.3, clipped"),
    _v("harvest_readiness", "Maturity indicator (husk and kernel colour)", "[0, 1.2]", "Visual inspection of cobs",
       "GDD / 2000", "estimated"),
    _v("harvested_fraction", "Share of the crop already cut", "[0, 1]", "Farmer's own record", "fraction"),
    _v("grain_stock", "Grain in store or on the drying yard", "[0, 1.5]", "Bag count", "kg / 2500, clipped"),
    _v("grain_moisture", "Grain moisture", "[0, 2]", "Hand-held grain moisture meter",
       "(percent - 10) / 20 + N(0, 0.075)", "estimated"),
    *[_v(f"store_{n}", f"Grain kept in {n} storage", "{0, 1}", "Farmer's own record", "one-hot", dtype="binary")
      for n in STORES],
    _v("weeks_in_store", "Weeks the grain has been stored", "[0, 1.5]", "Farmer's own record", "weeks / 20"),
    _v("spoilage_signs", "Visible mould and insect damage", "[0, 1]", "Visual inspection of grain",
       "mean(insect, mould) + N(0, 0.08)", "estimated"),
    _v("mulch_cover", "Mulch cover on the soil", "[0, 1]", "Visual inspection", "fraction"),
    _v("scouted_last_week", "The farmer scouted last week", "{0, 1}", "Farmer's own record", "none", dtype="binary"),
    *[_v(f"prev_action_{n}", f"Last executed action was {n}", "{0, 1}", "Farmer's own record", "one-hot",
         dtype="binary") for n in ACTION_NAMES],
]

OBS_NAMES = [v.name for v in OBS_VARS]
OBS_DIM = len(OBS_VARS)
OBS_LOW, OBS_HIGH = -2.0, 3.0


def spec_hash() -> str:
    payload = json.dumps({"v": ENV_VERSION, "obs": [(v.name, v.rng) for v in OBS_VARS], "actions": ACTION_NAMES,
                          "T": HORIZON}, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def obs_table_markdown() -> str:
    rows = ["| Observation | Meaning | Source | Encoding / type | Range | Update | Normalisation | Observed or estimated |",
            "|---|---|---|---|---|---|---|---|"]
    for v in OBS_VARS:
        rows.append(f"| {v.name} | {v.meaning} | {v.source} | {v.dtype} | {v.rng} | {v.update} | {v.norm} | {v.kind} |")
    return "\n".join(rows)


def build_observation(s: FarmState, rnd: EpisodeRandomness) -> np.ndarray:
    n = rnd.obs_noise[min(s.week, rnd.obs_noise.shape[0] - 1)]
    sf = P.noise_scout_factor if s.scouted else 1.0
    theta = 1.0 - s.dr / P.taw
    alive = s.alive.astype(float)
    sown = 1.0 if s.sown else 0.0
    stage = stage_of(s)

    f_n = np.clip(s.n_min / P.n_crit, P.stress_floor, 1.0)
    greenness = float(np.mean(s.leaf * alive)) * sown
    health = float(np.mean(alive * (0.5 * s.leaf + 0.5 * s.stress))) * sown
    stock = s.stock_kg > 1.0
    hist = s.price_hist
    price_change = float(np.clip(np.log(max(s.price, 1.0) / max(hist[0], 1.0)) / 0.3, -1.0, 1.0))
    fc = forecast_next_week(s, rnd) if s.week < HORIZON else 0.0
    spoil = 0.5 * (s.stock_insect + s.stock_mould) if stock else 0.0

    vals: dict[str, float] = {
        "week_frac": s.week / HORIZON,
        "sown": sown,
        "weeks_since_sowing": min(1.2, (s.week - s.sow_week) / 40.0) if s.sown else 0.0,
        "canopy_greenness": float(np.clip(greenness + sown * n[1] * P.noise_crop, 0, 1.2)),
        "crop_health": float(np.clip(health + sown * n[2] * P.noise_crop, 0, 1.2)),
        "soil_moisture_mean": float(np.clip(theta.mean() + n[3] * P.noise_moist * sf, -0.2, 1.3)),
        "soil_moisture_min": float(np.clip(theta.min() + n[4] * P.noise_moist * sf, -0.2, 1.3)),
        "rain_last_week": float(min(1.5, s.rain_last / 60.0)),
        "rain_last_4_weeks": float(min(1.5, sum(s.rain_hist) / 200.0)),
        "tmax_last_week": (s.tmax_last - 20.0) / 10.0,
        "rain_forecast_next_week": float(min(1.5, fc / 60.0)),
        "n_deficiency": float(np.clip(1.0 - f_n.mean() + sown * n[5] * 0.08, 0, 1)),
        "n_applied": min(1.5, s.n_applied / 60.0),
        "p_applied": min(1.5, s.p_applied / 20.0),
        "manure_applications": min(1.5, s.manure_applied / 3.0),
        "faw_estimate": float(np.clip(s.faw.mean() + sown * n[6] * P.noise_pest * sf, 0, 1)),
        "disease_estimate": float(np.clip(s.dis.mean() + sown * n[7] * P.noise_pest * sf, 0, 1)),
        "weed_estimate": float(np.clip(s.weeds.mean() + n[8] * P.noise_pest * sf, 0, 1)),
        "weeks_since_scout": min(8, max(0, s.weeks_since_scout)) / 8.0,
        "weeks_since_insecticide": min(8, max(0, s.weeks_since_insecticide)) / 8.0,
        "weeks_since_fungicide": min(8, max(0, s.weeks_since_fungicide)) / 8.0,
        "water_level": s.water_m3 / P.pan_cap_m3,
        "cash": float(np.clip(s.cash / P.r0, -1.0, 2.0)),
        "labour_available": s.labour_h / P.labour_cap_h,
        "vouchers": min(1.0, s.vouchers / 3.0),
        "fertilizer_price": float(np.clip(s.dap_kg * 50.0 / P.dap_market_bag, 0.5, 1.5)),
        "maize_price": float(np.clip(s.price / P.maize_base_bag, 0.5, 1.5)),
        "price_change_4wk": price_change,
        "harvest_readiness": float(min(1.2, s.gdd / P.gdd_mat)) * sown,
        "harvested_fraction": min(1.0, s.harvested_frac),
        "grain_stock": float(min(1.5, s.stock_kg / 2500.0)),
        "grain_moisture": float(np.clip((s.stock_moist + n[9] * P.noise_grain_moist - 10.0) / 20.0, 0, 2)) if stock else 0.0,
        "weeks_in_store": min(1.5, s.store_weeks / 20.0),
        "spoilage_signs": float(np.clip(spoil + n[10] * P.noise_spoil, 0, 1)) if stock else 0.0,
        "mulch_cover": float(s.mulch.mean()),
        "scouted_last_week": 1.0 if s.scouted else 0.0,
    }
    for i, nm in enumerate(STAGES):
        vals[f"stage_{nm}"] = 1.0 if i == stage else 0.0
    for i, nm in enumerate(STORES):
        vals[f"store_{nm}"] = 1.0 if (i == s.store_type) else 0.0
    for i, nm in enumerate(ACTION_NAMES):
        vals[f"prev_action_{nm}"] = 1.0 if i == s.last_action else 0.0
    return np.array([vals[k] for k in OBS_NAMES], dtype=np.float32)
