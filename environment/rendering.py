"""State serialisation for the frontend and a text renderer.

serialize_state() is the single documented bridge from the Gymnasium
environment to the Three.js view. It reads the true FarmState; it never
computes agronomy and never changes the environment.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from . import dynamics
from .actions import ACTION_NAMES, quality_factor
from .params import ENV_VERSION, HORIZON, P

SCHEMA_VERSION = "1.0"
STAGE_NAMES = ["unsown", "vegetative", "flowering", "grain_fill", "dry_down", "harvested"]
_YEAR0 = date(2027, 3, 1)


def week_date(week: int) -> date:
    return _YEAR0 + timedelta(days=7 * week)


def serialize_state(env) -> dict:
    s, rnd = env.state, env.rnd
    w = max(0, s.week - 1)
    d0 = 7 * w
    wx = rnd.weather
    theta = (1.0 - s.dr / P.taw)
    out = env.last_outcome
    dt = week_date(s.week)
    plots = []
    for i in range(int(P.plots)):
        plots.append({
            "id": i,
            "soil_moisture": float(theta[i]),
            "mineral_n": float(s.n_min[i]),
            "available_p": float(s.p_av[i]),
            "organic_matter": float(s.som[i]),
            "mulch": float(s.mulch[i]),
            "weeds": float(s.weeds[i]),
            "faw_pressure": float(s.faw[i]),
            "disease_severity": float(s.dis[i]),
            "canopy_retained": float(s.leaf[i]),
            "lai": float(dynamics.lai_potential(s.gdd) * s.leaf[i]) if s.sown else 0.0,
            "biomass_gm2": float(s.veg[i] + s.grain[i]),
            "grain_gm2": float(s.grain[i]),
            "stress": float(s.stress[i]),
            "alive": bool(s.alive[i]),
            "water_capacity_mm": float(P.taw[i]),
            "slope_class": i,
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "env_version": ENV_VERSION,
        "time": {"week": s.week, "horizon": HORIZON, "date": dt.isoformat(), "label": dt.strftime("%d %b"),
                 "doy": int((P.start_doy + 7 * s.week) % 365)},
        "scenario": {"name": rnd.scenario.name, "rain_mult": rnd.scenario.rain_mult,
                     "temp_shift": rnd.scenario.temp_shift, "onset_delay_weeks": rnd.scenario.onset_delay_weeks,
                     "forecast_skill": rnd.scenario.forecast_skill},
        "weather": {
            "rain_week_mm": float(s.rain_last), "rain_days": int(s.rain_days_last), "tmax_week": float(s.tmax_last),
            "daily_rain_mm": [float(x) for x in wx.rain[d0:d0 + 7]],
            "daily_tmax": [float(x) for x in wx.tmax[d0:d0 + 7]],
            "daily_tmin": [float(x) for x in wx.tmin[d0:d0 + 7]],
            "forecast_next_week_mm": float(dynamics.forecast_next_week(s, rnd)) if s.week < HORIZON else 0.0,
        },
        "crop": {"sown": bool(s.sown), "sow_week": int(s.sow_week), "gdd": float(s.gdd),
                 "stage": STAGE_NAMES[dynamics.stage_of(s)], "stage_index": int(dynamics.stage_of(s)),
                 "harvested_fraction": float(s.harvested_frac),
                 "standing_grain_kg": dynamics.standing_grain_kg(s)},
        "plots": plots,
        "resources": {"cash": float(s.cash), "water_m3": float(s.water_m3), "water_capacity_m3": P.pan_cap_m3,
                      "vouchers": float(s.vouchers), "manure_kg": float(s.manure_kg),
                      "labour_hours": float(s.labour_h), "labour_capacity": P.labour_cap_h},
        "storage": {"stock_kg": float(s.stock_kg), "bags": float(s.stock_kg / P.bag_kg),
                    "moisture": float(s.stock_moist), "insect_damage": float(s.stock_insect),
                    "mould_index": float(s.stock_mould), "store_type": int(s.store_type),
                    "store_type_name": ["none", "polypropylene", "hermetic"][int(s.store_type)],
                    "weeks_stored": int(s.store_weeks), "quality": float(quality_factor(s)),
                    "reserve_kg": P.reserve_kg},
        "market": {"maize_bag_kes": float(s.price), "dap_bag_kes": float(s.dap_kg * 50),
                   "can_bag_kes": float(s.can_kg * 50)},
        "protection": {"faw_protect_days": int(s.faw_protect_days), "fung_protect_days": int(s.fung_protect_days)},
        "action": None if out is None else {
            "requested": ACTION_NAMES[out.action], "executed": ACTION_NAMES[out.executed], "valid": bool(out.valid),
            "reason": out.reason, "detail": out.detail, "cash_cost": float(out.cash_cost),
            "water_m3": float(out.water_m3), "labour_hours": float(out.labour_h),
            "sold_kg": float(out.sold_kg), "revenue": float(out.revenue), "harvested_kg": float(out.harvested_kg),
        },
        "reward": {"last_components": {k: float(v) for k, v in env.last_components.items()},
                   "episode_return": float(env.episode_return)},
        "metrics": env.episode_summary(),
    }


def render_ansi(env) -> str:
    s = env.state
    d = serialize_state(env)
    return (
        f"Wk {s.week:2d} {d['time']['label']}  stage={d['crop']['stage']:<10s} "
        f"rain={s.rain_last:5.1f}mm  cash=KES {s.cash:8.0f}  water={s.water_m3:5.0f}m3  "
        f"stock={s.stock_kg:6.0f}kg  price={s.price:5.0f}  return={env.episode_return:6.3f}"
    )
