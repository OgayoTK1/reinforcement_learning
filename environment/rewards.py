"""Reward function.

r_t = dW/R0 + w_s * dSOM/R0 - water_shadow/R0 - family_labour_opportunity/R0
      - invalid_penalty                                   (every step)
r_T additionally: - food_reserve_shortfall_cost/R0        (at termination)

W is farm wealth: cash plus unsold grain valued at a liquidation share of the
quality-adjusted market price. Everything is in KES divided by the reference
gross revenue R0, so the reward is dimensionless and comparable across seeds.
There is no shaping on biomass, greenness or growth stage.
"""
from __future__ import annotations

from .actions import ActionOutcome, quality_factor
from .params import P
from .state import FarmState


def wealth(s: FarmState) -> float:
    stock_value = s.stock_kg / P.bag_kg * s.price * quality_factor(s) * P.liquid_disc
    return s.cash + stock_value


def soil_index(s: FarmState) -> float:
    return float(s.som.mean())


def shortfall_cost(s: FarmState) -> float:
    short = max(0.0, P.reserve_kg - s.stock_kg)
    return short / P.bag_kg * s.price * P.shortfall_mult


def step_reward(prev_wealth: float, prev_soil: float, s: FarmState, out: ActionOutcome, terminal: bool) -> tuple[float, dict]:
    d_wealth = (wealth(s) - prev_wealth) / P.r0
    soil = P.som_value_kes * (soil_index(s) - prev_soil) / P.r0
    water = -P.water_shadow_kes_m3 * out.water_m3 / P.r0
    hired_h = out.hired_cost / P.wage_h if P.wage_h else 0.0
    family_h = max(0.0, out.labour_h - hired_h)
    labour = -P.family_opp_h * family_h / P.r0
    invalid = 0.0 if out.valid else -P.invalid_penalty
    short = -shortfall_cost(s) / P.r0 if terminal else 0.0
    total = d_wealth + soil + water + labour + invalid + short
    return total, {
        "wealth": d_wealth, "soil": soil, "water": water, "labour": labour,
        "invalid": invalid, "shortfall": short,
    }
