"""Discrete action space: definitions, preconditions, costs and immediate effects.

Every action maps to a real smallholder management decision. Preconditions are
checked before execution; an invalid action is executed as a no-op, logged and
penalised (see rewards.py). The mask is exposed for diagnostics only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .params import P
from .state import FarmState

# action ids
WAIT, SOW, IRR_L, IRR_M, IRR_H, DAP, CAN_25, CAN_50, MANURE, SCOUT, INSECT, FUNG, WEED, MULCH, HARVEST, DRY, \
    STORE_PP, STORE_HERM, SELL_HALF, SELL_ALL = range(20)
N_ACTIONS = 20

ACTION_NAMES = [
    "wait", "sow", "irrigate_10mm", "irrigate_25mm", "irrigate_40mm", "basal_dap_50kg", "topdress_can_25kg",
    "topdress_can_50kg", "apply_manure", "scout_field", "spray_insecticide", "spray_fungicide", "hand_weed",
    "mulch", "harvest", "sun_dry", "store_pp_bags", "store_hermetic_bags", "sell_half_surplus", "sell_all_surplus",
]

ACTION_DESCRIPTIONS = [
    "No intervention this week",
    "Sow certified hybrid maize at rain onset",
    "Irrigate 10 mm from the farm pan",
    "Irrigate 25 mm from the farm pan",
    "Irrigate 40 mm from the farm pan",
    "Basal DAP, 50 kg per acre, at or soon after planting",
    "Top-dress CAN, 25 kg per acre",
    "Top-dress CAN, 50 kg per acre",
    "Apply cattle manure, 500 kg",
    "Walk the rows and inspect whorls, leaves and soil",
    "Spray insecticide against fall armyworm",
    "Spray fungicide against leaf blight",
    "Hand weeding with a hoe",
    "Spread crop residue or grass mulch",
    "Harvest a labour-limited share of the standing crop",
    "Sun-dry harvested grain on a drying sheet",
    "Store the grain in polypropylene bags",
    "Store the grain in hermetic bags",
    "Sell half of the stock above the household food reserve",
    "Sell all of the stock above the household food reserve",
]

LABOUR_H = {SOW: 24, IRR_L: 6, IRR_M: 12, IRR_H: 18, DAP: 6, CAN_25: 4, CAN_50: 6, MANURE: 14, SCOUT: 4,
            INSECT: 6, FUNG: 6, WEED: 40, MULCH: 20, HARVEST: 32, DRY: 10, STORE_PP: 8, STORE_HERM: 8,
            SELL_HALF: 8, SELL_ALL: 10}


def quality_factor(s: FarmState) -> float:
    """Price multiplier for grain quality: insect damage, mould (aflatoxin risk) and moisture."""
    q_ins = 1.0 - s.stock_insect
    q_mould = 1.0 - 0.7 * min(1.0, max(0.0, s.stock_mould - 0.1) / 0.5)
    q_moist = 1.0 - 0.04 * max(0.0, s.stock_moist - P.safe_moist)
    return max(0.05, q_ins * q_mould * q_moist)


def fert_cost(s: FarmState, bags: float, market_per_bag: float) -> float:
    """Voucher bags at the subsidised price first, remainder at the open-market price."""
    if s.vouchers + 1e-9 >= bags:
        return bags * P.voucher_price_bag
    return bags * market_per_bag


def _irrig_m3(mm: float) -> float:
    return mm * P.acre_m2 / 1000.0


def validate(s: FarmState, a: int) -> tuple[bool, str]:
    """Precondition check. Returns (valid, reason)."""
    if a == WAIT:
        return True, ""
    sown = s.sown
    if a == SOW:
        if sown:
            return False, "already sown"
        if s.week > P.sow_deadline_week:
            return False, "sowing window closed"
        if s.cash < P.seed_kg * P.seed_price:
            return False, "cannot afford seed"
        return True, ""
    if a in (IRR_L, IRR_M, IRR_H):
        mm = P.irr_mm[a - IRR_L]
        if not sown:
            return False, "not sown"
        if s.water_m3 < _irrig_m3(mm):
            return False, "not enough water"
        return True, ""
    if a == DAP:
        if not sown or s.gdd > P.gdd_silk:
            return False, "not in fertilising window"
        if s.cash < fert_cost(s, 1.0, s.dap_kg * 50):
            return False, "cannot afford"
        return True, ""
    if a in (CAN_25, CAN_50):
        if not sown or s.gdd > P.gdd_silk + 200:
            return False, "not in fertilising window"
        bags = 0.5 if a == CAN_25 else 1.0
        if s.cash < fert_cost(s, bags, s.can_kg * 50):
            return False, "cannot afford"
        return True, ""
    if a == MANURE:
        if s.manure_kg < P.manure_apply_kg:
            return False, "no manure in stock"
        return True, ""
    if a == SCOUT:
        return (True, "") if sown else (False, "not sown")
    if a == INSECT:
        if not sown or s.gdd > 1700:
            return False, "no crop to protect"
        if s.cash < P.insecticide_cost:
            return False, "cannot afford"
        return True, ""
    if a == FUNG:
        if not sown or s.gdd > 1800:
            return False, "no crop to protect"
        if s.cash < P.fungicide_cost:
            return False, "cannot afford"
        return True, ""
    if a == WEED:
        if not sown and s.week > P.sow_deadline_week:
            return False, "nothing to weed"
        return True, ""
    if a == MULCH:
        return (True, "") if float(s.mulch.mean()) < 0.8 else (False, "already mulched")
    if a == HARVEST:
        if not sown or s.gdd < P.gdd_harvest_ok:
            return False, "crop not ready"
        if s.harvested_frac > 0.999:
            return False, "fully harvested"
        return True, ""
    if a == DRY:
        if s.stock_kg <= 1 or s.stock_moist <= P.safe_moist + 0.05:
            return False, "nothing to dry"
        return True, ""
    if a in (STORE_PP, STORE_HERM):
        want = 1 if a == STORE_PP else 2
        if not s.sown or s.store_type >= want:
            return False, "not sown or already stored this way"
        bag = P.pp_bag_cost if want == 1 else P.herm_bag_cost
        if s.cash < s.stock_kg / P.bag_kg * bag:
            return False, "cannot afford bags"
        return True, ""
    if a in (SELL_HALF, SELL_ALL):
        if s.stock_kg - P.reserve_kg < P.bag_kg / 2:
            return False, "no surplus above food reserve"
        return True, ""
    return False, "unknown action"


def valid_mask(s: FarmState) -> np.ndarray:
    return np.array([validate(s, a)[0] for a in range(N_ACTIONS)], dtype=bool)


@dataclass
class ActionOutcome:
    action: int
    executed: int
    valid: bool
    reason: str
    cash_cost: float = 0.0
    labour_h: float = 0.0
    hired_cost: float = 0.0
    water_m3: float = 0.0
    sold_kg: float = 0.0
    revenue: float = 0.0
    harvested_kg: float = 0.0
    detail: str = ""


def apply_action(s: FarmState, a: int) -> ActionOutcome:
    """Validate and apply the immediate effect of an action on the state (mutates s)."""
    valid, reason = validate(s, a)
    out = ActionOutcome(a, a if valid else WAIT, valid, reason)
    if not valid:
        s.invalid_actions += 1
        return out
    a = int(a)
    hours = float(LABOUR_H.get(a, 0))
    cash = 0.0

    if a == SOW:
        cash = P.seed_kg * P.seed_price
        s.sown = True
        s.sow_week = s.week
        out.detail = f"sown week {s.week}"
    elif a in (IRR_L, IRR_M, IRR_H):
        mm = P.irr_mm[a - IRR_L]
        m3 = _irrig_m3(mm)
        s.water_m3 -= m3
        s.water_used_m3 += m3
        s.dr = np.maximum(0.0, s.dr - mm * P.irr_eff)
        out.water_m3 = m3
        out.detail = f"{mm:.0f} mm, {m3:.0f} m3"
    elif a == DAP:
        cash = fert_cost(s, 1.0, s.dap_kg * 50)
        if s.vouchers >= 1.0 - 1e-9:
            s.vouchers -= 1.0
        s.n_min = s.n_min + P.dap_n
        s.p_av = s.p_av + P.dap_p
        s.n_applied += P.dap_n
        s.p_applied += P.dap_p
        out.detail = "DAP 50 kg"
    elif a in (CAN_25, CAN_50):
        bags = 0.5 if a == CAN_25 else 1.0
        kg = 50.0 * bags
        cash = fert_cost(s, bags, s.can_kg * 50)
        if s.vouchers >= bags - 1e-9:
            s.vouchers -= bags
        n = kg * P.can_n_per_kg
        s.n_min = s.n_min + n
        s.n_applied += n
        out.detail = f"CAN {kg:.0f} kg"
    elif a == MANURE:
        s.manure_kg -= P.manure_apply_kg
        s.n_min = s.n_min + P.manure_n
        s.p_av = s.p_av + P.manure_p
        s.som = np.minimum(1.0, s.som + P.manure_som)
        s.manure_applied += 1
        out.detail = "manure 500 kg"
    elif a == SCOUT:
        s.scouted = True
        s.weeks_since_scout = -1
    elif a == INSECT:
        cash = P.insecticide_cost
        timing = 1.0 if s.gdd < 1000 else 0.35
        resist = 0.92 ** s.insecticide_n
        kill = 0.8 * timing * resist
        s.faw = s.faw * (1.0 - kill)
        s.insecticide_n += 1
        s.faw_protect_days = 7
        s.weeks_since_insecticide = -1
        out.detail = f"kill {kill:.2f}"
    elif a == FUNG:
        cash = P.fungicide_cost
        s.dis = s.dis * 0.7
        s.fung_protect_days = 14
        s.weeks_since_fungicide = -1
    elif a == WEED:
        s.weeds = s.weeds * 0.10
    elif a == MULCH:
        s.mulch = np.minimum(1.0, s.mulch + 0.7)
    elif a == HARVEST:
        remaining = 1.0 - s.harvested_frac
        share = min(P.harvest_share, remaining)
        take = share / remaining if remaining > 1e-9 else 1.0
        grain_dm = s.grain * take                       # g/m2 per plot
        kg = float(np.sum(grain_dm * P.area_plot_m2 / 1000.0) * P.dm_to_std)
        moist = float(np.interp(s.gdd, P.moist_nodes_gdd, P.moist_nodes_val))
        s.grain = s.grain - grain_dm
        s.harvested_frac += share
        # mix into stock
        tot = s.stock_kg + kg
        if tot > 0:
            s.stock_moist = (s.stock_moist * s.stock_kg + moist * kg) / tot
        if s.store_type > 0:
            bag = P.pp_bag_cost if s.store_type == 1 else P.herm_bag_cost
            cash += kg / P.bag_kg * bag
        s.stock_kg = tot
        s.harvested_kg += kg
        out.harvested_kg = kg
        out.detail = f"{kg:.0f} kg at {moist:.1f}% moisture"
    elif a == DRY:
        wet_week = s.rain_days_last >= 3
        gain = P.dry_gain_wet if wet_week else P.dry_gain_dry
        s.stock_moist = max(P.safe_moist - 0.3, s.stock_moist - gain)
    elif a in (STORE_PP, STORE_HERM):
        want = 1 if a == STORE_PP else 2
        bag = P.pp_bag_cost if want == 1 else P.herm_bag_cost
        cash = s.stock_kg / P.bag_kg * bag   # bags for grain already in hand; later harvests pay as they arrive
        s.store_type = want
        out.detail = "polypropylene" if want == 1 else "hermetic"
    elif a in (SELL_HALF, SELL_ALL):
        surplus = s.stock_kg - P.reserve_kg
        kg = surplus * (0.5 if a == SELL_HALF else 1.0)
        q = quality_factor(s)
        bags = kg / P.bag_kg
        rev = bags * (s.price * q - P.transport_bag)
        s.stock_kg -= kg
        s.sold_kg += kg
        s.revenue += rev
        s.cash += rev
        out.sold_kg, out.revenue = kg, rev
        out.detail = f"{kg:.0f} kg at KES {s.price * q:.0f}/bag"

    # labour: family hours first, remainder hired
    avail = s.labour_h
    hired_h = max(0.0, hours - avail)
    hired_cost = hired_h * P.wage_h
    s.labour_h = max(0.0, avail - hours)
    s.cash -= cash + hired_cost
    s.input_cost += cash
    s.labour_cost += hired_cost
    out.cash_cost, out.labour_h, out.hired_cost = cash, hours, hired_cost
    return out
