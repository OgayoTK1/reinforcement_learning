"""Farm state and the per-episode random draws.

FarmState is the complete, hidden simulation state. The agent never sees it
directly; the observation builder derives noisy estimates from it. The same
dictionary form is what the backend serialises for the 3D view.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from .params import HORIZON, N_DAYS, P
from .weather import Scenario, WeatherYear, generate_weather, sample_training_scenario

NP = int(P.plots)


@dataclass
class EpisodeRandomness:
    """Everything stochastic in an episode, drawn at reset and independent of actions."""

    seed: int
    scenario: Scenario
    weather: WeatherYear
    imm_u: np.ndarray
    imm_amt: np.ndarray
    labour_u: np.ndarray
    shock_u: np.ndarray
    price_eps: np.ndarray
    fert_eps: np.ndarray
    obs_noise: np.ndarray
    init: dict = field(default_factory=dict)


def draw_episode(seed: int, scenario: Scenario | None = None) -> EpisodeRandomness:
    rng = np.random.default_rng(seed)
    if scenario is None:
        scenario = sample_training_scenario(rng)
    else:
        sample_training_scenario(rng)  # keep the stream aligned whether or not the scenario is fixed
    weather = generate_weather(rng, scenario, N_DAYS)
    T = HORIZON + 1
    imm_u = rng.random(T)
    imm_amt = rng.uniform(P.faw_imm_amt[0], P.faw_imm_amt[1], (T, NP)) * rng.lognormal(0.0, 0.3, (T, NP))
    labour_u = rng.uniform(0.6, 1.0, T)
    shock_u = rng.random(T)
    price_eps = rng.normal(0.0, 1.0, T)
    fert_eps = rng.normal(0.0, 1.0, T)
    obs_noise = rng.normal(0.0, 1.0, (T, 24))
    init = {
        "dr_frac": rng.uniform(0.25, 0.6),
        "n0": rng.uniform(5.0, 11.0),
        "p0": rng.uniform(3.0, 6.0),
        "som0": rng.uniform(0.55, 0.68),
        "vouchers": float(rng.choice(P.vouchers_start)),
        "cash": rng.uniform(*P.cash_start),
        "water": rng.uniform(*P.pan_start_m3),
        "manure": rng.uniform(*P.manure_start_kg),
        "price_x": rng.normal(0.0, 0.04),
    }
    return EpisodeRandomness(seed, scenario, weather, imm_u, imm_amt, labour_u, shock_u,
                             price_eps, fert_eps, obs_noise, init)


@dataclass
class FarmState:
    week: int = 0
    day: int = 0
    # per-plot arrays
    dr: np.ndarray = None          # root-zone depletion, mm below field capacity
    n_min: np.ndarray = None       # mineral N, kg/acre
    p_av: np.ndarray = None        # available P, kg/acre
    som: np.ndarray = None         # soil organic matter index 0..1
    mulch: np.ndarray = None       # mulch cover 0..1
    weeds: np.ndarray = None       # relative weed biomass 0..1
    faw: np.ndarray = None         # fall armyworm pressure 0..1
    dis: np.ndarray = None         # leaf blight severity 0..1
    leaf: np.ndarray = None        # retained canopy fraction 0..1
    veg: np.ndarray = None         # vegetative biomass g/m2
    grain: np.ndarray = None       # standing grain dry matter g/m2
    kernel_num: np.ndarray = None
    kernel_den: np.ndarray = None
    stress: np.ndarray = None      # smoothed combined stress factor
    drought_days: np.ndarray = None
    alive: np.ndarray = None
    # crop clock
    gdd: float = 0.0
    sown: bool = False
    sow_week: int = -1
    # farm resources
    cash: float = 0.0
    water_m3: float = 0.0
    vouchers: float = 0.0
    manure_kg: float = 0.0
    labour_f: float = 1.0
    labour_h: float = 0.0
    # harvest and storage
    harvested_frac: float = 0.0
    stock_kg: float = 0.0
    stock_moist: float = 20.0
    stock_insect: float = 0.0
    stock_mould: float = 0.0
    store_type: int = 0            # 0 none, 1 polypropylene, 2 hermetic
    store_weeks: int = 0
    # market
    price: float = 0.0             # KES per 90 kg bag
    price_x: float = 0.0
    dap_kg: float = 0.0            # open-market DAP price per kg
    can_kg: float = 0.0
    fert_x: float = 0.0
    price_hist: list = field(default_factory=list)
    # protection timers and bookkeeping
    faw_protect_days: int = 0
    fung_protect_days: int = 0
    insecticide_n: int = 0
    wet_ewma: float = 0.2
    last_action: int = 0
    scouted: bool = False
    weeks_since_scout: int = 99
    weeks_since_insecticide: int = 99
    weeks_since_fungicide: int = 99
    n_applied: float = 0.0
    p_applied: float = 0.0
    manure_applied: int = 0
    # weather memory visible to the farmer
    rain_hist: list = field(default_factory=list)
    rain_last: float = 0.0
    tmax_last: float = 22.0
    rain_days_last: int = 0
    # cumulative metrics
    water_used_m3: float = 0.0
    season_rain_mm: float = 0.0
    harvested_kg: float = 0.0
    sold_kg: float = 0.0
    revenue: float = 0.0
    input_cost: float = 0.0
    labour_cost: float = 0.0
    ph_loss_kg: float = 0.0
    invalid_actions: int = 0
    som_start: float = 0.0
    terminated_reason: str = ""

    def copy(self) -> "FarmState":
        return copy.deepcopy(self)

    def to_dict(self) -> dict:
        """JSON-friendly snapshot of the full simulation state."""
        out = {}
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                out[k] = v.astype(float).tolist() if v.dtype != bool else v.tolist()
            elif isinstance(v, (np.floating, np.integer)):
                out[k] = float(v)
            elif isinstance(v, np.bool_):
                out[k] = bool(v)
            else:
                out[k] = v
        return out


def initial_state(rnd: EpisodeRandomness) -> FarmState:
    i = rnd.init
    plot_n = np.array([1.2, 1.0, 0.9, 0.8])
    plot_som = np.array([0.05, 0.0, -0.05, -0.1])
    s = FarmState()
    s.dr = P.taw * i["dr_frac"]
    s.n_min = i["n0"] * plot_n
    s.p_av = np.full(NP, i["p0"])
    s.som = np.clip(i["som0"] + plot_som, 0.2, 0.9)
    s.mulch = np.zeros(NP)
    s.weeds = np.full(NP, 0.15)
    s.faw = np.zeros(NP)
    s.dis = np.full(NP, 0.005)
    s.leaf = np.ones(NP)
    s.veg = np.zeros(NP)
    s.grain = np.zeros(NP)
    s.kernel_num = np.zeros(NP)
    s.kernel_den = np.zeros(NP)
    s.stress = np.ones(NP)
    s.drought_days = np.zeros(NP)
    s.alive = np.ones(NP, dtype=bool)
    s.cash = float(i["cash"])
    s.water_m3 = float(i["water"])
    s.vouchers = float(i["vouchers"])
    s.manure_kg = float(i["manure"])
    s.labour_f = 1.0
    s.labour_h = P.labour_cap_h
    s.price_x = float(i["price_x"])
    s.price = base_price(0, s.price_x)
    s.dap_kg = P.dap_market_bag / 50.0
    s.can_kg = P.can_market_bag / 50.0
    s.price_hist = [s.price] * 5
    s.som_start = float(s.som.mean())
    return s


def seasonal_price_factor(week: int) -> float:
    return 1.0 + P.price_seasonal_amp * float(np.cos(2 * np.pi * (week - 16) / 52.0))


def base_price(week: int, x: float) -> float:
    return float(P.maize_base_bag * seasonal_price_factor(week) * np.exp(x))
