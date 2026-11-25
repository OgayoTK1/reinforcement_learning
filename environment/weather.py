"""Stochastic weather generation for the Njoro reference farm.

All randomness for an episode is drawn up front from one seeded generator, so
the weather never depends on the agent's actions (common random numbers
across policies) and reset(seed=...) is fully deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .params import N_DAYS, P

MONTH_START = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
MONTH_DAYS = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
_MID = MONTH_START + MONTH_DAYS / 2.0


@dataclass(frozen=True)
class Scenario:
    """Climate and forecast conditions for one episode."""

    name: str = "baseline"
    rain_mult: float = 1.0
    temp_shift: float = 0.0
    onset_delay_weeks: int = 0
    forecast_skill: float = 0.6


# Held-out out-of-distribution families (never sampled in training).
SCENARIOS: dict[str, Scenario] = {
    "baseline": Scenario("baseline"),
    "severe_drought": Scenario("severe_drought", rain_mult=0.55),
    "late_onset": Scenario("late_onset", rain_mult=0.9, onset_delay_weeks=4),
    "wet_year": Scenario("wet_year", rain_mult=1.55),
    "warm_1p5": Scenario("warm_1p5", temp_shift=1.5),
    "warm_3": Scenario("warm_3", temp_shift=3.0),
    "poor_forecast": Scenario("poor_forecast", forecast_skill=0.1),
    "drought_warm": Scenario("drought_warm", rain_mult=0.65, temp_shift=1.5),
}


def sample_training_scenario(rng: np.random.Generator) -> Scenario:
    """Training and in-distribution evaluation draw: moderate interannual spread."""
    rain = float(np.clip(np.exp(rng.normal(0.0, 0.18)), 0.75, 1.30))
    shift = float(np.clip(rng.normal(0.0, 0.5), -1.0, 1.0))
    delay = int(rng.choice([0, 1, 2], p=[0.6, 0.2, 0.2]))
    skill = float(rng.uniform(0.4, 0.8))
    return Scenario("train_draw", rain, shift, delay, skill)


@dataclass
class WeatherYear:
    rain: np.ndarray
    tmax: np.ndarray
    tmin: np.ndarray
    rs: np.ndarray
    et0: np.ndarray
    clim_rain: np.ndarray  # climatological mean daily rain, used by the forecast
    forecast_skill: float


def _month_index(doy: np.ndarray) -> np.ndarray:
    return np.searchsorted(MONTH_START, doy, side="right") - 1


def generate_weather(rng: np.random.Generator, scenario: Scenario, n_days: int = N_DAYS) -> WeatherYear:
    doy = (int(P.start_doy) + np.arange(n_days)) % 365
    month = _month_index(doy)

    anomaly = np.exp(rng.normal(0.0, P.month_anom_sd, 12))
    u = rng.random(n_days)
    depth = rng.gamma(P.wet_shape, 1.0 / P.wet_shape, n_days)
    ar_noise = rng.normal(0.0, 1.0, n_days)
    tmin_noise = rng.normal(0.0, 1.0, n_days)

    p_month = np.minimum(0.9, P.rain_month_mm / (P.wet_day_mm * MONTH_DAYS))
    sc = scenario.rain_mult ** 0.5
    delay_days = scenario.onset_delay_weeks * 7

    rain = np.zeros(n_days)
    wet = False
    rho = P.rain_persist
    for i in range(n_days):
        m = month[i]
        p = p_month[m] * sc * anomaly[m] ** 0.5
        if i < delay_days:
            p *= 0.25
        p = min(p, 0.92)
        pw = p + rho * (1.0 - p) if wet else p * (1.0 - rho)
        wet = u[i] < pw
        if wet:
            rain[i] = depth[i] * P.wet_day_mm * sc * anomaly[m] ** 0.5

    # temperature: seasonal mean + AR(1) noise, cooler max and warmer min on rain days
    x = np.empty(n_days)
    x[0] = ar_noise[0] * P.temp_noise_sd
    a = P.temp_ar
    s = np.sqrt(1 - a * a)
    for i in range(1, n_days):
        x[i] = a * x[i - 1] + s * ar_noise[i] * P.temp_noise_sd
    rainy = rain > 1.0
    base_max = np.interp(doy, _MID, P.tmax_month, period=365)
    base_min = np.interp(doy, _MID, P.tmin_month, period=365)
    tmax = base_max + scenario.temp_shift + x - P.rain_day_tmax_drop * rainy
    tmin = base_min + scenario.temp_shift + 0.4 * x + 0.8 * tmin_noise * 0.5 + 0.4 * rainy
    tmax = np.maximum(tmax, tmin + 3.0)

    tmean = 0.5 * (tmax + tmin)
    tr = tmax - tmin
    ra = P.ra_mj + P.ra_amp * np.cos(4 * np.pi * (doy - 80) / 365.0)
    rs = P.rs_k * np.sqrt(tr) * ra * np.where(rainy, P.rs_rain, 1.0)
    et0 = P.hargreaves_k * (tmean + 17.8) * np.sqrt(tr) * ra / 2.45
    clim = P.rain_month_mm[month] / MONTH_DAYS[month]
    return WeatherYear(rain, tmax, tmin, rs, et0, clim, scenario.forecast_skill)
