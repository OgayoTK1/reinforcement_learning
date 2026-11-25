"""Reference policies. Both act on the same observation vector as the learners.

RandomPolicy picks uniformly from all 20 actions (invalid ones become no-ops).

ExtensionRulePolicy encodes a conventional agronomic calendar for highland
maize in plain rules. It never reads hidden state, only the observation.
"""
from __future__ import annotations

import numpy as np

from environment.actions import (CAN_25, CAN_50, DAP, DRY, FUNG, HARVEST, INSECT, IRR_M, MANURE, N_ACTIONS,
                                 SCOUT, SELL_ALL, SELL_HALF, SOW, STORE_HERM, WAIT, WEED)
from environment.observation import OBS_NAMES

IDX = {n: i for i, n in enumerate(OBS_NAMES)}


class RandomPolicy:
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray) -> int:
        return int(self.rng.integers(N_ACTIONS))


class ExtensionRulePolicy:
    """Calendar-and-threshold management, written the way an extension leaflet would put it."""

    name = "rule_based"

    def __init__(self, harvest_ready: float = 1.12, sells: bool = True):
        self.harvest_ready = harvest_ready
        self.sells = sells
        self.weeded_prep = False

    def reset(self) -> None:
        self.weeded_prep = False

    def act(self, obs: np.ndarray) -> int:
        g = lambda k: float(obs[IDX[k]])
        week = g("week_frac") * 52
        sown = g("sown") > 0.5
        age = g("weeks_since_sowing") * 40.0
        stage = [n for n in ("unsown", "vegetative", "flowering", "grain_fill", "dry_down", "harvested")
                 if g(f"stage_{n}") > 0.5][0]
        stock = g("grain_stock") * 2500.0
        store_none = g("store_none") > 0.5

        # land preparation and sowing at rain onset
        if not sown:
            if week < 1 and not self.weeded_prep:
                self.weeded_prep = True
                return WEED
            if g("rain_last_4_weeks") * 200.0 >= 70.0 or week >= 4:
                return SOW
            return WAIT

        # prepare hermetic bags before the crop is cut
        if store_none and g("harvest_readiness") >= 1.05:
            return STORE_HERM

        # post-harvest handling
        if stage in ("dry_down", "harvested") or g("harvested_fraction") > 0:
            if g("harvested_fraction") < 0.999 and stage == "dry_down" and g("harvest_readiness") >= self.harvest_ready:
                return HARVEST
            if stock > 50:
                moist = g("grain_moisture") * 20.0 + 10.0
                if moist > 14.2:
                    return DRY
                if store_none:
                    return STORE_HERM
                if self.sells and week >= 49:
                    return SELL_ALL
                if self.sells and g("maize_price") >= 1.04:
                    return SELL_HALF
            return WAIT if stage != "dry_down" else (HARVEST if g("harvest_readiness") >= self.harvest_ready else WAIT)

        # crop care
        if g("n_applied") == 0 and age >= 0:
            return DAP
        if age >= 2 and g("manure_applications") * 3.0 < 1:
            return MANURE
        if stage == "vegetative":
            if g("weed_estimate") > 0.45:
                return WEED
            if age >= 5 and g("n_applied") * 60.0 < 22:
                return CAN_50
            if age >= 10 and g("n_applied") * 60.0 < 34.5:
                return CAN_25
        if stage in ("vegetative", "flowering", "grain_fill"):
            if g("weeks_since_scout") >= 3 / 8:
                return SCOUT
            if g("faw_estimate") > 0.22 and g("weeks_since_insecticide") >= 2 / 8 and stage == "vegetative":
                return INSECT
            if g("disease_estimate") > 0.30 and g("weeks_since_fungicide") >= 2 / 8:
                return FUNG
            moist = g("soil_moisture_mean")
            if moist < 0.30 and g("rain_forecast_next_week") * 60.0 < 12 and g("water_level") > 0.25 \
                    and stage in ("vegetative", "flowering", "grain_fill"):
                return IRR_M
        return WAIT
