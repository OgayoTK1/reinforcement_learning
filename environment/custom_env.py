"""AgriResilienceEnv: smallholder maize management at Njoro, Nakuru County.

Authoritative simulation source of truth. It runs without a browser, is fully
deterministic for a given seed and scenario, and exposes the standard
Gymnasium reset() / step() interface.
"""
from __future__ import annotations

import json
import logging

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import dynamics
from .actions import ACTION_NAMES, N_ACTIONS, SCOUT, apply_action, valid_mask
from .observation import OBS_DIM, OBS_HIGH, OBS_LOW, build_observation, spec_hash
from .params import ENV_VERSION, HORIZON, P
from .rewards import shortfall_cost, soil_index, step_reward, wealth
from .state import EpisodeRandomness, FarmState, draw_episode, initial_state
from .weather import SCENARIOS, Scenario

log = logging.getLogger("agri.env")


class AgriResilienceEnv(gym.Env):
    metadata = {"render_modes": ["ansi", "json"], "render_fps": 4}

    def __init__(self, scenario: Scenario | str | None = None, render_mode: str | None = None):
        super().__init__()
        self.observation_space = spaces.Box(OBS_LOW, OBS_HIGH, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.render_mode = render_mode
        self._fixed_scenario = SCENARIOS[scenario] if isinstance(scenario, str) else scenario
        self.state: FarmState | None = None
        self.rnd: EpisodeRandomness | None = None
        self.episode_return = 0.0
        self.history: list[dict] = []
        self.record_history = False
        self.last_outcome = None
        self.last_components: dict = {}

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        # train draws use seeds >= 10_000, validation 1..99, test 1000..1999 (see training.splits)
        episode_seed = options.get("episode_seed")
        if episode_seed is None:
            episode_seed = int(self.np_random.integers(10_000, 2**31 - 1))
        scenario = options.get("scenario", self._fixed_scenario)
        if isinstance(scenario, str):
            scenario = SCENARIOS[scenario]
        self.rnd = draw_episode(episode_seed, scenario)
        self.state = initial_state(self.rnd)
        dynamics.refresh_labour(self.state, self.rnd, 0)
        self.episode_return = 0.0
        self.history = []
        self.last_outcome = None
        self.last_components = {}
        obs = build_observation(self.state, self.rnd)
        return obs, self._info(None)

    def step(self, action):
        s = self.state
        assert s is not None, "call reset() first"
        a = int(action)
        prev_w, prev_soil = wealth(s), soil_index(s)
        s.scouted = False
        out = apply_action(s, a)
        s.last_action = out.executed
        dynamics.advance_week(s, self.rnd)
        terminated, reason = self._terminal(s)
        s.terminated_reason = reason
        reward, comps = step_reward(prev_w, prev_soil, s, out, terminated)
        self.episode_return += reward
        self.last_outcome, self.last_components = out, comps
        obs = build_observation(s, self.rnd)
        info = self._info(out)
        info["reward_components"] = comps
        if terminated:
            info["episode"] = self.episode_summary()
        if self.record_history:
            self.history.append({"week": s.week, "action": ACTION_NAMES[out.executed], "reward": reward,
                                 "valid": out.valid, "detail": out.detail})
        if log.isEnabledFor(logging.DEBUG):
            log.debug(json.dumps({"week": s.week, "action": ACTION_NAMES[out.executed], "valid": out.valid,
                                  "reward": round(reward, 5), "cash": round(s.cash)}))
        return obs, float(reward), terminated, False, info

    # ------------------------------------------------------------------ helpers
    def _terminal(self, s: FarmState) -> tuple[bool, str]:
        if s.week >= HORIZON:
            return True, "end_of_year"
        if not s.sown and s.week > P.sow_deadline_week:
            return True, "never_sown"
        if s.sown and s.harvested_frac < 0.999 and not bool(s.alive.any()):
            return True, "crop_failure"
        if s.cash < P.cash_floor:
            return True, "bankrupt"
        return False, ""

    def _info(self, out) -> dict:
        s = self.state
        info = {
            "week": s.week,
            "scenario": self.rnd.scenario.name,
            "action_mask": valid_mask(s),
            "stage": dynamics.stage_of(s),
        }
        if out is not None:
            info.update(executed_action=out.executed, valid=out.valid, reason=out.reason, detail=out.detail)
        return info

    def action_masks(self) -> np.ndarray:
        return valid_mask(self.state)

    def episode_summary(self) -> dict:
        """Evaluation metrics for the (possibly unfinished) episode."""
        s = self.state
        q = max(0.0, 1.0)
        from .actions import quality_factor
        qf = quality_factor(s)
        stock_value = s.stock_kg / P.bag_kg * s.price * qf
        harvested = s.harvested_kg
        marketable = s.sold_kg + s.stock_kg * qf
        total_water_m3 = (s.season_rain_mm + 0.0) * P.acre_m2 / 1000.0 + s.water_used_m3
        yield_kg_acre = harvested
        failure = harvested < 500.0
        net_return = s.cash + stock_value - float(self._start_cash())
        return {
            "return": self.episode_return,
            "harvested_kg": harvested,
            "marketable_kg": marketable,
            "marketable_bags": marketable / P.bag_kg,
            "yield_kg_acre": yield_kg_acre,
            "crop_survival": float(s.alive.mean()) if s.sown else 0.0,
            "crop_failure": float(failure),
            "net_return_kes": net_return,
            "water_used_m3": s.water_used_m3,
            "water_use_efficiency_kg_m3": harvested / max(1.0, total_water_m3),
            "soil_health": float(s.som.mean()),
            "soil_change": float(s.som.mean() - s.som_start),
            "post_harvest_loss_frac": s.ph_loss_kg / max(1.0, harvested),
            "spoilage_mould": s.stock_mould,
            "final_cash": s.cash,
            "stock_kg": s.stock_kg,
            "food_reserve_met": float(s.stock_kg >= 0.9 * P.reserve_kg),   # within 10% of the 490 kg household reserve
            "reserve_shortfall_kg": max(0.0, P.reserve_kg - s.stock_kg),
            "invalid_actions": s.invalid_actions,
            "reason": s.terminated_reason,
            "sown_week": s.sow_week,
        }

    def _start_cash(self) -> float:
        return float(self.rnd.init["cash"])

    # ------------------------------------------------------------------ serialisation / rendering
    def get_state_dict(self) -> dict:
        from .rendering import serialize_state
        return serialize_state(self)

    def render(self):
        from .rendering import render_ansi, serialize_state
        if self.render_mode == "ansi":
            return render_ansi(self)
        if self.render_mode == "json":
            return json.dumps(serialize_state(self))
        return None

    @property
    def spec_hash(self) -> str:
        return spec_hash()

    @property
    def version(self) -> str:
        return ENV_VERSION
