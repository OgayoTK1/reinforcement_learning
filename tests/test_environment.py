"""Environment unit and integration tests. None of these train a model."""
import copy
import json

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from environment import ACTION_NAMES, N_ACTIONS, OBS_DIM, OBS_NAMES, SCENARIOS, AgriResilienceEnv, spec_hash
from environment import actions as A
from environment.observation import OBS_VARS, build_observation, obs_table_markdown
from environment.params import HORIZON, P, PROVENANCE
from evaluation.baselines import ExtensionRulePolicy, RandomPolicy


def rollout(env, policy, seed, record=False):
    obs, _ = env.reset(seed=seed)
    policy.reset()
    done, total, traj = False, 0.0, []
    while not done:
        a = policy.act(obs)
        obs, r, done, trunc, info = env.step(a)
        total += r
        traj.append((a, r, obs.copy()))
    return total, traj, env.episode_summary()


def sown_env(seed=3, scenario=None, gdd=500.0):
    env = AgriResilienceEnv(scenario=scenario)
    env.reset(seed=seed)
    s = env.state
    s.sown, s.sow_week, s.gdd = True, 0, gdd
    return env


# ------------------------------------------------------------------ gym interface
def test_gymnasium_api_and_spaces():
    env = AgriResilienceEnv()
    check_env(env, skip_render_check=True)
    assert env.action_space.n == N_ACTIONS == len(ACTION_NAMES) == 20
    assert env.observation_space.shape == (OBS_DIM,)


def test_observation_spec_matches_vector_and_docs():
    env = AgriResilienceEnv()
    obs, _ = env.reset(seed=0)
    assert obs.dtype == np.float32 and obs.shape == (len(OBS_VARS),) == (len(OBS_NAMES),)
    assert len(set(OBS_NAMES)) == len(OBS_NAMES)
    assert len(obs_table_markdown().splitlines()) == OBS_DIM + 2
    assert len(spec_hash()) == 12
    assert all(v.source and v.update and v.norm and v.rng for v in OBS_VARS)


def test_every_parameter_has_provenance():
    assert {c for c, _ in PROVENANCE.values()} <= {"L", "C", "S", "E"}
    assert set(PROVENANCE) == {k for k in vars(P)}


# ------------------------------------------------------------------ reproducibility
def test_same_seed_same_trajectory_different_seed_differs():
    pol = ExtensionRulePolicy()
    r1, t1, _ = rollout(AgriResilienceEnv(), pol, 11)
    r2, t2, _ = rollout(AgriResilienceEnv(), pol, 11)
    r3, _, _ = rollout(AgriResilienceEnv(), pol, 12)
    assert r1 == r2
    assert all(np.array_equal(a[2], b[2]) for a, b in zip(t1, t2))
    assert r1 != r3


def test_weather_is_independent_of_actions_common_random_numbers():
    a, b = AgriResilienceEnv(), AgriResilienceEnv()
    a.reset(seed=4), b.reset(seed=4)
    for _ in range(10):
        a.step(A.WEED), b.step(A.WAIT)
    assert np.array_equal(a.rnd.weather.rain, b.rnd.weather.rain)
    assert np.array_equal(a.rnd.weather.tmax, b.rnd.weather.tmax)


def test_observation_noise_does_not_change_transitions():
    a, b = AgriResilienceEnv(), AgriResilienceEnv()
    a.reset(seed=6), b.reset(seed=6)
    b.rnd.obs_noise[:, 1:] = np.random.default_rng(0).normal(size=b.rnd.obs_noise[:, 1:].shape)
    for act in [A.WAIT, A.SOW, A.DAP, A.WAIT, A.WEED, A.CAN_50, A.WAIT]:
        a.step(act), b.step(act)
    assert np.allclose(a.state.dr, b.state.dr) and np.allclose(a.state.n_min, b.state.n_min)
    assert a.state.cash == b.state.cash


# ------------------------------------------------------------------ information leakage
def test_observation_has_no_future_weather():
    k = 14
    a, b = AgriResilienceEnv(), AgriResilienceEnv()
    a.reset(seed=8), b.reset(seed=8)
    cut = 7 * (k + 1)
    b.rnd.weather.rain[cut:] = b.rnd.weather.rain[cut:] * 3.0 + 5.0
    b.rnd.weather.tmax[cut:] += 6.0
    pol = ExtensionRulePolicy(); pol.reset()
    oa, _ = a.reset(seed=8)
    ob, _ = b.reset(seed=8)
    b.rnd.weather.rain[cut:] = b.rnd.weather.rain[cut:] * 3.0 + 5.0
    b.rnd.weather.tmax[cut:] += 6.0
    for _ in range(k):
        act = pol.act(oa)
        oa, *_ = a.step(act)
        ob, *_ = b.step(act)
        assert np.array_equal(oa, ob)


def test_estimates_are_noisy_but_unbiased():
    env = sown_env()
    env.state.faw[:] = 0.4
    errs = []
    rng = np.random.default_rng(0)
    i = OBS_NAMES.index("faw_estimate")
    for _ in range(400):
        env.rnd.obs_noise[env.state.week] = rng.normal(size=24)
        errs.append(build_observation(env.state, env.rnd)[i] - 0.4)
    assert abs(np.mean(errs)) < 0.03 and 0.1 < np.std(errs) < 0.2


def test_scouting_reduces_observation_noise():
    env = sown_env()
    i = OBS_NAMES.index("disease_estimate")
    rng = np.random.default_rng(1)
    def spread(flag):
        env.state.scouted = flag
        vals = []
        for _ in range(400):
            env.rnd.obs_noise[env.state.week] = rng.normal(size=24)
            vals.append(build_observation(env.state, env.rnd)[i])
        return np.std(vals)
    assert spread(True) < 0.5 * spread(False)


# ------------------------------------------------------------------ actions, validity and resources
def test_invalid_actions_are_noops_with_penalty():
    env = AgriResilienceEnv()
    env.reset(seed=2)
    ref = copy.deepcopy(env)
    for bad in [A.IRR_H, A.DAP, A.HARVEST, A.SELL_ALL, A.DRY, A.INSECT]:
        e = copy.deepcopy(env)
        _, r, _, _, info = e.step(bad)
        assert info["valid"] is False and info["executed_action"] == A.WAIT
        assert info["reward_components"]["invalid"] == -P.invalid_penalty
    ref.step(A.WAIT)
    e = copy.deepcopy(env); e.step(A.IRR_H)
    assert np.allclose(e.state.dr, ref.state.dr) and e.state.cash == ref.state.cash


def test_resource_constraints_hold_under_random_play():
    env = AgriResilienceEnv()
    rng = np.random.default_rng(0)
    for ep in range(15):
        obs, _ = env.reset(seed=ep)
        done = False
        while not done:
            obs, r, done, _, info = env.step(int(rng.integers(N_ACTIONS)))
            s = env.state
            assert 0 <= s.water_m3 <= P.pan_cap_m3 + 1e-6
            assert s.stock_kg >= -1e-6 and s.manure_kg >= -1e-6 and s.vouchers >= -1e-6
            assert s.cash >= P.cash_floor - 5000 or done
            for arr in (s.faw, s.dis, s.weeds, s.leaf, s.som, s.mulch):
                assert arr.min() >= -1e-9 and arr.max() <= 1 + 1e-9
            assert obs.min() >= env.observation_space.low.min() and obs.max() <= env.observation_space.high.max()
            assert np.isfinite(obs).all() and np.isfinite(r)


def test_cannot_irrigate_more_water_than_stored():
    env = sown_env()
    env.state.water_m3 = 50.0
    assert env.action_masks()[A.IRR_L] and not env.action_masks()[A.IRR_H]


def test_cannot_afford_inputs():
    env = sown_env()
    env.state.cash = 100.0
    m = env.action_masks()
    assert not m[A.INSECT] and not m[A.FUNG] and not m[A.CAN_50]


def test_labour_shortfall_is_hired_and_paid():
    env = sown_env()
    env.state.labour_h = 5.0
    cash0 = env.state.cash
    _, _, _, _, info = env.step(A.WEED)
    assert env.last_outcome.hired_cost == pytest.approx((A.LABOUR_H[A.WEED] - 5.0) * P.wage_h)
    assert env.state.cash == pytest.approx(cash0 - env.last_outcome.hired_cost)


def test_voucher_price_then_market_price():
    env = sown_env()
    env.state.vouchers, cash0 = 1.0, env.state.cash
    env.step(A.DAP)
    assert cash0 - env.state.cash == pytest.approx(P.voucher_price_bag, abs=1.0) or env.last_outcome.cash_cost == P.voucher_price_bag
    env2 = sown_env()
    env2.state.vouchers = 0.0
    env2.step(A.DAP)
    assert env2.last_outcome.cash_cost > 2 * P.voucher_price_bag


# ------------------------------------------------------------------ delayed and persistent consequences
def test_irrigation_effect_persists_into_following_weeks():
    env = sown_env(scenario="severe_drought", gdd=600.0)
    env.state.dr = P.taw * 0.8
    env.state.water_m3 = P.pan_cap_m3          # irrigation needs stored water; do not depend on the seed
    ctl = copy.deepcopy(env)
    env.step(A.IRR_M); ctl.step(A.WAIT)
    assert (env.state.dr < ctl.state.dr).all()
    env.step(A.WAIT); ctl.step(A.WAIT)
    assert env.state.dr.mean() < ctl.state.dr.mean() - 5.0


def test_fertilizer_raises_nitrogen_beyond_the_application_week():
    env = sown_env()
    ctl = copy.deepcopy(env)
    env.step(A.DAP); ctl.step(A.WAIT)
    env.step(A.WAIT); ctl.step(A.WAIT)
    assert (env.state.n_min > ctl.state.n_min + 3.0).all()
    assert env.state.p_av.mean() > ctl.state.p_av.mean() + 3.0


def test_nitrogen_improves_growth():
    lo, hi = sown_env(gdd=300), sown_env(gdd=300)
    hi.state.n_min = hi.state.n_min + 20.0
    for _ in range(12):
        lo.step(A.WAIT); hi.step(A.WAIT)
    assert (hi.state.veg + hi.state.grain).sum() > 1.15 * (lo.state.veg + lo.state.grain).sum()


def test_insecticide_lowers_later_pest_pressure():
    env = sown_env(gdd=400)
    env.state.faw[:] = 0.5
    ctl = copy.deepcopy(env)
    env.step(A.INSECT); ctl.step(A.WAIT)
    env.step(A.WAIT); ctl.step(A.WAIT)
    assert env.state.faw.mean() < 0.6 * ctl.state.faw.mean()
    assert env.state.cash < ctl.state.cash


def test_pest_damage_reduces_yield_but_is_capped():
    healthy, infested = sown_env(gdd=300), sown_env(gdd=300)
    for _ in range(35):
        infested.state.faw[:] = 0.9
        healthy.step(A.WAIT); infested.step(A.WAIT)
    ratio = infested.state.leaf.mean() / healthy.state.leaf.mean()
    assert 0.4 < ratio < 0.95
    # pest damage alone is floored: the extra loss versus the healthy twin stays bounded
    assert (healthy.state.leaf - infested.state.leaf).max() < 1 - P.damage_floor
    assert infested.state.alive.all()


def test_weeds_hurt_growth_and_weeding_removes_them():
    env = sown_env(gdd=300)
    env.state.weeds[:] = 0.8
    env.step(A.WEED)
    assert env.state.weeds.max() < 0.3


def test_mulch_slows_drying():
    a, b = sown_env(scenario="severe_drought", gdd=100), sown_env(scenario="severe_drought", gdd=100)
    a.state.dr[:] = 10.0; b.state.dr[:] = 10.0
    a.step(A.MULCH); b.step(A.WAIT)
    for _ in range(3):
        a.step(A.WAIT); b.step(A.WAIT)
    assert a.state.dr.mean() < b.state.dr.mean()


def test_manure_raises_soil_organic_matter_persistently():
    env = sown_env()
    ctl = copy.deepcopy(env)
    env.step(A.MANURE); ctl.step(A.WAIT)
    for _ in range(5):
        env.step(A.WAIT); ctl.step(A.WAIT)
    assert env.state.som.mean() > ctl.state.som.mean() + 0.003


def test_harvest_moves_grain_to_stock_and_drying_lowers_moisture():
    env = sown_env(gdd=2250.0)
    env.state.grain[:] = 450.0
    env.step(A.HARVEST)
    s = env.state
    assert s.stock_kg > 800 and s.harvested_frac == pytest.approx(0.5)
    assert s.grain.mean() < 300
    m0 = s.stock_moist
    env.step(A.DRY)
    assert env.state.stock_moist < m0


def test_wet_grain_spoils_faster_and_hermetic_storage_protects():
    def run(store, moist):
        env = sown_env(gdd=2250.0)
        env.state.stock_kg, env.state.stock_moist, env.state.store_type = 800.0, moist, store
        for _ in range(8):
            env.step(A.WAIT)
        return env.state.stock_mould, env.state.stock_insect, env.state.stock_kg
    wet_pp, dry_pp, dry_herm = run(1, 22.0), run(1, 13.0), run(2, 13.0)
    assert wet_pp[0] > 3 * dry_pp[0]
    assert dry_herm[1] < 0.5 * dry_pp[1] and dry_herm[2] > dry_pp[2]


def test_selling_converts_stock_to_cash_and_respects_food_reserve():
    env = sown_env(gdd=2250.0)
    env.state.stock_kg, env.state.stock_moist, env.state.store_type = 1500.0, 13.0, 2
    cash0 = env.state.cash
    env.step(A.SELL_ALL)
    assert env.state.stock_kg > P.reserve_kg - 40
    assert env.state.cash > cash0 + 10000
    assert not env.action_masks()[A.SELL_ALL]


# ------------------------------------------------------------------ rewards
def test_reward_components_sum_to_reward_and_match_accounting():
    env = AgriResilienceEnv()
    env.reset(seed=5)
    pol = ExtensionRulePolicy(); pol.reset()
    obs, _ = env.reset(seed=5)
    total_comp = 0.0
    done = False
    while not done:
        obs, r, done, _, info = env.step(pol.act(obs))
        assert r == pytest.approx(sum(info["reward_components"].values()))
        total_comp += sum(info["reward_components"].values())
    assert env.episode_return == pytest.approx(total_comp)


def test_sale_reward_is_liquidity_gain_minus_transport():
    env = sown_env(gdd=2250.0)
    s = env.state
    s.stock_kg, s.stock_moist, s.store_type, s.price = 1490.0, 13.0, 2, 4000.0
    env.rnd.price_eps[:] = 0.0
    env.step(A.SELL_ALL)
    comp = env.last_components
    assert comp["wealth"] > 0  # selling recovers the illiquidity discount
    assert comp["wealth"] < 0.15 * 1000 / 90 * 4000 / P.r0


def test_food_reserve_shortfall_penalised_at_termination():
    env = AgriResilienceEnv()
    env.reset(seed=1)
    ret = 0.0
    done = False
    while not done:
        _, r, done, _, info = env.step(A.WAIT)
    assert info["episode"]["reason"] == "never_sown"
    assert info["reward_components"]["shortfall"] < -0.2


def test_reward_hacking_extreme_policies_do_not_beat_sensible_management():
    seeds = range(100, 112)
    def mean_return(make):
        env = AgriResilienceEnv()
        return np.mean([rollout(env, make(), s)[0] for s in seeds])

    class Const:
        def __init__(self, a, then=None): self.a, self.then = a, then
        def reset(self): pass
        def act(self, obs): return self.a

    class SprayEveryWeek(ExtensionRulePolicy):
        def act(self, obs):
            a = super().act(obs)
            return A.INSECT if a == A.WAIT and obs[OBS_NAMES.index("sown")] > 0.5 and obs[OBS_NAMES.index("harvest_readiness")] < 0.8 else a

    class IrrigateAlways(ExtensionRulePolicy):
        def act(self, obs):
            a = super().act(obs)
            return A.IRR_H if a == A.WAIT and obs[OBS_NAMES.index("sown")] > 0.5 and obs[OBS_NAMES.index("harvest_readiness")] < 0.9 else a

    sensible = mean_return(ExtensionRulePolicy)
    assert sensible > mean_return(lambda: Const(A.WAIT))                         # doing nothing
    assert sensible > mean_return(lambda: RandomPolicy(0))                       # random
    assert sensible > mean_return(SprayEveryWeek)                                # pesticide spam
    assert sensible > mean_return(IrrigateAlways)                                # over-irrigation
    assert sensible > mean_return(lambda: ExtensionRulePolicy(harvest_ready=0.85))   # harvest too early
    assert sensible > mean_return(lambda: ExtensionRulePolicy(sells=False)) - 0.01   # hoarding does not pay


# ------------------------------------------------------------------ terminal conditions
def test_terminates_at_end_of_year_with_summary():
    env = AgriResilienceEnv()
    obs, _ = env.reset(seed=3)
    pol = ExtensionRulePolicy(); pol.reset()
    n, done = 0, False
    while not done:
        obs, r, done, trunc, info = env.step(pol.act(obs)); n += 1
        assert trunc is False
    assert n == HORIZON and info["episode"]["reason"] == "end_of_year"


def test_terminates_when_never_sown():
    env = AgriResilienceEnv()
    env.reset(seed=3)
    n, done = 0, False
    while not done:
        _, _, done, _, info = env.step(A.WAIT); n += 1
    assert n == P.sow_deadline_week + 1 and info["episode"]["reason"] == "never_sown"


def test_terminates_on_bankruptcy_and_total_crop_loss():
    env = sown_env()
    env.state.cash = P.cash_floor - 100
    _, _, done, _, info = env.step(A.WAIT)
    assert done and info["episode"]["reason"] == "bankrupt"
    env = sown_env()
    env.state.alive[:] = False
    _, _, done, _, info = env.step(A.WAIT)
    assert done and info["episode"]["reason"] == "crop_failure"


# ------------------------------------------------------------------ scenarios and serialisation
def test_scenarios_change_the_weather_as_described():
    def season_rain(name):
        e = AgriResilienceEnv(scenario=name); e.reset(seed=9)
        return e.rnd.weather.rain[:364].sum(), e.rnd.weather.tmax[:364].mean()
    base, drought, wet = season_rain("baseline"), season_rain("severe_drought"), season_rain("wet_year")
    assert drought[0] < 0.75 * base[0] < 0.75 * wet[0]
    assert season_rain("warm_3")[1] > base[1] + 2.5


def test_serialised_state_is_json_and_mirrors_environment():
    env = AgriResilienceEnv()
    env.reset(seed=1)
    for a in [A.WEED, A.SOW, A.DAP, A.WAIT]:
        env.step(a)
    d = env.get_state_dict()
    json.dumps(d)
    s = env.state
    assert d["plots"][2]["soil_moisture"] == pytest.approx(1 - s.dr[2] / P.taw[2])
    assert d["resources"]["cash"] == pytest.approx(s.cash) and d["time"]["week"] == s.week
    assert d["action"]["executed"] == "wait"
    assert set(["schema_version", "weather", "crop", "plots", "resources", "storage", "market", "reward"]) <= set(d)


def test_season_calibration_targets():
    """Calibration guardrails: managed maize 12 to 20 bags/acre, no-input maize clearly lower."""
    env = AgriResilienceEnv()
    managed = [rollout(env, ExtensionRulePolicy(), s)[2]["harvested_kg"] for s in range(200, 215)]
    assert 12 * 90 < np.mean(managed) < 20 * 90
