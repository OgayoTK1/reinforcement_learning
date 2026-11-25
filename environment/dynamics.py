"""State transition functions.

Each function mutates a FarmState given the pre-drawn episode randomness.
The weekly step runs seven daily sub-steps (soil water, crop, pests, nutrients)
followed by weekly processes (events, storage, market, labour).

Method references: FAO-56 single crop coefficient water balance, SCS curve
number runoff, Hargreaves ET0, radiation-use-efficiency biomass, thermal-time
phenology. Coefficients are tagged in params.py.
"""
from __future__ import annotations

import numpy as np
from numpy._core.umath import clip as _clip

from .params import P
from .state import EpisodeRandomness, FarmState, base_price

_EXP = np.exp


def lai_potential(gdd: float) -> float:
    return float(np.interp(gdd, P.lai_gdd, P.lai_val))


def kc_curve(gdd: float) -> float:
    return float(np.interp(gdd, P.kc_gdd, P.kc_val))


def stage_of(s: FarmState) -> int:
    """0 unsown, 1 vegetative, 2 flowering, 3 grain fill, 4 dry-down, 5 harvested."""
    if not s.sown:
        return 0
    if s.harvested_frac > 0.999 or (s.gdd >= P.gdd_mat and float(s.grain.sum()) < 1.0):
        return 5
    g = s.gdd
    if g < P.gdd_silk - 100:
        return 1
    if g < P.gdd_silk + 150:
        return 2
    if g < P.gdd_mat:
        return 3
    return 4


def simulate_day(s: FarmState, rnd: EpisodeRandomness, d: int) -> None:
    w = rnd.weather
    rain, tmax, tmin, rs, et0 = w.rain[d], w.tmax[d], w.tmin[d], w.rs[d], w.et0[d]
    tmean = 0.5 * (tmax + tmin)
    sown = s.sown
    gdd = s.gdd
    taw = P.taw

    # --- farm pan
    s.water_m3 = float(_clip(s.water_m3 + rain * P.catch_m2 / 1000.0 - et0 * P.pan_evap_k * P.pan_area_m2 / 1000.0,
                               0.0, P.pan_cap_m3))

    # --- phenology
    gdd_inc = 0.0
    if sown and gdd < P.gdd_end:
        gdd_inc = max(0.0, 0.5 * (min(tmax, P.tcap) + max(tmin, P.tbase)) - P.tbase)

    lai_p = lai_potential(gdd) if sown else 0.0
    cover_c = 1.0 - _EXP(-P.k_ext * lai_p * s.leaf)
    cover = np.maximum(s.mulch, cover_c)

    # --- runoff (SCS curve number) and water balance (FAO-56)
    cn = P.cn - P.cn_cover * cover
    S = 25400.0 / cn - 254.0
    ia = 0.2 * S
    q = np.where(rain > ia, (rain - ia) ** 2 / (rain + 0.8 * S), 0.0)
    eff_rain = rain - q
    kc = kc_curve(gdd) if sown else P.kc_bare
    raw = P.p_depl * taw
    ks = np.where(s.dr <= raw, 1.0, _clip((taw - s.dr) / ((1.0 - P.p_depl) * taw), 0.0, 1.0))
    kc_eff = kc * (1.0 - P.mulch_kc * s.mulch * (1.0 - cover_c))
    etc = kc_eff * ks * et0
    dr = s.dr + etc - eff_rain
    drain = np.maximum(0.0, -dr)
    s.dr = _clip(dr, 0.0, taw)
    theta = 1.0 - s.dr / taw

    # --- soil nutrients and organic matter
    fm = float(_clip((tmean - 5.0) / 15.0, 0.2, 1.2)) * (0.3 + 0.7 * theta)
    mineral = P.kmin * s.som * fm
    s.n_min = s.n_min + mineral
    s.som = s.som - P.som_per_n * mineral - P.erosion_k * q * (1.0 - cover) * P.slope_k
    s.n_min = s.n_min * (1.0 - np.minimum(0.6, P.leach_k * drain))
    s.p_av = s.p_av + P.p_release
    s.mulch = s.mulch * _EXP(-P.mulch_decay)
    # weeds grow with moisture and warmth
    s.weeds = _clip(s.weeds + P.weed_r * (0.3 + 0.7 * theta) * (s.weeds + 0.01) * (1.0 - s.weeds), 0.0, 1.0)

    if sown and P.gdd_emerge <= gdd < P.gdd_end + 1e-9:
        active = s.alive & (gdd < P.gdd_end)
        lai = lai_p * s.leaf
        fint = 1.0 - _EXP(-P.k_ext * lai)
        f_t = float(_clip((tmean - 4.0) / 8.0, 0.0, 1.0))
        f_n = _clip(s.n_min / P.n_crit, P.stress_floor, 1.0)
        f_p = _clip(s.p_av / P.p_crit, P.stress_floor, 1.0)
        f_nut = np.minimum(f_n, f_p)
        crit = 1.0 if gdd < P.gdd_weed_crit else 0.6
        f_weed = 1.0 - P.weed_comp * s.weeds * crit
        stress = np.minimum(ks, f_nut)
        db = P.rue * rs * fint * f_t * stress * f_weed * active

        # uptake from the soil pools
        s.n_min = s.n_min - np.minimum(s.n_min, db * P.n_cont * P.g2kgac)
        s.p_av = s.p_av - np.minimum(s.p_av, db * P.p_cont * P.g2kgac)

        # kernel set during the flowering window
        in_win = active & (gdd >= P.gdd_silk - P.kernel_win) & (gdd <= P.gdd_silk + P.kernel_win)
        s.kernel_num = s.kernel_num + np.where(in_win, stress * gdd_inc, 0.0)
        s.kernel_den = s.kernel_den + np.where(in_win, gdd_inc, 0.0)
        kset = np.where(s.kernel_den > 0, s.kernel_num / np.maximum(s.kernel_den, 1e-9), 1.0)
        kset = _clip(kset, 0.1, 1.0)

        # partition to grain after silking
        fill = active & (gdd >= P.gdd_silk) & (gdd < P.gdd_mat)
        span = P.gdd_mat - P.gdd_silk
        sink = P.sink_max * kset / span * gdd_inc
        remob = P.remob * s.veg / span * gdd_inc
        g_inc = np.where(fill, np.minimum(sink, P.partition * db + remob), 0.0)
        s.grain = s.grain + g_inc
        s.veg = np.maximum(0.0, s.veg + db - g_inc)

        # pests, disease
        if gdd < 950:
            susc = 1.0
        elif gdd < 1500:
            susc = 0.5
        else:
            susc = 0.15
        warm = float(_clip((tmean - 10.0) / 10.0, 0.2, 1.2))
        prot = 0.4 if s.faw_protect_days > 0 else 1.0
        s.faw = _clip(s.faw + P.faw_g * susc * warm * prot * s.faw * (1.0 - s.faw) - P.faw_mort * s.faw, 0.0, 1.0)
        s.wet_ewma = 0.85 * s.wet_ewma + 0.15 * (1.0 if rain > 1.0 else 0.0)
        fav = s.wet_ewma * float(_EXP(-((tmean - P.dis_topt) / P.dis_width) ** 2))
        dprot = 0.15 if s.fung_protect_days > 0 else 1.0
        s.dis = _clip(s.dis + (P.dis_r * fav * s.dis * (1.0 - s.dis) + P.dis_seed * fav) * dprot, 0.0, 1.0)

        # canopy
        veg_phase = 1.0 if gdd < P.gdd_silk + 300 else 0.0
        drought_sen = P.drought_leaf * np.maximum(0.0, 0.6 - ks) / 0.6
        # pest and disease damage saturates (observed Kenyan FAW losses cap near 47 percent); drought can still kill
        room = _clip((s.leaf - P.damage_floor) / (1.0 - P.damage_floor), 0.0, 1.0)
        dleaf = (P.leaf_regrow * (1.0 - s.leaf) * veg_phase
                 - (P.faw_leaf * s.faw * susc + P.dis_leaf * s.dis) * s.leaf * room - drought_sen * s.leaf)
        s.leaf = _clip(s.leaf + dleaf, 0.0, 1.0)

        s.drought_days = np.where(ks < P.ks_dead, s.drought_days + 1, 0.0)
        s.alive = s.alive & (s.drought_days < 14) & (s.leaf > 0.03)
        s.stress = 0.9 * s.stress + 0.1 * stress

    if s.faw_protect_days > 0:
        s.faw_protect_days -= 1
    if s.fung_protect_days > 0:
        s.fung_protect_days -= 1
    s.gdd = gdd + gdd_inc
    if sown and gdd < P.gdd_mat:
        s.season_rain_mm += float(rain)


def weekly_events(s: FarmState, rnd: EpisodeRandomness, w: int) -> None:
    """Processes that run once at the start of week w, before the daily loop."""
    # fall armyworm flights
    if s.sown and s.gdd >= P.gdd_emerge and s.gdd < 1500 and rnd.imm_u[w] < P.faw_imm_p:
        s.faw = _clip(s.faw + rnd.imm_amt[w] * s.alive, 0.0, 1.0)


def refresh_labour(s: FarmState, rnd: EpisodeRandomness, w: int) -> None:
    """Family labour available in week w (autoregressive, with occasional shocks)."""
    s.labour_f = P.labour_ar * s.labour_f + (1 - P.labour_ar) * rnd.labour_u[w]
    f = s.labour_f * (0.4 if rnd.shock_u[w] < P.labour_shock_p else 1.0)
    s.labour_h = P.labour_cap_h * f


def end_of_week(s: FarmState, rnd: EpisodeRandomness, w: int) -> None:
    """Storage, field losses and market, after the daily loop of week w."""
    d0 = 7 * w
    rain_w = rnd.weather.rain[d0:d0 + 7]
    rain_days = int((rain_w > 1.0).sum())
    s.rain_last = float(rain_w.sum())
    s.rain_days_last = rain_days
    s.tmax_last = float(rnd.weather.tmax[d0:d0 + 7].mean())
    s.rain_hist = (s.rain_hist + [s.rain_last])[-4:]

    # standing crop losses after maturity
    if s.sown and s.gdd >= P.gdd_mat and float(s.grain.sum()) > 0:
        loss = P.field_loss_base + P.field_loss_rain * rain_days
        s.grain = s.grain * (1.0 - loss)

    # stored grain
    if s.stock_kg > 0:
        st = int(s.store_type)
        humid = 1.3 if rain_days >= 3 else 1.0
        over = max(0.0, s.stock_moist - P.safe_moist)
        d_ins = P.insect_rate[st] + P.insect_age * s.store_weeks
        s.stock_insect = min(0.95, s.stock_insect + d_ins)
        s.stock_mould = min(1.0, s.stock_mould + P.mould_rate[st] * over * humid + 0.0005 * humid)
        loss_frac = 0.6 * d_ins + (P.none_loss if st == 0 else 0.0)
        lost = s.stock_kg * loss_frac
        s.stock_kg -= lost
        s.ph_loss_kg += lost
        # moisture drifts toward ambient equilibrium
        target = 13.5 + (1.0 if rain_days >= 3 else 0.0) if st < 2 else s.stock_moist
        if st < 2:
            s.stock_moist += 0.15 * (target - s.stock_moist)
        s.store_weeks += 1
    else:
        s.store_weeks = 0

    # market (week w+1 prices are visible at the next decision)
    nxt = w + 1
    s.price_x = P.price_ar * s.price_x + P.price_sd * rnd.price_eps[nxt]
    s.price = base_price(nxt, s.price_x)
    s.fert_x = P.fert_ar * s.fert_x + P.fert_sd * rnd.fert_eps[nxt]
    s.dap_kg = P.dap_market_bag / 50.0 * float(_EXP(s.fert_x))
    s.can_kg = P.can_market_bag / 50.0 * float(_EXP(s.fert_x))
    s.price_hist = (s.price_hist + [s.price])[-5:]

    s.weeks_since_scout += 1
    s.weeks_since_insecticide += 1
    s.weeks_since_fungicide += 1
    s.manure_kg += P.manure_weekly_kg
    s.week = w + 1
    s.day = 7 * (w + 1)
    if nxt < rnd.labour_u.shape[0]:
        refresh_labour(s, rnd, nxt)


def advance_week(s: FarmState, rnd: EpisodeRandomness) -> None:
    """Run the weekly processes and the seven daily sub-steps for the current week."""
    w = s.week
    weekly_events(s, rnd, w)
    for d in range(7 * w, 7 * w + 7):
        simulate_day(s, rnd, d)
    end_of_week(s, rnd, w)


def forecast_next_week(s: FarmState, rnd: EpisodeRandomness) -> float:
    """Noisy forecast of next week's rainfall. Skill mixes the truth with climatology."""
    d0 = 7 * s.week
    wx = rnd.weather
    truth = float(wx.rain[d0:d0 + 7].sum())
    clim = float(wx.clim_rain[d0:d0 + 7].sum())
    k = wx.forecast_skill
    noise = rnd.obs_noise[min(s.week, rnd.obs_noise.shape[0] - 1), 0] * 15.0 * (1.0 - k)
    return max(0.0, k * truth + (1.0 - k) * clim + noise)


def standing_grain_kg(s: FarmState) -> float:
    return float(np.sum(s.grain * P.area_plot_m2 / 1000.0) * P.dm_to_std)
