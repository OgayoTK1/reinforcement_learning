"""Fixed, disjoint episode-seed ranges for the three evaluation splits.

Training episodes draw seeds >= 10_000 (environment default), so neither
validation nor test episodes can ever appear in training.
"""
VAL_SEEDS = list(range(1, 51))          # 50 episodes, in-distribution weather, model selection only
TEST_SEEDS = list(range(1000, 1200))    # 200 episodes, in-distribution, evaluated once per final model
OOD_SEEDS = list(range(2000, 2050))     # 50 episodes per out-of-distribution scenario
OOD_SCENARIOS = ["severe_drought", "late_onset", "wet_year", "warm_1p5", "warm_3", "poor_forecast", "drought_warm"]
