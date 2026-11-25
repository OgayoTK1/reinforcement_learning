"""Agent plumbing tests. Training budgets are tiny: these check that the code runs and that checkpoints
round-trip, not that anything has been learned."""
import json
import warnings

import numpy as np
import pytest
import torch

from agents.loader import load_policy
from environment import N_ACTIONS, OBS_DIM, AgriResilienceEnv
from evaluation.baselines import ExtensionRulePolicy, RandomPolicy
from evaluation.evaluator import evaluate_policy
from experiments.matrix import ALGOS, configs
from training.splits import OOD_SEEDS, TEST_SEEDS, VAL_SEEDS
from training.train_one import train_one

warnings.filterwarnings("ignore")
FAST = {
    "dqn": dict(learn_start=100, frames_per_batch=32),
    "reinforce": dict(episodes_per_update=3),
    "ppo": dict(frames_per_batch=128, minibatch=64, epochs=2),
    "a2c": dict(frames_per_batch=32),
}


def test_matrix_has_ten_distinct_configs_per_algorithm():
    cs = configs(seeds=(0,))
    for algo in ALGOS:
        mine = [c for c in cs if c["algo"] == algo]
        assert len(mine) >= 10
        keys = [json.dumps({k: v for k, v in c.items() if k not in ("name", "factor")}, sort_keys=True) for c in mine]
        assert len(set(keys)) == len(keys)               # no duplicated configuration


def test_splits_are_disjoint():
    assert not set(VAL_SEEDS) & set(TEST_SEEDS) and not set(TEST_SEEDS) & set(OOD_SEEDS)
    assert max(VAL_SEEDS) < 1000 and min(OOD_SEEDS) > max(TEST_SEEDS)
    env = AgriResilienceEnv()
    env.reset(seed=1)                                  # no explicit episode seed: a training draw
    assert env.rnd.seed >= 10_000
    env.reset(seed=1, options={"episode_seed": 7})
    assert env.rnd.seed == 7


@pytest.mark.parametrize("algo", ALGOS)
def test_each_algorithm_trains_checkpoints_and_reloads(algo, tmp_path):
    res = train_one(dict(algo=algo, name="t", seed=0, total_steps=600, eval_every_frac=0.5, **FAST[algo]), tmp_path / algo)
    assert res["status"] == "ok", res.get("error")
    for f in ("best.pt", "best.json", "final.pt", "episodes.csv", "eval.csv", "updates.csv", "result.json"):
        assert (tmp_path / algo / f).exists(), f
    pol = load_policy(tmp_path / algo / "best.pt")
    obs, _ = AgriResilienceEnv().reset(seed=1)
    assert 0 <= pol.act(obs) < N_ACTIONS
    assert pol.act(obs) == pol.act(obs)                      # greedy playback is deterministic
    a = evaluate_policy(pol, [1, 2, 3])
    b = evaluate_policy(pol, [1, 2, 3])
    assert np.allclose(a["return"], b["return"])


def test_training_is_reproducible_for_a_seed(tmp_path):
    cfg = dict(algo="reinforce", name="t", seed=3, total_steps=500, eval_every_frac=1.0, episodes_per_update=3)
    r1 = train_one(cfg, tmp_path / "a")
    r2 = train_one(cfg, tmp_path / "b")
    assert r1["best_val_return"] == pytest.approx(r2["best_val_return"])


def test_checkpoint_with_wrong_observation_spec_is_refused(tmp_path):
    train_one(dict(algo="a2c", name="t", seed=0, total_steps=300, eval_every_frac=1.0, frames_per_batch=32), tmp_path)
    meta = json.loads((tmp_path / "best.json").read_text())
    meta["spec_hash"] = "deadbeef0000"
    (tmp_path / "best.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="observation"):
        load_policy(tmp_path / "best.pt")


def test_failures_are_recorded_not_hidden(tmp_path):
    res = train_one(dict(algo="ppo", name="bad", seed=0, total_steps=300, hidden=-5), tmp_path)
    assert res["status"] == "failed" and res["error"]
    assert json.loads((tmp_path / "result.json").read_text())["status"] == "failed"


def test_baselines_use_the_same_interface():
    for pol in (RandomPolicy(0), ExtensionRulePolicy()):
        df = evaluate_policy(pol, [5, 6])
        assert len(df) == 2 and np.isfinite(df["return"]).all()
