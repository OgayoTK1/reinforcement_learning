"""REINFORCE (Monte-Carlo policy gradient). Whole episodes are rolled out, the discounted return-to-go
is computed from the actual rewards, and there is no learned critic anywhere. Only TorchRL's environment
wrapper, rollout and ProbabilisticActor are used; the loss is written out because TorchRL ships none
that is free of a value function."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from tensordict.nn import InteractionType, TensorDictModule
from torch.distributions import Categorical
from torchrl.modules import ProbabilisticActor

from environment import AgriResilienceEnv, N_ACTIONS
from environment.params import HORIZON
from evaluation.evaluator import evaluate_policy
from training.splits import VAL_SEEDS

from .common import RunLogger, make_env, make_net, seed_everything
from .loader import CheckpointPolicy, save_checkpoint

DEFAULTS = dict(
    lr=1e-3, gamma=0.99, episodes_per_update=16, ent_coef=0.01, baseline="batch_mean", normalize=True,
    hidden=128, depth=2, activation="tanh", max_grad_norm=1.0, eval_every_frac=0.1,
)


def returns_to_go(rewards: np.ndarray, gamma: float) -> np.ndarray:
    g, out = 0.0, np.zeros_like(rewards)
    for t in range(len(rewards) - 1, -1, -1):
        g = rewards[t] + gamma * g
        out[t] = g
    return out


def train(cfg: dict, run_dir: Path, log: RunLogger) -> dict:
    h = {**DEFAULTS, **cfg}
    seed_everything(h["seed"])
    total = int(h["total_steps"])
    env = make_env(seed=h["seed"])
    net = make_net(N_ACTIONS, h["hidden"], h["depth"], h["activation"], last_gain=0.01)
    actor = ProbabilisticActor(TensorDictModule(net, in_keys=["observation"], out_keys=["logits"]),
                               in_keys=["logits"], spec=env.action_spec, distribution_class=Categorical,
                               default_interaction_type=InteractionType.RANDOM, return_log_prob=False)
    opt = torch.optim.Adam(net.parameters(), lr=h["lr"])
    eval_env = AgriResilienceEnv()
    eval_every = max(1, int(h["eval_every_frac"] * total))
    next_eval, best, steps, n_updates = eval_every, -1e9, 0, 0
    meta = dict(algo="reinforce", hidden=h["hidden"], depth=h["depth"], activation=h["activation"], config=h)

    def do_eval():
        nonlocal best
        df = evaluate_policy(CheckpointPolicy(net), VAL_SEEDS, env=eval_env)
        r = float(df["return"].mean())
        log.log("eval", step=steps, val_return=r, val_return_std=float(df["return"].std()),
                val_yield_kg=float(df["harvested_kg"].mean()), val_failure=float(df["crop_failure"].mean()),
                val_invalid=float(df["invalid_actions"].mean()))
        if r > best:
            best = r
            save_checkpoint(run_dir, "best", net, dict(meta, step=steps, val_return=r))

    while steps < total:
        obs_l, act_l, ret_l = [], [], []
        for _ in range(h["episodes_per_update"]):
            with torch.no_grad():
                ro = env.rollout(max_steps=HORIZON + 2, policy=actor, break_when_any_done=True)
            rew = ro["next", "reward"].reshape(-1).numpy().astype(np.float64)
            obs_l.append(ro["observation"].reshape(-1, ro["observation"].shape[-1]))
            act_l.append(ro["action"].reshape(-1))
            ret_l.append(returns_to_go(rew, h["gamma"]))
            steps += len(rew)
            log.log("episodes", step=steps, ret=float(rew.sum()), length=len(rew))
        obs = torch.cat(obs_l)
        act = torch.cat(act_l)
        G = torch.as_tensor(np.concatenate(ret_l), dtype=torch.float32)
        if h["baseline"] == "batch_mean":
            G = G - G.mean()
        if h["normalize"]:
            G = G / (G.std() + 1e-8)
        logits = net(obs)
        dist = Categorical(logits=logits)
        logp = dist.log_prob(act)
        ent = dist.entropy().mean()
        loss = -(logp * G).mean() - h["ent_coef"] * ent
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), h["max_grad_norm"])
        opt.step()
        n_updates += 1
        log.log("updates", step=steps, policy_loss=float(loss), entropy=float(ent), grad_norm=float(gn),
                n_updates=n_updates)
        if steps >= next_eval:
            do_eval(); next_eval += eval_every
            log.flush()
    if not log.tables.get("eval") or log.tables["eval"][-1]["step"] < steps - eval_every // 2:
        do_eval()
    save_checkpoint(run_dir, "final", net, dict(meta, step=steps))
    return dict(best_val_return=best, steps=steps, updates=n_updates)
