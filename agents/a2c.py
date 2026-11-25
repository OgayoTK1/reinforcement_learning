"""Advantage Actor-Critic with TorchRL: short on-policy rollouts, GAE advantages, one gradient step per rollout."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torchrl.objectives import A2CLoss

from environment import AgriResilienceEnv
from evaluation.evaluator import evaluate_policy
from training.splits import VAL_SEEDS

from ._actor_critic import build, episode_stats
from .common import RunLogger, seed_everything
from .loader import CheckpointPolicy, save_checkpoint

DEFAULTS = dict(
    lr=7e-4, gamma=0.99, lmbda=0.95, ent_coef=0.01, critic_coef=0.5, frames_per_batch=128, hidden=128, depth=2,
    activation="tanh", max_grad_norm=0.5, eval_every_frac=0.1,
)


def _entropy(out, actor, td) -> float:
    """Policy entropy for the log. TorchRL omits it from the loss output when the coefficient is 0."""
    if "entropy" in out.keys():
        return float(out["entropy"].mean())
    with torch.no_grad():
        return float(actor.get_dist(td).entropy().mean())


def train(cfg: dict, run_dir: Path, log: RunLogger) -> dict:
    h = {**DEFAULTS, **cfg}
    seed_everything(h["seed"])
    total = int(h["total_steps"])
    env, pnet, vnet, actor, critic, coll, gae = build(h)
    loss = A2CLoss(actor, critic, entropy_bonus=True, entropy_coeff=h["ent_coef"], critic_coeff=h["critic_coef"],
                   loss_critic_type="smooth_l1")
    params = list(pnet.parameters()) + list(vnet.parameters())
    opt = torch.optim.Adam(params, lr=h["lr"])
    eval_env = AgriResilienceEnv()
    eval_every = max(1, int(h["eval_every_frac"] * total))
    next_eval, best, steps, n_updates = eval_every, -1e9, 0, 0
    st = {"ret": 0.0, "len": 0}
    meta = dict(algo="a2c", hidden=h["hidden"], depth=h["depth"], activation=h["activation"], config=h)

    def do_eval():
        nonlocal best
        df = evaluate_policy(CheckpointPolicy(pnet), VAL_SEEDS, env=eval_env)
        r = float(df["return"].mean())
        log.log("eval", step=steps, val_return=r, val_return_std=float(df["return"].std()),
                val_yield_kg=float(df["harvested_kg"].mean()), val_failure=float(df["crop_failure"].mean()),
                val_invalid=float(df["invalid_actions"].mean()))
        if r > best:
            best = r
            save_checkpoint(run_dir, "best", pnet, dict(meta, step=steps, val_return=r))

    acc = dict(p=[], v=[], e=[], g=[])
    for batch in coll:
        steps += batch.numel()
        episode_stats(batch, st, steps, log)
        with torch.no_grad():
            gae(batch)
        out = loss(batch.reshape(-1))
        total_loss = sum(v for k, v in out.items() if k.startswith("loss_"))   # loss_entropy is absent when ent_coef == 0
        opt.zero_grad()
        total_loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, h["max_grad_norm"])
        opt.step()
        n_updates += 1
        acc["p"].append(float(out["loss_objective"])); acc["v"].append(float(out["loss_critic"]))
        acc["e"].append(_entropy(out, actor, batch.reshape(-1))); acc["g"].append(float(gn))
        if len(acc["p"]) >= 8 or steps >= total:      # log every 8 updates to keep the CSV small
            log.log("updates", step=steps, policy_loss=np.mean(acc["p"]), value_loss=np.mean(acc["v"]),
                    entropy=np.mean(acc["e"]), grad_norm=np.mean(acc["g"]), n_updates=n_updates)
            acc = dict(p=[], v=[], e=[], g=[])
        if steps >= next_eval:
            do_eval(); next_eval += eval_every
            log.flush()
    coll.shutdown()
    if not log.tables.get("eval") or log.tables["eval"][-1]["step"] < steps - eval_every // 2:
        do_eval()
    save_checkpoint(run_dir, "final", pnet, dict(meta, step=steps))
    return dict(best_val_return=best, steps=steps, updates=n_updates)
