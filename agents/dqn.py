"""DQN with TorchRL: QValueActor, epsilon-greedy exploration, replay buffer, DQNLoss, target network."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from tensordict.nn import TensorDictSequential
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.modules import EGreedyModule, QValueActor
from torchrl.objectives import DQNLoss, HardUpdate, SoftUpdate

from environment import AgriResilienceEnv, N_ACTIONS
from evaluation.evaluator import evaluate_policy
from training.splits import VAL_SEEDS

from .common import RunLogger, make_env, make_net, seed_everything
from .loader import CheckpointPolicy, save_checkpoint

DEFAULTS = dict(
    lr=5e-4, gamma=0.99, batch_size=64, buffer_size=50_000, hidden=128, depth=2, activation="relu",
    eps_end=0.05, eps_decay_frac=0.4,           # epsilon falls linearly over this fraction of the budget
    double_dqn=True, target="soft", tau=0.01, hard_every=500,   # target network: soft (Polyak) or hard copy
    utd=0.25,                                    # gradient updates per environment step
    learn_start=1000, frames_per_batch=32, max_grad_norm=10.0, eval_every_frac=0.1,
)


def train(cfg: dict, run_dir: Path, log: RunLogger) -> dict:
    h = {**DEFAULTS, **cfg}
    seed_everything(h["seed"])
    total = int(h["total_steps"])
    env = make_env(seed=h["seed"])
    net = make_net(N_ACTIONS, h["hidden"], h["depth"], h["activation"], last_gain=1.0)
    qactor = QValueActor(net, in_keys=["observation"], spec=env.action_spec, action_space="categorical")
    egreedy = EGreedyModule(spec=env.action_spec, eps_init=1.0, eps_end=h["eps_end"],
                            annealing_num_steps=max(1, int(h["eps_decay_frac"] * total)))
    explore = TensorDictSequential(qactor, egreedy)

    loss = DQNLoss(qactor, action_space="categorical", loss_function="smooth_l1", double_dqn=h["double_dqn"],
                   delay_value=True)
    loss.make_value_estimator(gamma=h["gamma"])
    target_upd = SoftUpdate(loss, eps=1.0 - h["tau"]) if h["target"] == "soft" else HardUpdate(
        loss, value_network_update_interval=h["hard_every"])
    opt = torch.optim.Adam(qactor.parameters(), lr=h["lr"])
    rb = ReplayBuffer(storage=LazyTensorStorage(h["buffer_size"]))
    coll = Collector(env, explore, frames_per_batch=h["frames_per_batch"], total_frames=total, init_random_frames=0)

    eval_env = AgriResilienceEnv()
    eval_every = max(1, int(h["eval_every_frac"] * total))
    next_eval, best, steps, updates = eval_every, -1e9, 0, 0
    ep_ret, ep_len = 0.0, 0
    meta = dict(algo="dqn", hidden=h["hidden"], depth=h["depth"], activation=h["activation"], config=h)

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
        return r

    for batch in coll:
        n = batch.numel()
        steps += n
        rb.extend(batch.reshape(-1))
        r = batch["next", "reward"].reshape(-1).numpy()
        d = batch["next", "done"].reshape(-1).numpy()
        for ri, di in zip(r, d):
            ep_ret += float(ri); ep_len += 1
            if di:
                log.log("episodes", step=steps, ret=ep_ret, length=ep_len)
                ep_ret, ep_len = 0.0, 0
        egreedy.step(n)
        if steps >= h["learn_start"] and len(rb) >= h["batch_size"]:
            tl, qm = [], []
            for _ in range(max(1, int(round(h["utd"] * n)))):
                td = rb.sample(h["batch_size"])
                out = loss(td)
                opt.zero_grad()
                out["loss"].backward()
                gn = torch.nn.utils.clip_grad_norm_(qactor.parameters(), h["max_grad_norm"])
                opt.step()
                target_upd.step()
                updates += 1
                tl.append(float(out["loss"])); qm.append(float(td.get("chosen_action_value", torch.zeros(1)).mean()) if "chosen_action_value" in td.keys() else 0.0)
            log.log("updates", step=steps, td_loss=float(np.mean(tl)), grad_norm=float(gn),
                    epsilon=float(egreedy.eps), n_updates=updates)
        if steps >= next_eval:
            do_eval(); next_eval += eval_every
            log.flush()
    coll.shutdown()
    if not log.tables.get("eval") or log.tables["eval"][-1]["step"] < steps - eval_every // 2:
        do_eval()
    save_checkpoint(run_dir, "final", net, dict(meta, step=steps))
    return dict(best_val_return=best, steps=steps, updates=updates)
