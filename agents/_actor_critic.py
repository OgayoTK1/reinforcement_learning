"""Shared construction for PPO and A2C: categorical actor, value critic, collector and GAE."""
from __future__ import annotations

import torch
from tensordict.nn import InteractionType, TensorDictModule
from torch.distributions import Categorical
from torchrl.collectors import Collector
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.objectives.value import GAE

from environment import N_ACTIONS

from .common import make_env, make_net, make_value_net


def build(h: dict):
    env = make_env(seed=h["seed"])
    pnet = make_net(N_ACTIONS, h["hidden"], h["depth"], h["activation"], last_gain=0.01)
    vnet = make_value_net(h["hidden"], h["depth"], h["activation"])
    actor = ProbabilisticActor(TensorDictModule(pnet, in_keys=["observation"], out_keys=["logits"]),
                               in_keys=["logits"], spec=env.action_spec, distribution_class=Categorical,
                               default_interaction_type=InteractionType.RANDOM, return_log_prob=True)
    critic = ValueOperator(vnet, in_keys=["observation"])
    coll = Collector(env, actor, frames_per_batch=h["frames_per_batch"], total_frames=int(h["total_steps"]),
                     init_random_frames=0)
    gae = GAE(gamma=h["gamma"], lmbda=h["lmbda"], value_network=critic, average_gae=False)
    return env, pnet, vnet, actor, critic, coll, gae


def episode_stats(batch, state: dict, steps: int, log) -> None:
    r = batch["next", "reward"].reshape(-1).numpy()
    d = batch["next", "done"].reshape(-1).numpy()
    for ri, di in zip(r, d):
        state["ret"] += float(ri); state["len"] += 1
        if di:
            log.log("episodes", step=steps, ret=state["ret"], length=state["len"])
            state["ret"], state["len"] = 0.0, 0
