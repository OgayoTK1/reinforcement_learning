"""A playback session: one AgriResilienceEnv plus one policy. The environment is the only source of truth;
the session only chooses actions and forwards serialize_state() to the client."""
from __future__ import annotations

from pathlib import Path

from agents.loader import load_policy
from environment import ACTION_DESCRIPTIONS, ACTION_NAMES, SCENARIOS, AgriResilienceEnv
from environment.rendering import serialize_state
from evaluation.baselines import ExtensionRulePolicy, RandomPolicy

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIRS = [ROOT / "models"]


def list_policies() -> list[dict]:
    out = [{"id": "manual", "label": "Manual (you choose each action)", "kind": "manual"},
           {"id": "rule_based", "label": "Rule-based extension calendar", "kind": "baseline"},
           {"id": "random", "label": "Random actions", "kind": "baseline"}]
    for d in MODEL_DIRS:
        for p in sorted(d.rglob("*.pt")):
            if p.with_suffix(".json").exists():
                rel = p.relative_to(ROOT).as_posix()
                import json
                meta = json.loads(p.with_suffix(".json").read_text())
                out.append({"id": rel, "label": f"{meta.get('algo', '?').upper()}  {rel}", "kind": "checkpoint",
                            "algo": meta.get("algo"), "val_return": meta.get("val_return")})
    return out


def make_policy(pid: str, seed: int = 0):
    if pid == "rule_based":
        return ExtensionRulePolicy()
    if pid == "random":
        return RandomPolicy(seed)
    if pid == "manual":
        return None
    path = (ROOT / pid).resolve()
    if ROOT not in path.parents or path.suffix != ".pt":
        raise ValueError("checkpoint must be a .pt file inside the project")
    return load_policy(path)


class Session:
    def __init__(self, policy_id: str, scenario: str | None, seed: int):
        if scenario not in (None, "training") and scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario {scenario}")
        self.env = AgriResilienceEnv()
        self.policy_id = policy_id
        self.policy = make_policy(policy_id, seed)
        self.scenario = None if scenario in (None, "training") else scenario
        self.seed = int(seed)
        self.obs = None
        self.done = False
        self.reset()

    def reset(self) -> dict:
        self.obs, _ = self.env.reset(seed=self.seed, options={"episode_seed": self.seed, "scenario": self.scenario})
        if self.policy is not None:
            self.policy.reset()
        self.done = False
        return self.snapshot()

    def snapshot(self) -> dict:
        d = serialize_state(self.env)
        d["valid_actions"] = [bool(v) for v in self.env.action_masks()]
        d["done"] = self.done
        return d

    def step(self, action: int | None = None) -> dict:
        if self.done:
            return self.snapshot()
        if action is None:
            if self.policy is None:
                raise ValueError("manual session needs an action")
            action = self.policy.act(self.obs)
        self.obs, _, term, trunc, info = self.env.step(int(action))
        self.done = bool(term or trunc)
        snap = self.snapshot()
        snap["termination_reason"] = info.get("episode", {}).get("reason") if self.done else None
        return snap


def catalogue() -> dict:
    return {"actions": [{"id": i, "name": n, "description": ACTION_DESCRIPTIONS[i]} for i, n in enumerate(ACTION_NAMES)],
            "scenarios": ["training"] + list(SCENARIOS)}
