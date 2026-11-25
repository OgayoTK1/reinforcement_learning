"""Headless playback of a trained checkpoint or a baseline, with no browser.

    uv run play.py --policy models/pg/ppo.pt --seed 1000 --scenario severe_drought
    uv run play.py --policy rule_based --seed 1001 --quiet
"""
from __future__ import annotations

import argparse

from backend.session import Session, list_policies
from environment import ACTION_NAMES
from environment.rendering import render_ansi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="rule_based", help="rule_based, random, or a path to a .pt checkpoint")
    ap.add_argument("--scenario", default="training")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--quiet", action="store_true", help="print only the season summary")
    ap.add_argument("--list", action="store_true", help="list available policies and exit")
    a = ap.parse_args()
    if a.list:
        for p in list_policies():
            print(p["id"], "-", p["label"])
        return
    s = Session(a.policy, a.scenario, a.seed)
    while not s.done:
        snap = s.step()
        if not a.quiet:
            act = snap["action"]
            print(render_ansi(s.env), "|", act["requested"] + ("" if act["valid"] else " (impossible)"))
    m = s.env.episode_summary()
    print(f"\n{a.policy}  seed {a.seed}  scenario {s.env.rnd.scenario.name}")
    for k in ("reason", "return", "harvested_kg", "marketable_bags", "net_return_kes", "water_used_m3", "soil_change",
              "post_harvest_loss_frac", "food_reserve_met", "invalid_actions"):
        v = m[k]
        print(f"  {k:24s} {v:.3f}" if isinstance(v, float) else f"  {k:24s} {v}")


if __name__ == "__main__":
    main()
