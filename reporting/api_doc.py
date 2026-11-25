"""Generate docs/API.md from the live code so the schema cannot drift from serialize_state()."""
from __future__ import annotations

from pathlib import Path

from backend.session import Session, catalogue
from environment import SCENARIOS
from environment.rendering import SCHEMA_VERSION


def _flatten(d, prefix=""):
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            yield from _flatten(v, key + ".")
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            yield from _flatten(v[0], key + "[i].")
        else:
            yield key, v


def build(out: Path = Path("docs/API.md")) -> Path:
    s = Session("rule_based", "training", 1000)
    for _ in range(14):
        snap = s.step()
    rows = []
    for k, v in _flatten(snap):
        t = "boolean" if isinstance(v, bool) else "integer" if isinstance(v, int) else "number" if isinstance(v, float) else "array" if isinstance(v, list) else "string" if isinstance(v, str) else "null" if v is None else type(v).__name__
        ex = str(round(v, 3) if isinstance(v, float) else (v if not isinstance(v, list) else f"[{len(v)} items]"))
        rows.append(f"| `{k}` | {t} | {ex[:40]} |")
    cat = catalogue()
    text = f"""# API and WebSocket specification

Everything the viewer shows is produced by `environment.rendering.serialize_state(env)`, schema version
**{SCHEMA_VERSION}**. The browser never computes agronomy. It only maps these numbers to colours, heights and water levels.

## REST

| Method and path | Purpose |
|---|---|
| `GET /api/health` | `status`, `env_version`, `schema_version`, `spec_hash` of the observation spec |
| `GET /api/catalogue` | the 20 actions (id, name, description) and the scenario names |
| `GET /api/policies` | `manual`, `rule_based`, `random`, and every checkpoint found under `models/` |
| `GET /api/observation_spec` | the 65 observation names and the spec hash |
| `POST /api/episode` | body `{{"policy", "scenario", "seed"}}`; runs a whole episode server side and returns `{{"states": [...], "summary": {{...}}}}` |

Errors: `400` with a text reason for an unknown scenario, an unknown policy, or a checkpoint path outside the project.

## WebSocket `/ws`

Text frames carrying JSON, one connection per viewer. The server owns the environment; the client only sends intent.

Client to server:

| `type` | fields | effect |
|---|---|---|
| `start` | `policy`, `scenario`, `seed`, `speed`, `autoplay` | create a session, reset the environment with `episode_seed = seed`, send the first state, then step on a timer if the policy is not manual |
| `step` | `action` (integer, manual mode only) | apply one action and send the new state |
| `pause`, `resume` | none | stop or continue automatic stepping |
| `reset` | none | restart the same seed and scenario |
| `speed` | `value` | playback speed multiplier (about 0.9 s per week at 1.0) |

Server to client:

| `type` | fields |
|---|---|
| `hello` | `env_version`, `schema_version`, `spec_hash` |
| `state` | `data`: the state object below |
| `episode_end` | `summary` (episode metrics), `reason` (`end_of_year`, `never_sown`, `crop_failure`, `bankrupt`) |
| `error` | `message` |

Policies and seeds are reproducible: `seed` becomes the environment's `episode_seed`, so a replay shows exactly the season that
`evaluation/` scored for that seed.

## State object (`data`)

Valid scenarios: {", ".join(f"`{n}`" for n in ["training"] + list(SCENARIOS))}.

| Field | Type | Example |
|---|---|---|
{chr(10).join(rows)}

`valid_actions` is a list of 20 booleans (which actions are possible now); `done` marks the final frame. Per-plot values use the
index `i = 0..3`. `plots[i].soil_moisture` is available water as a fraction of capacity, `canopy_retained` the fraction of leaf
area not lost to stress, pests and disease, and `stress` the accumulated water and nitrogen stress.

## Actions

| id | name | description |
|---|---|---|
{chr(10).join(f"| {a['id']} | `{a['name']}` | {a['description']} |" for a in cat['actions'])}
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return out


if __name__ == "__main__":
    print(build())
