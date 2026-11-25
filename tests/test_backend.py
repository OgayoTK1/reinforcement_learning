"""Backend tests: REST catalogue, whole-episode endpoint and the WebSocket protocol. The frontend never
simulates anything, so these tests check that what it receives is exactly what the Gymnasium env produced."""
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app import app
from backend.session import Session
from environment import AgriResilienceEnv
from environment.rendering import serialize_state
from evaluation.baselines import ExtensionRulePolicy

client = TestClient(app)


def test_health_and_catalogues():
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and len(h["spec_hash"]) == 12
    cat = client.get("/api/catalogue").json()
    assert len(cat["actions"]) == 20 and "severe_drought" in cat["scenarios"]
    ids = [p["id"] for p in client.get("/api/policies").json()]
    assert {"manual", "rule_based", "random"} <= set(ids)
    assert client.get("/api/observation_spec").json()["names"][0]


def test_http_episode_matches_a_direct_environment_rollout():
    body = client.post("/api/episode", json={"policy": "rule_based", "seed": 1003}).json()
    env, pol = AgriResilienceEnv(), ExtensionRulePolicy()
    obs, _ = env.reset(seed=1003, options={"episode_seed": 1003})
    done = False
    while not done:
        obs, _, done, _, _ = env.step(pol.act(obs))
    assert body["summary"]["return"] == pytest.approx(env.episode_summary()["return"])
    assert body["states"][-1]["resources"]["cash"] == pytest.approx(serialize_state(env)["resources"]["cash"])


def test_unknown_policy_and_path_traversal_are_rejected():
    assert client.post("/api/episode", json={"policy": "../../etc/passwd"}).status_code == 400
    assert client.post("/api/episode", json={"policy": "rule_based", "scenario": "nope"}).status_code == 400


def test_websocket_streams_a_full_episode():
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "hello"
        ws.send_json({"type": "start", "policy": "rule_based", "seed": 1001, "speed": 100})
        weeks, end = [], None
        while end is None:
            m = ws.receive_json()
            if m["type"] == "state":
                weeks.append(m["data"]["time"]["week"])
            elif m["type"] == "episode_end":
                end = m
        assert weeks[0] == 0 and weeks[-1] == 52 and weeks == sorted(weeks)
        assert end["reason"] == "end_of_year" and "return" in end["summary"]


def test_websocket_manual_stepping_and_invalid_input():
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "step"})
        assert ws.receive_json()["type"] == "error"
        ws.send_json({"type": "start", "policy": "manual", "seed": 5})
        first = ws.receive_json()["data"]
        assert first["valid_actions"][1] is True or first["valid_actions"][1] is False
        ws.send_json({"type": "step", "action": 9})        # scout
        m = ws.receive_json()
        assert m["data"]["action"]["requested"] == "scout_field" and m["data"]["time"]["week"] == 1
        ws.send_json({"type": "bogus"})
        assert ws.receive_json()["type"] == "error"
        ws.send_json({"type": "start", "policy": "rule_based", "scenario": "not_a_scenario"})
        assert ws.receive_json()["type"] == "error"


def test_serialised_schema_has_no_nans_and_is_json():
    s = Session("rule_based", "wet_year", 1000)
    for _ in range(20):
        snap = s.step()
    text = json.dumps(snap, allow_nan=False)
    assert "plots" in snap and len(snap["plots"]) == 4 and len(text) > 1000


def test_static_frontend_is_served():
    r = client.get("/")
    assert r.status_code == 200 and "AgriResilience" in r.text
    assert client.get("/vendor/three.module.min.js").status_code == 200
    assert client.get("/src/main.js").status_code == 200
