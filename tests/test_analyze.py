from __future__ import annotations

import json
import pathlib

from fastapi.testclient import TestClient

from apps.api.main import app

client = TestClient(app)
FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures"


def _load_fixture(name: str) -> dict:
    with open(FIXTURES_DIR / name) as f:
        data = json.load(f)
    data.pop("_comment", None)
    data.pop("_expected", None)
    return data


def _assert_contract(data: dict) -> None:
    for bucket in (
        "strongest_resistance",
        "weakest_resistance",
        "strongest_support",
        "weakest_support",
    ):
        assert bucket in data
        assert len(data[bucket]) == 2
        for entry in data[bucket]:
            for field in ("label", "price", "score", "distance_atr", "rationale"):
                assert field in entry


def test_analyze_nq_fixture_success():
    payload = _load_fixture("nq_sample.json")
    resp = client.post("/analyze", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    _assert_contract(data)
    assert data["action_state"] in ("ACTIVE_LONG", "ACTIVE_SHORT", "STAND_DOWN")
    assert 0.0 <= data["confidence"] <= 1.0
    assert payload["ticker"] in data["poster_text"]
    assert "SESSION CONFLUENCE MAP" in data["poster_text"]


def test_analyze_es_fixture_success():
    payload = _load_fixture("es_sample.json")
    resp = client.post("/analyze", json=payload)
    assert resp.status_code == 200, resp.text
    _assert_contract(resp.json())


def test_missing_required_field_returns_422():
    payload = _load_fixture("nq_sample.json")
    payload.pop("ticker")
    resp = client.post("/analyze", json=payload)
    assert resp.status_code == 422


def test_invalid_atr_zero_returns_422():
    payload = _load_fixture("nq_sample.json")
    payload["levels"]["atr14"] = 0
    resp = client.post("/analyze", json=payload)
    assert resp.status_code == 422


def test_high_less_than_low_returns_422():
    payload = _load_fixture("nq_sample.json")
    payload["levels"]["pdh"] = payload["levels"]["pdl"] - 1
    resp = client.post("/analyze", json=payload)
    assert resp.status_code == 422


def test_invalid_date_format_returns_422():
    payload = _load_fixture("nq_sample.json")
    payload["date_et"] = "21-07-2026"
    resp = client.post("/analyze", json=payload)
    assert resp.status_code == 422


def test_invalid_timeframe_returns_422():
    payload = _load_fixture("nq_sample.json")
    payload["timeframe"] = "2h"
    resp = client.post("/analyze", json=payload)
    assert resp.status_code == 422


def test_health_check():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
