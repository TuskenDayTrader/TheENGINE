"""
Tests for POST /analyze endpoint.

Coverage:
- Endpoint success with NQ-style payload
- Endpoint success with ES-style payload
- Validation failure: missing required field
- Validation failure: non-positive ATR
- Validation failure: high < low constraint
- Response contains all 2+2+2+2 buckets and action_state
- Poster text block is non-empty string containing ticker
"""

from __future__ import annotations

import json
import os

import pytest
from fastapi.testclient import TestClient

from apps.api.main import app

client = TestClient(app)

# ---------------------------------------------------------------------------
# Fixtures – minimal valid payloads
# ---------------------------------------------------------------------------

NQ_PAYLOAD = {
    "date_et": "2026-07-21",
    "ticker": "NQU2026",
    "timeframe": "30m",
    "lookback_days": 14,
    "current_price": 21850.0,
    "levels": {
        "pdh": 22100.0,
        "pdl": 21600.0,
        "prior_settle": 21700.0,
        "rth_open": 21750.0,
        "globex_high": 21900.0,
        "globex_low": 21580.0,
        "asia_high": 21880.0,
        "asia_low": 21640.0,
        "london_high": 21920.0,
        "london_low": 21700.0,
        "ny_high": 21875.0,
        "ny_low": 21780.0,
        "asia_ib_high": 21855.0,
        "asia_ib_low": 21720.0,
        "london_ib_high": 21910.0,
        "london_ib_low": 21740.0,
        "ny_ib_high": 21870.0,
        "ny_ib_low": 21800.0,
        "atr14": 220.0,
    },
}

ES_PAYLOAD = {
    "date_et": "2026-07-21",
    "ticker": "ESU2026",
    "timeframe": "30m",
    "lookback_days": 14,
    "current_price": 5540.0,
    "levels": {
        "pdh": 5580.0,
        "pdl": 5490.0,
        "prior_settle": 5510.0,
        "rth_open": 5520.0,
        "globex_high": 5565.0,
        "globex_low": 5475.0,
        "asia_high": 5558.0,
        "asia_low": 5495.0,
        "london_high": 5572.0,
        "london_low": 5510.0,
        "ny_high": 5548.0,
        "ny_low": 5522.0,
        "asia_ib_high": 5550.0,
        "asia_ib_low": 5512.0,
        "london_ib_high": 5560.0,
        "london_ib_low": 5518.0,
        "ny_ib_high": 5545.0,
        "ny_ib_low": 5525.0,
        "atr14": 45.0,
    },
}


# ---------------------------------------------------------------------------
# Success tests
# ---------------------------------------------------------------------------


def test_analyze_nq_success():
    """POST /analyze with a valid NQ payload returns HTTP 200."""
    resp = client.post("/analyze", json=NQ_PAYLOAD)
    assert resp.status_code == 200, resp.text


def test_analyze_es_success():
    """POST /analyze with a valid ES payload returns HTTP 200."""
    resp = client.post("/analyze", json=ES_PAYLOAD)
    assert resp.status_code == 200, resp.text


def test_response_contains_all_buckets_and_action_state():
    """Response must include all four 2+2+2+2 buckets and action_state."""
    resp = client.post("/analyze", json=NQ_PAYLOAD)
    assert resp.status_code == 200
    data = resp.json()

    for bucket in (
        "strongest_resistance",
        "weakest_resistance",
        "strongest_support",
        "weakest_support",
    ):
        assert bucket in data, f"Missing bucket: {bucket}"
        assert len(data[bucket]) == 2, f"Expected 2 items in {bucket}, got {len(data[bucket])}"

    assert "action_state" in data
    assert data["action_state"] in ("ACTIVE_LONG", "ACTIVE_SHORT", "STAND_DOWN")


def test_response_level_decision_fields():
    """Each level decision must have label, price, score, distance_atr, rationale."""
    resp = client.post("/analyze", json=NQ_PAYLOAD)
    assert resp.status_code == 200
    data = resp.json()

    for bucket in ("strongest_resistance", "weakest_resistance", "strongest_support", "weakest_support"):
        for entry in data[bucket]:
            for field in ("label", "price", "score", "distance_atr", "rationale"):
                assert field in entry, f"Field '{field}' missing in bucket '{bucket}'"


def test_response_poster_text_block():
    """Response must include a non-empty poster_text string containing the ticker."""
    resp = client.post("/analyze", json=NQ_PAYLOAD)
    assert resp.status_code == 200
    data = resp.json()

    assert "poster_text" in data
    assert isinstance(data["poster_text"], str)
    assert len(data["poster_text"]) > 0
    assert "NQU2026" in data["poster_text"]
    assert "SESSION CONFLUENCE MAP" in data["poster_text"]


def test_response_contains_confidence():
    """Confidence must be a float between 0 and 1."""
    resp = client.post("/analyze", json=NQ_PAYLOAD)
    assert resp.status_code == 200
    data = resp.json()

    assert "confidence" in data
    conf = data["confidence"]
    assert isinstance(conf, (int, float))
    assert 0.0 <= conf <= 1.0


def test_nq_sample_json_matches_schema():
    """The committed examples/nq_sample.json must produce a valid 200 response."""
    sample_path = os.path.join(os.path.dirname(__file__), "..", "examples", "nq_sample.json")
    with open(sample_path) as f:
        payload = json.load(f)
    resp = client.post("/analyze", json=payload)
    assert resp.status_code == 200, resp.text


def test_es_sample_json_matches_schema():
    """The committed examples/es_sample.json must produce a valid 200 response."""
    sample_path = os.path.join(os.path.dirname(__file__), "..", "examples", "es_sample.json")
    with open(sample_path) as f:
        payload = json.load(f)
    resp = client.post("/analyze", json=payload)
    assert resp.status_code == 200, resp.text


# ---------------------------------------------------------------------------
# Validation failure tests
# ---------------------------------------------------------------------------


def test_missing_required_field_returns_422():
    """Omitting 'ticker' must return HTTP 422 with a field-level error."""
    bad = {k: v for k, v in NQ_PAYLOAD.items() if k != "ticker"}
    resp = client.post("/analyze", json=bad)
    assert resp.status_code == 422
    detail = resp.json().get("detail", [])
    # Pydantic v2 returns a list of error dicts; at least one must mention 'ticker'
    error_str = str(detail)
    assert "ticker" in error_str.lower()


def test_missing_levels_field_returns_422():
    """Omitting the entire 'levels' block must return HTTP 422."""
    bad = {k: v for k, v in NQ_PAYLOAD.items() if k != "levels"}
    resp = client.post("/analyze", json=bad)
    assert resp.status_code == 422


def test_invalid_atr_zero_returns_422():
    """ATR of 0 is invalid; must return HTTP 422."""
    bad = {**NQ_PAYLOAD, "levels": {**NQ_PAYLOAD["levels"], "atr14": 0}}
    resp = client.post("/analyze", json=bad)
    assert resp.status_code == 422


def test_negative_current_price_returns_422():
    """Negative current_price must return HTTP 422."""
    bad = {**NQ_PAYLOAD, "current_price": -100.0}
    resp = client.post("/analyze", json=bad)
    assert resp.status_code == 422


def test_high_less_than_low_returns_422():
    """pdh < pdl must return HTTP 422 with a clear message."""
    bad_levels = {**NQ_PAYLOAD["levels"], "pdh": 21000.0, "pdl": 22000.0}
    bad = {**NQ_PAYLOAD, "levels": bad_levels}
    resp = client.post("/analyze", json=bad)
    assert resp.status_code == 422
    detail = str(resp.json().get("detail", ""))
    assert "pdh" in detail.lower() or "pdl" in detail.lower()


def test_invalid_date_format_returns_422():
    """date_et not matching YYYY-MM-DD must return HTTP 422."""
    bad = {**NQ_PAYLOAD, "date_et": "21-07-2026"}
    resp = client.post("/analyze", json=bad)
    assert resp.status_code == 422


def test_invalid_timeframe_returns_422():
    """An unsupported timeframe value must return HTTP 422."""
    bad = {**NQ_PAYLOAD, "timeframe": "2h"}
    resp = client.post("/analyze", json=bad)
    assert resp.status_code == 422


def test_lookback_days_zero_returns_422():
    """lookback_days of 0 is below the minimum of 1; must return HTTP 422."""
    bad = {**NQ_PAYLOAD, "lookback_days": 0}
    resp = client.post("/analyze", json=bad)
    assert resp.status_code == 422


def test_empty_body_returns_422():
    """Completely empty body must return HTTP 422."""
    resp = client.post("/analyze", json={})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


def test_health_check():
    """GET /health must return HTTP 200 with status=ok."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
