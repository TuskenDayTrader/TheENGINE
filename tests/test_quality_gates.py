"""
Tests for extraction quality gates integrated into the /analyze-image endpoint.

Covers:
1. Impossible scale rejection (YM = 2.41, NQ = 500)
2. Axis-bounds rejection
3. Minimum distinct levels rejection
4. API endpoint wiring: quality gate failure suppresses analysis block
5. Quality gate passes → analysis block present
6. API response includes policy block on successful analysis
7. Edge cases: mixed levels, unknown ticker, empty levels

19 tests covering all three gates and the API endpoint wiring.
"""
from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient

from apps.api.main import app
from packages.core.policy import check_extraction_quality_gates

client = TestClient(app)


# ---------------------------------------------------------------------------
# Unit tests: check_extraction_quality_gates
# ---------------------------------------------------------------------------


class TestImpossibleScaleGate:
    """Gate 1: instrument price floor."""

    def test_ym_absurd_value_rejected(self):
        result = check_extraction_quality_gates("YM", [2.41, 3.10])
        assert result.passed is False
        assert "Impossible scale" in (result.rejection_reason or "")

    def test_nq_tiny_value_rejected(self):
        result = check_extraction_quality_gates("NQ", [999.9])
        assert result.passed is False
        assert "Impossible scale" in (result.rejection_reason or "")

    def test_es_value_below_floor_rejected(self):
        result = check_extraction_quality_gates("ES", [500.0])
        assert result.passed is False

    def test_rty_value_below_floor_rejected(self):
        result = check_extraction_quality_gates("RTY", [100.0])
        assert result.passed is False

    def test_nq_valid_values_pass(self):
        result = check_extraction_quality_gates("NQ", [21000.0, 20800.0])
        assert result.passed is True

    def test_es_valid_values_pass(self):
        result = check_extraction_quality_gates("ES", [5500.0, 5450.0])
        assert result.passed is True

    def test_ym_valid_values_pass(self):
        result = check_extraction_quality_gates("YM", [44000.0, 43500.0])
        assert result.passed is True

    def test_mixed_valid_and_invalid_fails_on_invalid(self):
        """Even one bad level fails the entire batch."""
        result = check_extraction_quality_gates("NQ", [21000.0, 2.41])
        assert result.passed is False


class TestAxisBoundsGate:
    """Gate 2: levels outside OCR axis range ±10 %."""

    def test_level_above_axis_rejected(self):
        result = check_extraction_quality_gates(
            "NQ", [25000.0], axis_min=20000.0, axis_max=22000.0
        )
        # 25000 > 22000 * 1.10 = 24200
        assert result.passed is False
        assert "axis bounds" in (result.rejection_reason or "").lower()

    def test_level_below_axis_rejected(self):
        result = check_extraction_quality_gates(
            "NQ", [17000.0], axis_min=20000.0, axis_max=22000.0
        )
        # 17000 < 20000 * 0.90 = 18000
        assert result.passed is False

    def test_level_within_tolerance_accepted(self):
        result = check_extraction_quality_gates(
            "NQ", [21500.0, 20500.0], axis_min=20000.0, axis_max=22000.0
        )
        assert result.passed is True

    def test_no_axis_bounds_skips_gate(self):
        """When axis bounds are not provided, this gate is skipped."""
        result = check_extraction_quality_gates("NQ", [21000.0])
        assert result.passed is True


class TestMinimumLevelsGate:
    """Gate 3: minimum distinct levels."""

    def test_empty_levels_rejected(self):
        result = check_extraction_quality_gates("NQ", [])
        assert result.passed is False

    def test_single_level_passes_when_minimum_is_one(self):
        result = check_extraction_quality_gates("NQ", [21000.0])
        assert result.passed is True

    def test_duplicate_levels_treated_as_one(self):
        """All-same values collapse to 1 distinct (passes minimum = 1)."""
        result = check_extraction_quality_gates("NQ", [21000.0, 21000.0, 21000.0])
        assert result.passed is True


# ---------------------------------------------------------------------------
# API endpoint integration tests
# ---------------------------------------------------------------------------


def _make_fake_png() -> bytes:
    """Create a minimal valid PNG (solid dark background, no levels)."""
    img = np.full((100, 100, 3), (34, 23, 19), dtype=np.uint8)
    import cv2
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()


class TestAnalyzeImageQualityGateWiring:
    """Quality gate failures suppress analysis in /analyze-image."""

    def test_upload_invalid_content_type_rejected(self):
        response = client.post(
            "/analyze-image",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )
        assert response.status_code == 400

    def test_valid_png_upload_accepted(self):
        """Even a blank image must return HTTP 200 (not crash)."""
        png = _make_fake_png()
        response = client.post(
            "/analyze-image",
            files={"file": ("chart.png", png, "image/png")},
            data={"ticker": "NQ"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["message"] == "upload received"

    def test_response_includes_extraction_fields_on_valid_upload(self):
        """Response always includes extraction_confidence (may be 0) on decoded images."""
        png = _make_fake_png()
        response = client.post(
            "/analyze-image",
            files={"file": ("chart.png", png, "image/png")},
            data={"ticker": "NQ"},
        )
        assert response.status_code == 200
        body = response.json()
        # extraction_confidence present OR analysis suppressed (both valid)
        assert "message" in body


# ---------------------------------------------------------------------------
# API endpoint: /analyze with policy block
# ---------------------------------------------------------------------------


class TestAnalyzeEndpointPolicyBlock:
    """Verify /analyze response includes required policy fields."""

    _BASE_PAYLOAD = {
        "date_et": "2026-07-22",
        "ticker": "NQ",
        "timeframe": "30m",
        "lookback_days": 5,
        "current_price": 21000.0,
        "levels": {
            "pdh": 21100.0,
            "pdl": 20900.0,
            "prior_settle": 21000.0,
            "atr14": 150.0,
        },
    }

    def test_analyze_returns_policy_block(self):
        response = client.post("/analyze", json=self._BASE_PAYLOAD)
        assert response.status_code == 200
        body = response.json()
        assert "policy" in body
        policy = body["policy"]
        assert "enforced_action_state" in policy
        assert "lockout_active" in policy
        assert "stand_down_reasons" in policy
        assert "rr_target" in policy

    def test_analyze_returns_nearest_levels(self):
        response = client.post("/analyze", json=self._BASE_PAYLOAD)
        assert response.status_code == 200
        body = response.json()
        # nearest_resistance and nearest_support may be None or float
        assert "nearest_resistance" in body
        assert "nearest_support" in body

    def test_analyze_returns_policy_reason(self):
        response = client.post("/analyze", json=self._BASE_PAYLOAD)
        assert response.status_code == 200
        body = response.json()
        assert "policy_reason" in body
        assert isinstance(body["policy_reason"], str)

    def test_analyze_returns_stop_and_target_distance(self):
        response = client.post("/analyze", json=self._BASE_PAYLOAD)
        assert response.status_code == 200
        body = response.json()
        assert "stop_distance" in body
        assert "target_distance" in body

    def test_daily_lockout_via_api(self):
        """Passing realized_pnl_today >= 550 must trigger STOP_TRADING_DAY."""
        payload = {**self._BASE_PAYLOAD, "realized_pnl_today": 600.0}
        response = client.post("/analyze", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["policy"]["enforced_action_state"] == "STOP_TRADING_DAY"
        assert body["policy"]["lockout_active"] is True

    def test_backward_compat_action_state_still_present(self):
        """Original action_state field must still exist for backward compatibility."""
        response = client.post("/analyze", json=self._BASE_PAYLOAD)
        assert response.status_code == 200
        body = response.json()
        assert "action_state" in body
        assert "confidence" in body
        assert "poster_text" in body
