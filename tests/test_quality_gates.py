"""
Tests for extraction quality gates.

Verifies that instrument-aware sanity checks correctly reject impossible
scale values (e.g., YM support near 2.41), out-of-axis-bounds levels, and
insufficient-evidence extractions.
"""
from __future__ import annotations

import pytest

from packages.core.policy import (
    ExtractionQualityResult,
    PolicyConfig,
    SymbolTickModel,
    check_extraction_quality_gates,
)
import packages.core.policy as policy_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config_with_floors(**floors: float) -> PolicyConfig:
    """Build a minimal PolicyConfig with custom instrument_price_floor values."""
    return PolicyConfig(
        daily_profit_cap_usd=550.0,
        lockout_reset_timezone="America/New_York",
        lockout_reset_time="00:00",
        default_rr=1.0,
        min_confidence_for_action=0.70,
        allow_only_nearby_structure=True,
        proximity_model_type="min_of_atr_and_ticks",
        proximity_atr_multiple=0.25,
        proximity_max_ticks_by_symbol={"NQ": 40, "ES": 16, "YM": 100, "RTY": 20},
        template_value=350,
        template_unit="ticks",
        symbol_tick_model={
            "NQ": SymbolTickModel(tick_size=0.25, ticks_per_point=4.0, dollars_per_tick=5.0),
            "ES": SymbolTickModel(tick_size=0.25, ticks_per_point=4.0, dollars_per_tick=12.5),
            "YM": SymbolTickModel(tick_size=1.0, ticks_per_point=1.0, dollars_per_tick=5.0),
            "RTY": SymbolTickModel(tick_size=0.1, ticks_per_point=10.0, dollars_per_tick=5.0),
        },
        minimum_distinct_levels=1,
        instrument_price_floor=floors,
    )


# ---------------------------------------------------------------------------
# Gate 1: Instrument price floor
# ---------------------------------------------------------------------------


class TestInstrumentPriceFloor:
    """Impossible scale values must be rejected by the instrument floor gate."""

    def test_ym_absurd_low_support_rejected(self, monkeypatch):
        """YM support near 2.41 is physically impossible; must be rejected."""
        cfg = _make_config_with_floors(YM=10000.0)
        monkeypatch.setattr(policy_mod, "load_policy_config", lambda: cfg)

        result = check_extraction_quality_gates(
            ticker="YM",
            level_prices=[52821.82, 2.41],
        )
        assert not result.passed
        assert len(result.rejection_reasons) >= 1
        assert any("Impossible scale" in r for r in result.rejection_reasons)

    def test_ym_valid_levels_pass(self, monkeypatch):
        """Valid YM levels around 42000 must pass the floor gate."""
        cfg = _make_config_with_floors(YM=10000.0)
        monkeypatch.setattr(policy_mod, "load_policy_config", lambda: cfg)

        result = check_extraction_quality_gates(
            ticker="YM",
            level_prices=[42800.0, 42500.0, 42200.0],
        )
        assert result.passed
        assert result.rejection_reasons == []

    def test_nq_levels_below_floor_rejected(self, monkeypatch):
        """NQ levels below 10000 (e.g. OCR producing 999.0) must be rejected."""
        cfg = _make_config_with_floors(NQ=10000.0)
        monkeypatch.setattr(policy_mod, "load_policy_config", lambda: cfg)

        result = check_extraction_quality_gates(
            ticker="NQ",
            level_prices=[21000.0, 999.0],
        )
        assert not result.passed
        assert any("Impossible scale" in r for r in result.rejection_reasons)

    def test_nq_valid_range_passes(self, monkeypatch):
        """Normal NQ levels (~21000) pass the floor check."""
        cfg = _make_config_with_floors(NQ=10000.0)
        monkeypatch.setattr(policy_mod, "load_policy_config", lambda: cfg)

        result = check_extraction_quality_gates(
            ticker="NQ",
            level_prices=[21000.0, 20800.0, 20600.0],
        )
        assert result.passed

    def test_es_valid_range_passes(self, monkeypatch):
        """Normal ES levels (~6000) pass the floor check."""
        cfg = _make_config_with_floors(ES=1000.0)
        monkeypatch.setattr(policy_mod, "load_policy_config", lambda: cfg)

        result = check_extraction_quality_gates(
            ticker="ES",
            level_prices=[6050.0, 5980.0],
        )
        assert result.passed

    def test_es_single_digit_rejected(self, monkeypatch):
        """ES level near 5.0 is impossible; must fail the floor gate."""
        cfg = _make_config_with_floors(ES=1000.0)
        monkeypatch.setattr(policy_mod, "load_policy_config", lambda: cfg)

        result = check_extraction_quality_gates(
            ticker="ES",
            level_prices=[6050.0, 5.0],
        )
        assert not result.passed
        assert any("Impossible scale" in r for r in result.rejection_reasons)

    def test_rty_below_floor_rejected(self, monkeypatch):
        """RTY below 500 (e.g. 2.0) must be rejected."""
        cfg = _make_config_with_floors(RTY=500.0)
        monkeypatch.setattr(policy_mod, "load_policy_config", lambda: cfg)

        result = check_extraction_quality_gates(
            ticker="RTY",
            level_prices=[2100.0, 2.0],
        )
        assert not result.passed
        assert any("Impossible scale" in r for r in result.rejection_reasons)

    def test_unknown_ticker_no_floor_always_passes_gate1(self, monkeypatch):
        """Without a configured floor for the ticker, gate 1 is skipped."""
        cfg = _make_config_with_floors()  # empty floor map
        monkeypatch.setattr(policy_mod, "load_policy_config", lambda: cfg)

        result = check_extraction_quality_gates(
            ticker="UNKNOWN_SYM",
            level_prices=[1.0, 2.0],  # would fail if a floor existed
        )
        # Gate 1 skipped; only gate 3 could fail (2 distinct levels ≥ 1)
        assert result.passed


# ---------------------------------------------------------------------------
# Gate 2: Axis-bounds enforcement
# ---------------------------------------------------------------------------


class TestAxisBoundsEnforcement:
    """Levels outside the OCR axis range (± 10 % tolerance) must be rejected."""

    def test_level_within_axis_bounds_passes(self, monkeypatch):
        cfg = _make_config_with_floors()
        monkeypatch.setattr(policy_mod, "load_policy_config", lambda: cfg)

        result = check_extraction_quality_gates(
            ticker="NQ",
            level_prices=[20800.0],
            axis_min=20000.0,
            axis_max=21500.0,
        )
        assert result.passed

    def test_level_outside_axis_bounds_rejected(self, monkeypatch):
        """A level of 99999 is well outside [20000, 21500] ± 10 %."""
        cfg = _make_config_with_floors()
        monkeypatch.setattr(policy_mod, "load_policy_config", lambda: cfg)

        result = check_extraction_quality_gates(
            ticker="NQ",
            level_prices=[20800.0, 99999.0],
            axis_min=20000.0,
            axis_max=21500.0,
        )
        assert not result.passed
        assert any("Out-of-axis-bounds" in r for r in result.rejection_reasons)

    def test_level_within_tolerance_passes(self, monkeypatch):
        """A level 5 % above axis_max should still pass the 10 % tolerance."""
        cfg = _make_config_with_floors()
        monkeypatch.setattr(policy_mod, "load_policy_config", lambda: cfg)

        axis_max = 21500.0
        axis_min = 20000.0
        tolerance = (axis_max - axis_min) * 0.10  # 150
        just_inside = axis_max + tolerance - 1  # within tolerance

        result = check_extraction_quality_gates(
            ticker="NQ",
            level_prices=[just_inside],
            axis_min=axis_min,
            axis_max=axis_max,
        )
        assert result.passed

    def test_no_axis_info_skips_gate2(self, monkeypatch):
        """Without axis bounds, gate 2 is skipped (no false rejections)."""
        cfg = _make_config_with_floors()
        monkeypatch.setattr(policy_mod, "load_policy_config", lambda: cfg)

        result = check_extraction_quality_gates(
            ticker="NQ",
            level_prices=[99999.0],  # would fail if axis bounds were provided
        )
        assert result.passed


# ---------------------------------------------------------------------------
# Gate 3: Minimum distinct levels
# ---------------------------------------------------------------------------


class TestMinimumDistinctLevels:
    """Duplicate-collapse edge cases and low-evidence extractions must be caught."""

    def test_zero_levels_fails_gate3(self, monkeypatch):
        cfg = _make_config_with_floors()
        monkeypatch.setattr(policy_mod, "load_policy_config", lambda: cfg)

        result = check_extraction_quality_gates(
            ticker="NQ",
            level_prices=[],
        )
        assert not result.passed
        assert any("Insufficient visual evidence" in r for r in result.rejection_reasons)

    def test_one_distinct_level_passes_default(self, monkeypatch):
        """Default minimum_distinct_levels=1; a single valid level is enough."""
        cfg = _make_config_with_floors()
        monkeypatch.setattr(policy_mod, "load_policy_config", lambda: cfg)

        result = check_extraction_quality_gates(
            ticker="NQ",
            level_prices=[21000.0],
        )
        assert result.passed

    def test_duplicate_collapse_rejected_when_distinct_minimum_is_two(self, monkeypatch):
        """With minimum_distinct_levels=2, all-identical collapse is rejected."""
        cfg = PolicyConfig(
            daily_profit_cap_usd=550.0,
            lockout_reset_timezone="America/New_York",
            lockout_reset_time="00:00",
            default_rr=1.0,
            min_confidence_for_action=0.70,
            allow_only_nearby_structure=True,
            proximity_model_type="min_of_atr_and_ticks",
            proximity_atr_multiple=0.25,
            proximity_max_ticks_by_symbol={},
            template_value=350,
            template_unit="ticks",
            symbol_tick_model={},
            minimum_distinct_levels=2,
            instrument_price_floor={},
        )
        monkeypatch.setattr(policy_mod, "load_policy_config", lambda: cfg)

        result = check_extraction_quality_gates(
            ticker="NQ",
            level_prices=[21000.0, 21000.0],  # duplicate collapse
        )
        assert not result.passed
        assert any("Insufficient visual evidence" in r for r in result.rejection_reasons)


# ---------------------------------------------------------------------------
# ExtractionQualityResult contract
# ---------------------------------------------------------------------------


class TestExtractionQualityResultContract:
    """The returned dataclass always has the expected structure."""

    def test_passed_result_has_empty_reasons(self, monkeypatch):
        cfg = _make_config_with_floors()
        monkeypatch.setattr(policy_mod, "load_policy_config", lambda: cfg)

        result = check_extraction_quality_gates(ticker="NQ", level_prices=[21000.0])
        assert isinstance(result, ExtractionQualityResult)
        assert result.passed is True
        assert result.rejection_reasons == []

    def test_failed_result_has_non_empty_reasons(self, monkeypatch):
        cfg = _make_config_with_floors(YM=10000.0)
        monkeypatch.setattr(policy_mod, "load_policy_config", lambda: cfg)

        result = check_extraction_quality_gates(ticker="YM", level_prices=[2.41])
        assert result.passed is False
        assert len(result.rejection_reasons) >= 1
        assert all(isinstance(r, str) for r in result.rejection_reasons)


# ---------------------------------------------------------------------------
# End-to-end: /analyze-image quality gate wiring
# ---------------------------------------------------------------------------


class TestAnalyzeImageQualityGateWiring:
    """When quality gates fail, /analyze-image returns a warning but no analysis."""

    def test_ym_absurd_support_suppresses_analysis(self, monkeypatch):
        """
        When quality gates reject the extraction, the analysis field must be
        absent and the extraction_warning must explain the rejection.
        """
        from fastapi.testclient import TestClient
        from apps.api.main import app
        import packages.core.image_extractor as img_ext
        import apps.api.routes.analyze as analyze_route
        from packages.core.models import LevelsPayload

        mock_result = img_ext.ExtractionResult(
            image_decoded=True,
            levels_payload=LevelsPayload(
                pdh=52821.82,
                pdl=2.41,           # absurd YM support
                prior_settle=52000.0,
                atr14=200.0,
            ),
            current_price=52411.0,
            num_lines_detected=2,
            num_axis_points=5,
            extraction_confidence=0.75,
        )

        cfg = _make_config_with_floors(YM=10000.0)
        monkeypatch.setattr(policy_mod, "load_policy_config", lambda: cfg)
        monkeypatch.setattr(analyze_route, "extract_from_image", lambda *a, **kw: mock_result)

        client = TestClient(app)
        resp = client.post(
            "/analyze-image",
            files={"file": ("chart.png", b"\x89PNG\r\n\x1a\nfakepng", "image/png")},
            data={"ticker": "YM"},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert "analysis" not in data, "Analysis should be suppressed when QG fails"
        assert "extraction_warning" in data
        assert "Quality gate failed" in data["extraction_warning"]
        assert "Impossible scale" in data["extraction_warning"]

    def test_valid_ym_levels_allow_analysis(self, monkeypatch):
        """Valid YM levels allow analysis to proceed normally."""
        from fastapi.testclient import TestClient
        from apps.api.main import app
        import packages.core.image_extractor as img_ext
        import apps.api.routes.analyze as analyze_route
        from packages.core.models import LevelsPayload

        mock_result = img_ext.ExtractionResult(
            image_decoded=True,
            levels_payload=LevelsPayload(
                pdh=42900.0,
                pdl=42300.0,
                prior_settle=42600.0,
                atr14=200.0,
            ),
            current_price=42600.0,
            num_lines_detected=2,
            num_axis_points=5,
            extraction_confidence=0.75,
        )

        cfg = _make_config_with_floors(YM=10000.0)
        monkeypatch.setattr(policy_mod, "load_policy_config", lambda: cfg)
        monkeypatch.setattr(analyze_route, "extract_from_image", lambda *a, **kw: mock_result)

        client = TestClient(app)
        resp = client.post(
            "/analyze-image",
            files={"file": ("chart.png", b"\x89PNG\r\n\x1a\nfakepng", "image/png")},
            data={"ticker": "YM"},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data.get("analysis") is not None, "Valid levels should produce an analysis"
