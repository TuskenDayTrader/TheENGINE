"""
Tests for packages/core/policy.py – scalper policy engine.

Covers:
1. Daily PnL lockout at +$550
2. Low-confidence STAND_DOWN (confidence < 0.70)
3. Too-far-from-structure STAND_DOWN
4. Valid setup produces ACTIVE_LONG_1TO1 / ACTIVE_SHORT_1TO1 with 1:1 template
5. Template350 values for each supported symbol
6. QualityGateResult: impossible price floor, axis-bounds, minimum-levels
"""
from __future__ import annotations

import pytest

from packages.core.policy import (
    PolicyDecision,
    QualityGateResult,
    check_extraction_quality_gates,
    enforce_scalper_policy,
)


# ---------------------------------------------------------------------------
# enforce_scalper_policy – gate 1: daily PnL lockout
# ---------------------------------------------------------------------------


class TestDailyLockout:
    def test_lockout_at_cap(self):
        decision = enforce_scalper_policy(
            ticker="NQ",
            action_state="ACTIVE_LONG",
            confidence=0.90,
            nearest_resistance=21100.0,
            current_price=21000.0,
            atr14=100.0,
            realized_pnl_today=550.0,
        )
        assert decision.enforced_action_state == "STOP_TRADING_DAY"
        assert decision.lockout_active is True
        assert len(decision.stand_down_reasons) == 1
        assert "550" in decision.stand_down_reasons[0]

    def test_lockout_above_cap(self):
        decision = enforce_scalper_policy(
            ticker="NQ",
            action_state="ACTIVE_SHORT",
            confidence=0.95,
            current_price=21000.0,
            atr14=100.0,
            realized_pnl_today=750.0,
        )
        assert decision.enforced_action_state == "STOP_TRADING_DAY"
        assert decision.lockout_active is True

    def test_no_lockout_below_cap(self):
        decision = enforce_scalper_policy(
            ticker="NQ",
            action_state="ACTIVE_LONG",
            confidence=0.90,
            nearest_resistance=21050.0,
            current_price=21000.0,
            atr14=200.0,
            realized_pnl_today=549.99,
        )
        assert decision.enforced_action_state != "STOP_TRADING_DAY"
        assert decision.lockout_active is False

    def test_lockout_zero_pnl_no_trigger(self):
        decision = enforce_scalper_policy(
            ticker="ES",
            action_state="STAND_DOWN",
            confidence=0.80,
            realized_pnl_today=0.0,
        )
        assert decision.enforced_action_state != "STOP_TRADING_DAY"
        assert decision.lockout_active is False


# ---------------------------------------------------------------------------
# enforce_scalper_policy – gate 2: low confidence STAND_DOWN
# ---------------------------------------------------------------------------


class TestConfidenceGate:
    def test_stand_down_below_min_confidence(self):
        decision = enforce_scalper_policy(
            ticker="NQ",
            action_state="ACTIVE_LONG",
            confidence=0.60,
            nearest_resistance=21100.0,
            current_price=21000.0,
            atr14=100.0,
        )
        assert decision.enforced_action_state == "STAND_DOWN"
        assert any("0.60" in r for r in decision.stand_down_reasons)

    def test_stand_down_at_zero_confidence(self):
        decision = enforce_scalper_policy(
            ticker="NQ",
            action_state="ACTIVE_LONG",
            confidence=0.0,
            current_price=21000.0,
        )
        assert decision.enforced_action_state == "STAND_DOWN"

    def test_no_stand_down_at_min_confidence(self):
        decision = enforce_scalper_policy(
            ticker="NQ",
            action_state="ACTIVE_LONG",
            confidence=0.70,
            nearest_resistance=21050.0,
            current_price=21000.0,
            atr14=200.0,
        )
        # At exactly 0.70 (= minimum), should NOT stand down on confidence
        assert "0.70" not in " ".join(decision.stand_down_reasons)

    def test_no_stand_down_high_confidence(self):
        decision = enforce_scalper_policy(
            ticker="NQ",
            action_state="ACTIVE_LONG",
            confidence=0.90,
            nearest_resistance=21050.0,
            current_price=21000.0,
            atr14=200.0,
        )
        assert decision.enforced_action_state != "STAND_DOWN"


# ---------------------------------------------------------------------------
# enforce_scalper_policy – gate 3: too far from structure
# ---------------------------------------------------------------------------


class TestProximityGate:
    def test_stand_down_too_far_from_resistance(self):
        """ACTIVE_LONG fails when resistance is far away."""
        decision = enforce_scalper_policy(
            ticker="NQ",
            action_state="ACTIVE_LONG",
            confidence=0.90,
            nearest_resistance=21500.0,  # 500 pts away
            current_price=21000.0,
            atr14=50.0,                  # limit = 50 pts (ATR * 1.0)
        )
        assert decision.enforced_action_state == "STAND_DOWN"
        assert any("proximity" in r.lower() or "exceeds" in r.lower()
                   for r in decision.stand_down_reasons)

    def test_stand_down_no_resistance_for_long(self):
        """ACTIVE_LONG fails when nearest_resistance is None."""
        decision = enforce_scalper_policy(
            ticker="NQ",
            action_state="ACTIVE_LONG",
            confidence=0.90,
            nearest_resistance=None,
            current_price=21000.0,
            atr14=100.0,
        )
        assert decision.enforced_action_state == "STAND_DOWN"
        assert any("No qualified" in r for r in decision.stand_down_reasons)

    def test_stand_down_no_support_for_short(self):
        """ACTIVE_SHORT fails when nearest_support is None."""
        decision = enforce_scalper_policy(
            ticker="NQ",
            action_state="ACTIVE_SHORT",
            confidence=0.90,
            nearest_support=None,
            current_price=21000.0,
            atr14=100.0,
        )
        assert decision.enforced_action_state == "STAND_DOWN"

    def test_nearby_structure_passes(self):
        """Level within ATR limit → does not trigger proximity gate."""
        decision = enforce_scalper_policy(
            ticker="NQ",
            action_state="ACTIVE_LONG",
            confidence=0.90,
            nearest_resistance=21030.0,  # only 30 pts away
            current_price=21000.0,
            atr14=100.0,                 # limit = 100 pts
        )
        assert decision.enforced_action_state == "ACTIVE_LONG_1TO1"

    def test_stand_down_passthrough_for_stand_down_input(self):
        """If analysis is already STAND_DOWN, proximity gate is skipped."""
        decision = enforce_scalper_policy(
            ticker="NQ",
            action_state="STAND_DOWN",
            confidence=0.90,
            nearest_resistance=None,
            current_price=21000.0,
            atr14=100.0,
        )
        # No proximity reason added for STAND_DOWN input
        assert "No qualified" not in " ".join(decision.stand_down_reasons)


# ---------------------------------------------------------------------------
# enforce_scalper_policy – valid setup: 1:1 template applied
# ---------------------------------------------------------------------------


class TestValidSetup:
    def test_active_long_produces_1to1(self):
        decision = enforce_scalper_policy(
            ticker="NQ",
            action_state="ACTIVE_LONG",
            confidence=0.90,
            nearest_resistance=21050.0,
            current_price=21000.0,
            atr14=200.0,
        )
        assert decision.enforced_action_state == "ACTIVE_LONG_1TO1"
        assert decision.rr_target == 1.0
        assert decision.template_350 is not None
        assert decision.template_350.symbol == "NQ"
        assert decision.template_350.template_price_distance == 87.5
        assert decision.template_350.estimated_risk_usd == pytest.approx(1750.0)
        assert decision.lockout_active is False
        assert decision.stand_down_reasons == []

    def test_active_short_produces_1to1(self):
        decision = enforce_scalper_policy(
            ticker="NQ",
            action_state="ACTIVE_SHORT",
            confidence=0.85,
            nearest_support=20950.0,
            current_price=21000.0,
            atr14=200.0,
        )
        assert decision.enforced_action_state == "ACTIVE_SHORT_1TO1"
        assert decision.rr_target == 1.0

    def test_template_es(self):
        decision = enforce_scalper_policy(
            ticker="ES",
            action_state="ACTIVE_LONG",
            confidence=0.80,
            nearest_resistance=5540.0,
            current_price=5500.0,
            atr14=100.0,
        )
        tpl = decision.template_350
        assert tpl is not None
        assert tpl.symbol == "ES"
        assert tpl.template_price_distance == 35.0
        assert tpl.estimated_risk_usd == pytest.approx(1750.0)

    def test_template_ym(self):
        decision = enforce_scalper_policy(
            ticker="YM",
            action_state="ACTIVE_LONG",
            confidence=0.80,
            nearest_resistance=44500.0,
            current_price=44000.0,
            atr14=1000.0,
        )
        tpl = decision.template_350
        assert tpl is not None
        assert tpl.symbol == "YM"
        assert tpl.template_price_distance == 350
        assert tpl.estimated_risk_usd == pytest.approx(1750.0)

    def test_stand_down_state_passthrough(self):
        """STAND_DOWN analysis → STAND_DOWN policy (not active 1:1)."""
        decision = enforce_scalper_policy(
            ticker="NQ",
            action_state="STAND_DOWN",
            confidence=0.90,
            current_price=21000.0,
            atr14=100.0,
        )
        assert decision.enforced_action_state == "STAND_DOWN"

    def test_unknown_ticker_no_template(self):
        """Unknown ticker has no template but still returns a decision."""
        decision = enforce_scalper_policy(
            ticker="UNKNOWN",
            action_state="ACTIVE_LONG",
            confidence=0.90,
            nearest_resistance=100.5,
            current_price=100.0,
            atr14=5.0,
        )
        assert decision.template_350 is None


# ---------------------------------------------------------------------------
# check_extraction_quality_gates
# ---------------------------------------------------------------------------


class TestExtractionQualityGates:
    def test_passes_valid_nq_levels(self):
        result = check_extraction_quality_gates(
            ticker="NQ",
            levels=[21000.0, 20800.0, 21200.0],
        )
        assert result.passed is True
        assert result.rejection_reason is None

    def test_rejects_impossible_ym_scale(self):
        """YM level of 2.41 is physically impossible (floor = 10000)."""
        result = check_extraction_quality_gates(
            ticker="YM",
            levels=[2.41, 3.50],
        )
        assert result.passed is False
        assert result.rejection_reason is not None
        assert "Impossible scale" in result.rejection_reason
        assert "2.41" in result.rejection_reason

    def test_rejects_impossible_nq_scale(self):
        result = check_extraction_quality_gates(
            ticker="NQ",
            levels=[500.0],  # NQ floor = 10000
        )
        assert result.passed is False
        assert "Impossible scale" in (result.rejection_reason or "")

    def test_rejects_out_of_axis_bounds(self):
        result = check_extraction_quality_gates(
            ticker="NQ",
            levels=[25000.0, 21000.0],
            axis_min=20000.0,
            axis_max=22000.0,
        )
        # 25000 > 22000 * 1.10 = 24200 → rejected
        assert result.passed is False
        assert result.rejection_reason is not None

    def test_accepts_within_axis_tolerance(self):
        result = check_extraction_quality_gates(
            ticker="NQ",
            levels=[21900.0, 20200.0],
            axis_min=20000.0,
            axis_max=22000.0,
        )
        assert result.passed is True

    def test_rejects_empty_levels(self):
        result = check_extraction_quality_gates(
            ticker="NQ",
            levels=[],
        )
        assert result.passed is False
        assert result.rejection_reason is not None

    def test_rejects_below_minimum_distinct_levels(self):
        """Duplicate-collapsed: only 1 distinct level when minimum = 1 passes,
        but config minimum is 1 so this should pass. Test 0 distinct."""
        result = check_extraction_quality_gates(
            ticker="NQ",
            levels=[],
        )
        assert result.passed is False

    def test_unknown_ticker_passes_on_no_floor(self):
        """Unknown ticker has no price floor configured → gate skipped."""
        result = check_extraction_quality_gates(
            ticker="UNKNOWN",
            levels=[5.0, 10.0],
        )
        # No floor → impossible-scale gate skipped; only minimum-levels gate
        assert result.passed is True

    def test_rejects_mixed_valid_and_invalid_levels(self):
        result = check_extraction_quality_gates(
            ticker="YM",
            levels=[44000.0, 2.41],  # 2.41 is below YM floor
        )
        assert result.passed is False
