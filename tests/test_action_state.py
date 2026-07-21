"""
Unit tests for the action-state engine.

Coverage
--------
- STAND_DOWN under mixed / balanced scores.
- ACTIVE_SHORT when strong resistance is close and dominant.
- ACTIVE_LONG when strong support is close and dominant.
- STAND_DOWN when levels are far from current price.
- STAND_DOWN when either bucket is empty.
- Rationale string is non-empty for all states.
- Regression fixture: nq_stand_down produces STAND_DOWN.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from packages.core.action_state import compute_action_state
from packages.core.models import ActionState, ConvictionTag, LevelDecision
from packages.core.scoring import score
from packages.core.models import AnalysisPayload, LevelsPayload

FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ld(price: float, score_val: float, level_type: str = "resistance") -> LevelDecision:
    """Build a minimal LevelDecision for action-state unit tests."""
    conviction = (
        ConvictionTag.HIGH if score_val >= 0.70
        else ConvictionTag.MODERATE if score_val >= 0.45
        else ConvictionTag.LOW
    )
    return LevelDecision(
        price=price,
        sources=["test_source"],
        level_type=level_type,
        conviction=conviction,
        score=score_val,
        trigger_note=f"{conviction.value} {level_type}: test_source @ {price:.2f}",
    )


def _load_fixture_payload(name: str) -> AnalysisPayload:
    with open(FIXTURES_DIR / name) as f:
        data = json.load(f)
    return AnalysisPayload(
        date_et=data["date_et"],
        ticker=data["ticker"],
        timeframe=data["timeframe"],
        lookback_days=data["lookback_days"],
        current_price=data["current_price"],
        levels=LevelsPayload(**data["levels"]),
    )


# ---------------------------------------------------------------------------
# 1. STAND_DOWN — empty buckets
# ---------------------------------------------------------------------------


class TestStandDownEmptyBuckets:
    def test_no_resistance(self):
        state, rationale = compute_action_state(
            current_price=100.0,
            strongest_resistance=[],
            strongest_support=[_ld(95.0, 0.80, "support")],
            atr14=5.0,
        )
        assert state == ActionState.STAND_DOWN

    def test_no_support(self):
        state, rationale = compute_action_state(
            current_price=100.0,
            strongest_resistance=[_ld(105.0, 0.80, "resistance")],
            strongest_support=[],
            atr14=5.0,
        )
        assert state == ActionState.STAND_DOWN

    def test_both_empty(self):
        state, _ = compute_action_state(
            current_price=100.0,
            strongest_resistance=[],
            strongest_support=[],
        )
        assert state == ActionState.STAND_DOWN


# ---------------------------------------------------------------------------
# 2. STAND_DOWN — mixed / balanced scores
# ---------------------------------------------------------------------------


class TestStandDownMixedSignals:
    def test_balanced_scores_near_both_levels(self):
        """Equal scores on both sides → STAND_DOWN regardless of proximity."""
        current_price = 100.0
        atr14 = 10.0
        state, rationale = compute_action_state(
            current_price=current_price,
            strongest_resistance=[_ld(103.0, 0.75, "resistance")],
            strongest_support=[_ld(97.0, 0.75, "support")],
            atr14=atr14,
        )
        assert state == ActionState.STAND_DOWN
        assert rationale  # non-empty

    def test_nearly_equal_scores_within_dominance_ratio(self):
        """Scores differing by < 15 % → STAND_DOWN."""
        state, _ = compute_action_state(
            current_price=100.0,
            strongest_resistance=[_ld(102.0, 0.80, "resistance")],
            strongest_support=[_ld(98.0, 0.76, "support")],   # ratio ≈ 1.053 < 1.15
            atr14=5.0,
        )
        assert state == ActionState.STAND_DOWN

    def test_stand_down_fixture_regression(self):
        """The nq_stand_down fixture must produce STAND_DOWN."""
        payload = _load_fixture_payload("nq_stand_down.json")
        result = score(payload)
        assert result.action_state == ActionState.STAND_DOWN


# ---------------------------------------------------------------------------
# 3. ACTIVE_SHORT — resistance close and dominant
# ---------------------------------------------------------------------------


class TestActiveShort:
    def test_near_dominant_resistance(self):
        """
        Resistance within 0.75 ATR and score >> support → ACTIVE_SHORT.
        """
        current_price = 100.0
        atr14 = 10.0
        # resistance at 105 (dist=5 > 0.75*10=7.5 → NOT near) — need dist < 7.5
        # resistance at 104 (dist=4 < 7.5 → near)
        state, rationale = compute_action_state(
            current_price=current_price,
            strongest_resistance=[_ld(104.0, 0.90, "resistance")],   # 4 away, near
            strongest_support=[_ld(70.0, 0.50, "support")],           # far, weak
            atr14=atr14,
        )
        assert state == ActionState.ACTIVE_SHORT, f"Expected ACTIVE_SHORT, got {state}: {rationale}"

    def test_resistance_dominant_score_ratio_above_threshold(self):
        """Score ratio must be above SCORE_DOMINANCE_RATIO (1.15) to trigger ACTIVE_SHORT."""
        current_price = 100.0
        atr14 = 20.0
        # resistance at 103 (dist=3, within 0.75*20=15)
        # resistance score=0.92, support score=0.75; ratio = 0.92/0.75 ≈ 1.227 > 1.15
        state, _ = compute_action_state(
            current_price=current_price,
            strongest_resistance=[_ld(103.0, 0.92, "resistance")],
            strongest_support=[_ld(98.0, 0.75, "support")],
            atr14=atr14,
        )
        assert state == ActionState.ACTIVE_SHORT

    def test_active_short_rationale_mentions_resistance(self):
        state, rationale = compute_action_state(
            current_price=100.0,
            strongest_resistance=[_ld(104.0, 0.90, "resistance")],
            strongest_support=[_ld(70.0, 0.50, "support")],
            atr14=10.0,
        )
        assert state == ActionState.ACTIVE_SHORT
        assert "resistance" in rationale.lower()


# ---------------------------------------------------------------------------
# 4. ACTIVE_LONG — support close and dominant
# ---------------------------------------------------------------------------


class TestActiveLong:
    def test_near_dominant_support(self):
        """
        Support within 0.75 ATR and score >> resistance → ACTIVE_LONG.
        """
        current_price = 100.0
        atr14 = 10.0
        # support at 96 (dist=4 < 7.5 → near), score=0.90 >> resistance 0.50
        state, rationale = compute_action_state(
            current_price=current_price,
            strongest_resistance=[_ld(130.0, 0.50, "resistance")],   # far, weak
            strongest_support=[_ld(96.0, 0.90, "support")],           # near, strong
            atr14=atr14,
        )
        assert state == ActionState.ACTIVE_LONG, f"Expected ACTIVE_LONG, got {state}: {rationale}"

    def test_active_long_rationale_mentions_support(self):
        state, rationale = compute_action_state(
            current_price=100.0,
            strongest_resistance=[_ld(130.0, 0.50, "resistance")],
            strongest_support=[_ld(96.0, 0.90, "support")],
            atr14=10.0,
        )
        assert state == ActionState.ACTIVE_LONG
        assert "support" in rationale.lower()


# ---------------------------------------------------------------------------
# 5. STAND_DOWN — levels not in proximity
# ---------------------------------------------------------------------------


class TestStandDownNoProximity:
    def test_both_levels_far(self):
        """Even if one side dominates, far levels must not trigger active state."""
        current_price = 100.0
        atr14 = 5.0
        # threshold = 0.75 * 5 = 3.75
        # resistance at 110 (dist=10, not near), support at 90 (dist=10, not near)
        state, _ = compute_action_state(
            current_price=current_price,
            strongest_resistance=[_ld(110.0, 0.95, "resistance")],
            strongest_support=[_ld(90.0, 0.40, "support")],
            atr14=atr14,
        )
        # resistance score 0.95 / support score 0.40 = 2.375 > 1.15 → dominance OK
        # but resistance is not near (dist=10 > 3.75)
        # support also not near
        assert state == ActionState.STAND_DOWN


# ---------------------------------------------------------------------------
# 6. Rationale non-empty for all states
# ---------------------------------------------------------------------------


class TestRationaleNonEmpty:
    @pytest.mark.parametrize("fixture_name", [
        "nq_sample.json",
        "es_sample.json",
        "nq_stand_down.json",
    ])
    def test_rationale_non_empty(self, fixture_name: str):
        payload = _load_fixture_payload(fixture_name)
        result = score(payload)
        assert result.rationale, f"Rationale empty for {fixture_name}"

    def test_direct_compute_rationale_non_empty(self):
        _, rationale = compute_action_state(
            current_price=100.0,
            strongest_resistance=[_ld(105.0, 0.70, "resistance")],
            strongest_support=[_ld(95.0, 0.70, "support")],
            atr14=5.0,
        )
        assert rationale


# ---------------------------------------------------------------------------
# 7. Fallback (no ATR14) tests
# ---------------------------------------------------------------------------


class TestFallbackNoATR:
    def test_stand_down_no_atr_balanced(self):
        """Without ATR14, fallback proximity uses 0.5 % of price."""
        state, _ = compute_action_state(
            current_price=100.0,
            strongest_resistance=[_ld(100.4, 0.80, "resistance")],
            strongest_support=[_ld(99.6, 0.76, "support")],
            atr14=None,
        )
        # Scores 0.80/0.76 ≈ 1.053 < 1.15 → STAND_DOWN
        assert state == ActionState.STAND_DOWN

    def test_active_short_no_atr(self):
        """Without ATR14, proximity uses 0.5 % of price (= 0.5 for price=100)."""
        state, _ = compute_action_state(
            current_price=100.0,
            # resistance at 100.3 (dist=0.3 < 0.5 → near), dominant
            strongest_resistance=[_ld(100.3, 0.95, "resistance")],
            strongest_support=[_ld(80.0, 0.40, "support")],
            atr14=None,
        )
        assert state == ActionState.ACTIVE_SHORT
