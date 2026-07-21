"""
Unit tests for canonical schema and validation contracts (PR1).

Covers:
- Valid AnalysisPayload passes validation
- Missing required fields produce precise field-path errors
- Invalid numeric fields are rejected with clear messages
- AnalysisResult enforces exactly 2 items in each bucket
- ActionState enum accepts only valid values
- High/low pair constraint is enforced
"""

from __future__ import annotations

import copy
import json
import os
import sys
import pytest

# Ensure the project root is on the path so imports work from any CWD.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from packages.core.validators import (
    PayloadValidationError,
    ResultValidationError,
    validate_payload,
    validate_result,
)
from packages.core.models import ActionState, AnalysisPayload, AnalysisResult


# ---------------------------------------------------------------------------
# Fixtures — canonical valid data
# ---------------------------------------------------------------------------

VALID_LEVELS = {
    "pdh": 29198.50,
    "pdl": 28701.25,
    "prior_settle": 29117.25,
    "rth_open": 29125.00,
    "globex_high": 29195.25,
    "globex_low": 28701.00,
    "asia_high": 28804.00,
    "asia_low": 28701.00,
    "london_high": 29195.25,
    "london_low": 28803.00,
    "ny_high": 29191.25,
    "ny_low": 28751.25,
    "asia_ib_high": 28780.00,
    "asia_ib_low": 28720.00,
    "london_ib_high": 29100.00,
    "london_ib_low": 28950.00,
    "ny_ib_high": 29191.00,
    "ny_ib_low": 28906.00,
    "atr14": 57.75,
}

VALID_PAYLOAD = {
    "date_et": "2026-07-21",
    "ticker": "NQU2026",
    "timeframe": "30m",
    "lookback_days": 14,
    "current_price": 29142.25,
    "levels": VALID_LEVELS,
}


def _level_item(
    level: str = "29195-29205",
    source: str = "PDH",
    conviction: str = "HIGH",
    trigger_note: str = "Rejection expected on first test",
) -> dict:
    return {
        "level": level,
        "source": source,
        "conviction": conviction,
        "trigger_note": trigger_note,
    }


VALID_RESULT = {
    "strongest_resistance": [_level_item("29195-29205", "PDH"), _level_item("29320-29350", "Globex High")],
    "weakest_resistance": [_level_item("29115-29125", "NY IB High", "LOW"), _level_item("29165-29175", "London High", "LOW")],
    "strongest_support": [_level_item("29000-29020", "Prior Settle", "HIGH"), _level_item("28885-28905", "PDL", "HIGH")],
    "weakest_support": [_level_item("28955-28970", "NY Low", "LOW"), _level_item("28760-28780", "Asia Low", "LOW")],
    "action_state": "STAND_DOWN",
    "confidence": 0.62,
    "rationale": [
        "Price is between PDH and prior settle — no clear directional acceptance.",
        "Globex range is unusually wide; avoid chasing.",
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _payload_without(*keys: str) -> dict:
    """Return a copy of VALID_PAYLOAD with top-level keys removed."""
    p = copy.deepcopy(VALID_PAYLOAD)
    for k in keys:
        p.pop(k, None)
    return p


def _payload_without_level(*level_keys: str) -> dict:
    """Return a copy of VALID_PAYLOAD with specific level sub-fields removed."""
    p = copy.deepcopy(VALID_PAYLOAD)
    for k in level_keys:
        p["levels"].pop(k, None)
    return p


def _with_level(**overrides: object) -> dict:
    """Return VALID_PAYLOAD with level sub-fields overridden."""
    p = copy.deepcopy(VALID_PAYLOAD)
    p["levels"].update(overrides)
    return p


def _with_top(**overrides: object) -> dict:
    """Return VALID_PAYLOAD with top-level fields overridden."""
    p = copy.deepcopy(VALID_PAYLOAD)
    p.update(overrides)
    return p


def _error_paths(exc: PayloadValidationError | ResultValidationError) -> list[str]:
    """Extract just the field-path prefix from each error string."""
    return [msg.split(":")[0] for msg in exc.field_errors]


# ===========================================================================
# AnalysisPayload — passing test
# ===========================================================================

class TestValidPayload:
    def test_valid_payload_passes(self) -> None:
        payload = validate_payload(VALID_PAYLOAD)
        assert isinstance(payload, AnalysisPayload)

    def test_field_values_preserved(self) -> None:
        payload = validate_payload(VALID_PAYLOAD)
        assert payload.ticker == "NQU2026"
        assert payload.timeframe == "30m"
        assert payload.lookback_days == 14
        assert payload.current_price == 29142.25
        assert payload.levels.atr14 == 57.75

    def test_ticker_is_stripped(self) -> None:
        payload = validate_payload(_with_top(ticker="  ESU2026  "))
        assert payload.ticker == "ESU2026"

    def test_valid_payload_from_file(self) -> None:
        """The committed example file must also pass validation."""
        example_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "examples",
            "nq_sample_valid.json",
        )
        with open(example_path) as fh:
            raw = json.load(fh)
        payload = validate_payload(raw)
        assert payload.ticker == "NQU2026"


# ===========================================================================
# AnalysisPayload — missing field tests
# ===========================================================================

class TestMissingTopLevelFields:
    @pytest.mark.parametrize("field", ["date_et", "ticker", "timeframe", "lookback_days", "current_price", "levels"])
    def test_missing_required_field(self, field: str) -> None:
        with pytest.raises(PayloadValidationError) as exc_info:
            validate_payload(_payload_without(field))
        assert field in _error_paths(exc_info.value), (
            f"Expected error path containing '{field}', got {exc_info.value.field_errors}"
        )

    def test_error_message_contains_field_required(self) -> None:
        with pytest.raises(PayloadValidationError) as exc_info:
            validate_payload(_payload_without("ticker"))
        error_text = str(exc_info.value)
        assert "ticker" in error_text
        assert "required" in error_text.lower()


class TestMissingLevelFields:
    ALL_LEVEL_FIELDS = list(VALID_LEVELS.keys())

    @pytest.mark.parametrize("field", ALL_LEVEL_FIELDS)
    def test_missing_level_field(self, field: str) -> None:
        with pytest.raises(PayloadValidationError) as exc_info:
            validate_payload(_payload_without_level(field))
        paths = _error_paths(exc_info.value)
        # Path must reference the levels sub-object
        assert any("levels" in p for p in paths), (
            f"Expected path containing 'levels' for missing field '{field}', got {paths}"
        )

    def test_error_includes_exact_subfield(self) -> None:
        with pytest.raises(PayloadValidationError) as exc_info:
            validate_payload(_payload_without_level("atr14"))
        paths = _error_paths(exc_info.value)
        assert any("atr14" in p for p in paths)


# ===========================================================================
# AnalysisPayload — invalid type / value tests
# ===========================================================================

class TestInvalidNumericFields:
    @pytest.mark.parametrize("level_field", ["pdh", "pdl", "atr14", "globex_high", "ny_ib_low"])
    def test_non_numeric_level_rejected(self, level_field: str) -> None:
        with pytest.raises(PayloadValidationError) as exc_info:
            validate_payload(_with_level(**{level_field: "not-a-number"}))
        paths = _error_paths(exc_info.value)
        assert any(level_field in p for p in paths), (
            f"Expected path containing '{level_field}', got {paths}"
        )

    def test_negative_atr14_rejected(self) -> None:
        with pytest.raises(PayloadValidationError) as exc_info:
            validate_payload(_with_level(atr14=-1.0))
        assert any("atr14" in p for p in _error_paths(exc_info.value))

    def test_zero_atr14_rejected(self) -> None:
        with pytest.raises(PayloadValidationError) as exc_info:
            validate_payload(_with_level(atr14=0.0))
        assert any("atr14" in p for p in _error_paths(exc_info.value))

    def test_non_numeric_current_price_rejected(self) -> None:
        with pytest.raises(PayloadValidationError) as exc_info:
            validate_payload(_with_top(current_price="high"))
        assert any("current_price" in p for p in _error_paths(exc_info.value))

    def test_negative_current_price_rejected(self) -> None:
        with pytest.raises(PayloadValidationError) as exc_info:
            validate_payload(_with_top(current_price=-999.0))
        assert any("current_price" in p for p in _error_paths(exc_info.value))

    def test_non_integer_lookback_days_rejected(self) -> None:
        with pytest.raises(PayloadValidationError) as exc_info:
            validate_payload(_with_top(lookback_days="two weeks"))
        assert any("lookback_days" in p for p in _error_paths(exc_info.value))

    def test_zero_lookback_days_rejected(self) -> None:
        with pytest.raises(PayloadValidationError) as exc_info:
            validate_payload(_with_top(lookback_days=0))
        assert any("lookback_days" in p for p in _error_paths(exc_info.value))

    def test_blank_ticker_rejected(self) -> None:
        with pytest.raises(PayloadValidationError) as exc_info:
            validate_payload(_with_top(ticker="   "))
        assert any("ticker" in p for p in _error_paths(exc_info.value))

    def test_invalid_date_format_rejected(self) -> None:
        with pytest.raises(PayloadValidationError) as exc_info:
            validate_payload(_with_top(date_et="07-21-2026"))
        assert any("date_et" in p for p in _error_paths(exc_info.value))


class TestHighLowConstraints:
    def test_pdh_below_pdl_rejected(self) -> None:
        with pytest.raises(PayloadValidationError) as exc_info:
            validate_payload(_with_level(pdh=100.0, pdl=200.0))
        assert exc_info.value.field_errors  # at least one error

    def test_globex_high_below_low_rejected(self) -> None:
        with pytest.raises(PayloadValidationError) as exc_info:
            validate_payload(_with_level(globex_high=100.0, globex_low=200.0))
        assert exc_info.value.field_errors


# ===========================================================================
# AnalysisResult — passing test
# ===========================================================================

class TestValidResult:
    def test_valid_result_passes(self) -> None:
        result = validate_result(VALID_RESULT)
        assert isinstance(result, AnalysisResult)

    def test_action_state_enum(self) -> None:
        result = validate_result(VALID_RESULT)
        assert result.action_state == ActionState.STAND_DOWN

    def test_confidence_range(self) -> None:
        result = validate_result(VALID_RESULT)
        assert 0.0 <= result.confidence <= 1.0

    def test_rationale_nonempty(self) -> None:
        result = validate_result(VALID_RESULT)
        assert len(result.rationale) >= 1


# ===========================================================================
# AnalysisResult — bucket size enforcement (exactly 2 items each)
# ===========================================================================

class TestResultBucketSizes:
    BUCKETS = [
        "strongest_resistance",
        "weakest_resistance",
        "strongest_support",
        "weakest_support",
    ]

    def _result_with(self, bucket: str, items: list) -> dict:
        r = copy.deepcopy(VALID_RESULT)
        r[bucket] = items
        return r

    @pytest.mark.parametrize("bucket", BUCKETS)
    def test_bucket_with_one_item_rejected(self, bucket: str) -> None:
        bad = self._result_with(bucket, [_level_item()])
        with pytest.raises(ResultValidationError) as exc_info:
            validate_result(bad)
        assert any(bucket in p for p in _error_paths(exc_info.value))

    @pytest.mark.parametrize("bucket", BUCKETS)
    def test_bucket_with_three_items_rejected(self, bucket: str) -> None:
        bad = self._result_with(bucket, [_level_item(), _level_item(), _level_item()])
        with pytest.raises(ResultValidationError) as exc_info:
            validate_result(bad)
        assert any(bucket in p for p in _error_paths(exc_info.value))

    @pytest.mark.parametrize("bucket", BUCKETS)
    def test_bucket_with_empty_list_rejected(self, bucket: str) -> None:
        bad = self._result_with(bucket, [])
        with pytest.raises(ResultValidationError) as exc_info:
            validate_result(bad)
        assert any(bucket in p for p in _error_paths(exc_info.value))

    @pytest.mark.parametrize("bucket", BUCKETS)
    def test_bucket_with_two_items_passes(self, bucket: str) -> None:
        """Confirm two items per bucket is the valid state."""
        good = self._result_with(bucket, [_level_item(), _level_item()])
        result = validate_result(good)
        assert len(getattr(result, bucket)) == 2


class TestResultInvalidActionState:
    def test_invalid_action_state_rejected(self) -> None:
        bad = copy.deepcopy(VALID_RESULT)
        bad["action_state"] = "BUY_THE_DIP"
        with pytest.raises(ResultValidationError) as exc_info:
            validate_result(bad)
        assert any("action_state" in p for p in _error_paths(exc_info.value))

    @pytest.mark.parametrize("state", ["ACTIVE_LONG", "ACTIVE_SHORT", "STAND_DOWN"])
    def test_all_valid_action_states_accepted(self, state: str) -> None:
        r = copy.deepcopy(VALID_RESULT)
        r["action_state"] = state
        result = validate_result(r)
        assert result.action_state.value == state

    def test_confidence_above_1_rejected(self) -> None:
        bad = copy.deepcopy(VALID_RESULT)
        bad["confidence"] = 1.5
        with pytest.raises(ResultValidationError) as exc_info:
            validate_result(bad)
        assert any("confidence" in p for p in _error_paths(exc_info.value))

    def test_confidence_below_0_rejected(self) -> None:
        bad = copy.deepcopy(VALID_RESULT)
        bad["confidence"] = -0.1
        with pytest.raises(ResultValidationError) as exc_info:
            validate_result(bad)
        assert any("confidence" in p for p in _error_paths(exc_info.value))

    def test_empty_rationale_rejected(self) -> None:
        bad = copy.deepcopy(VALID_RESULT)
        bad["rationale"] = []
        with pytest.raises(ResultValidationError) as exc_info:
            validate_result(bad)
        assert any("rationale" in p for p in _error_paths(exc_info.value))
