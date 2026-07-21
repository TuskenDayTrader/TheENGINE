"""
Unit tests for packages/core/image_extractor.py.

Covers:
- Pixel-to-price mapping (_map_y_to_price)
- Line deduplication (_deduplicate_lines)
- Levels payload construction (_build_levels_payload)
- Confidence calculation (_compute_confidence)
- Horizontal line detection with synthetic images
- Full extract_from_image entry point for invalid/blank inputs
"""
from __future__ import annotations

import io

import numpy as np
import pytest

from packages.core.image_extractor import (
    CONFIDENCE_LOW_THRESHOLD,
    ExtractionResult,
    _build_levels_payload,
    _compute_confidence,
    _deduplicate_lines,
    _detect_horizontal_lines,
    _map_y_to_price,
    extract_from_image,
)
from packages.core.models import LevelsPayload


# ---------------------------------------------------------------------------
# _map_y_to_price
# ---------------------------------------------------------------------------


class TestMapYToPrice:
    """Pixel-to-price linear interpolation and extrapolation."""

    _AXIS = [(100, 21000.0), (200, 20000.0), (300, 19000.0)]

    def test_interpolates_midpoint(self):
        price = _map_y_to_price(150, self._AXIS)
        assert price == pytest.approx(20500.0)

    def test_interpolates_three_quarters(self):
        price = _map_y_to_price(175, self._AXIS)
        assert price == pytest.approx(20250.0)

    def test_exact_first_point(self):
        assert _map_y_to_price(100, self._AXIS) == pytest.approx(21000.0)

    def test_exact_last_point(self):
        assert _map_y_to_price(300, self._AXIS) == pytest.approx(19000.0)

    def test_extrapolates_above_range(self):
        price = _map_y_to_price(50, self._AXIS)
        assert price == pytest.approx(21500.0)

    def test_extrapolates_below_range(self):
        price = _map_y_to_price(350, self._AXIS)
        assert price == pytest.approx(18500.0)

    def test_single_point_returns_none(self):
        assert _map_y_to_price(100, [(100, 20000.0)]) is None

    def test_empty_axis_returns_none(self):
        assert _map_y_to_price(100, []) is None

    def test_two_points(self):
        axis = [(0, 100.0), (100, 0.0)]
        assert _map_y_to_price(50, axis) == pytest.approx(50.0)

    def test_price_decreases_as_y_increases(self):
        """Verify the slope sign: higher y → lower price."""
        axis = [(100, 22000.0), (500, 18000.0)]
        p100 = _map_y_to_price(100, axis)
        p500 = _map_y_to_price(500, axis)
        assert p100 > p500  # type: ignore[operator]


# ---------------------------------------------------------------------------
# _deduplicate_lines
# ---------------------------------------------------------------------------


class TestDeduplicateLines:
    def test_removes_within_tolerance(self):
        lines = [(100, "green"), (103, "red"), (200, "green")]
        result = _deduplicate_lines(lines, tol=5)
        assert len(result) == 2

    def test_keeps_lines_outside_tolerance(self):
        lines = [(100, "green"), (200, "red"), (300, "green")]
        result = _deduplicate_lines(lines, tol=5)
        assert len(result) == 3

    def test_empty_returns_empty(self):
        assert _deduplicate_lines([]) == []

    def test_single_line_unchanged(self):
        assert _deduplicate_lines([(100, "green")]) == [(100, "green")]

    def test_preserves_first_on_tie(self):
        """When two lines are within tolerance, the lower y-value (first) is kept."""
        lines = [(100, "green"), (102, "red")]
        result = _deduplicate_lines(lines, tol=5)
        assert len(result) == 1
        assert result[0][0] == 100

    def test_boundary_tolerance(self):
        """Lines exactly tol pixels apart are kept as separate."""
        lines = [(100, "green"), (106, "red")]  # diff = 6 > tol=5
        result = _deduplicate_lines(lines, tol=5)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# _build_levels_payload
# ---------------------------------------------------------------------------


class TestBuildLevelsPayload:
    def test_pdh_is_highest_resistance(self):
        payload = _build_levels_payload([21000.0, 20500.0, 19500.0, 19000.0], 20000.0)
        assert payload.pdh == pytest.approx(21000.0)

    def test_pdl_is_closest_support(self):
        payload = _build_levels_payload([21000.0, 20500.0, 19500.0, 19000.0], 20000.0)
        # support sorted ascending: [19000, 19500]; pdl = support[0] = 19000
        assert payload.pdl == pytest.approx(19000.0)

    def test_prior_settle_equals_current_price(self):
        payload = _build_levels_payload([21000.0, 19000.0], 20000.0)
        assert payload.prior_settle == pytest.approx(20000.0)

    def test_atr14_is_positive(self):
        payload = _build_levels_payload([21000.0, 19000.0], 20000.0)
        assert payload.atr14 is not None
        assert payload.atr14 > 0

    def test_atr14_derived_from_range(self):
        # range = 21000 - 19000 = 2000; atr14 = 2000/14 ≈ 142.86
        payload = _build_levels_payload([21000.0, 19000.0], 20000.0)
        assert payload.atr14 == pytest.approx(2000.0 / 14.0, rel=1e-3)

    def test_no_resistance_uses_synthetic_pdh(self):
        payload = _build_levels_payload([19000.0, 18500.0], 20000.0)
        assert payload.pdh > 20000.0

    def test_no_support_uses_synthetic_pdl(self):
        payload = _build_levels_payload([21000.0, 22000.0], 20000.0)
        assert payload.pdl < 20000.0

    def test_multiple_resistance_levels_assigned(self):
        line_prices = [22000.0, 21500.0, 21000.0, 19000.0]
        payload = _build_levels_payload(line_prices, 20000.0)
        assert payload.pdh == pytest.approx(22000.0)
        assert payload.globex_high == pytest.approx(21500.0)
        assert payload.asia_high == pytest.approx(21000.0)

    def test_multiple_support_levels_assigned(self):
        line_prices = [21000.0, 19000.0, 18500.0, 18000.0]
        payload = _build_levels_payload(line_prices, 20000.0)
        assert payload.pdl == pytest.approx(18000.0)
        assert payload.globex_low == pytest.approx(18500.0)
        assert payload.asia_low == pytest.approx(19000.0)

    def test_returns_levels_payload_instance(self):
        payload = _build_levels_payload([21000.0, 19000.0], 20000.0)
        assert isinstance(payload, LevelsPayload)


# ---------------------------------------------------------------------------
# _compute_confidence
# ---------------------------------------------------------------------------


class TestComputeConfidence:
    def test_all_conditions_met(self):
        conf = _compute_confidence(
            num_axis_points=5,
            num_lines=3,
            num_mapped=2,
            cp_from_axis=True,
        )
        assert conf == pytest.approx(1.0)

    def test_no_conditions_met(self):
        conf = _compute_confidence(
            num_axis_points=0,
            num_lines=0,
            num_mapped=0,
            cp_from_axis=False,
        )
        assert conf == pytest.approx(0.0)

    def test_partial_conditions(self):
        conf = _compute_confidence(
            num_axis_points=5,
            num_lines=1,
            num_mapped=0,
            cp_from_axis=False,
        )
        # axis_points + lines = 0.25 + 0.25 = 0.50
        assert conf == pytest.approx(0.50)

    def test_axis_threshold_is_three(self):
        below = _compute_confidence(num_axis_points=2, num_lines=0, num_mapped=0, cp_from_axis=False)
        at = _compute_confidence(num_axis_points=3, num_lines=0, num_mapped=0, cp_from_axis=False)
        assert below == pytest.approx(0.0)
        assert at == pytest.approx(0.25)

    def test_result_is_in_unit_interval(self):
        conf = _compute_confidence(100, 100, 100, True)
        assert 0.0 <= conf <= 1.0


# ---------------------------------------------------------------------------
# _detect_horizontal_lines  (synthetic image)
# ---------------------------------------------------------------------------


class TestDetectHorizontalLines:
    """Use small synthetic NumPy images to test line detection."""

    @staticmethod
    def _blank(h: int = 400, w: int = 800) -> np.ndarray:
        return np.zeros((h, w, 3), dtype=np.uint8)

    def test_detects_tradingview_green_line(self):
        img = self._blank()
        # TradingView green #089981: BGR = (129, 153, 8)
        img[200, 80:700] = [129, 153, 8]
        lines = _detect_horizontal_lines(img)
        assert len(lines) > 0, "Expected at least one green line"
        y_vals = [y for y, _ in lines]
        assert any(abs(y - 200) <= 5 for y in y_vals)

    def test_detects_tradingview_red_line(self):
        img = self._blank()
        # TradingView red #f23645: BGR = (69, 54, 242)
        img[150, 80:700] = [69, 54, 242]
        lines = _detect_horizontal_lines(img)
        assert len(lines) > 0, "Expected at least one red line"
        y_vals = [y for y, _ in lines]
        assert any(abs(y - 150) <= 5 for y in y_vals)

    def test_green_line_labeled_green(self):
        img = self._blank()
        img[200, 80:700] = [129, 153, 8]
        lines = _detect_horizontal_lines(img)
        colors = [c for _, c in lines]
        assert "green" in colors

    def test_red_line_labeled_red(self):
        img = self._blank()
        img[150, 80:700] = [69, 54, 242]
        lines = _detect_horizontal_lines(img)
        colors = [c for _, c in lines]
        assert "red" in colors

    def test_no_lines_on_blank_image(self):
        img = self._blank()
        lines = _detect_horizontal_lines(img)
        assert lines == []

    def test_short_segment_not_detected(self):
        """A segment shorter than 10 % of image width should not be detected."""
        img = self._blank()
        # Only 5 % of width → below _LINE_MIN_WIDTH_FRACTION
        img[200, 0:40] = [129, 153, 8]  # 40 / 800 = 5 %
        lines = _detect_horizontal_lines(img)
        assert lines == []

    def test_multiple_lines_detected(self):
        img = self._blank()
        img[100, 80:700] = [129, 153, 8]  # green at y=100
        img[300, 80:700] = [69, 54, 242]  # red at y=300
        lines = _detect_horizontal_lines(img)
        assert len(lines) >= 2


# ---------------------------------------------------------------------------
# extract_from_image  (entry point, invalid / edge-case inputs)
# ---------------------------------------------------------------------------


class TestExtractFromImage:
    def test_empty_bytes_returns_undecoded(self):
        result = extract_from_image(b"")
        assert result.image_decoded is False
        assert result.extraction_confidence == 0.0
        assert result.levels_payload is None

    def test_invalid_bytes_returns_undecoded(self):
        result = extract_from_image(b"not-an-image-at-all")
        assert result.image_decoded is False
        assert result.extraction_confidence == 0.0

    def test_fake_png_header_returns_undecoded(self):
        # PNG magic bytes but invalid payload
        result = extract_from_image(b"\x89PNG\r\n\x1a\nfakepng")
        assert result.image_decoded is False

    def test_returns_extraction_result_type(self):
        result = extract_from_image(b"")
        assert isinstance(result, ExtractionResult)

    def test_blank_image_decoded_but_no_levels(self):
        """A blank (all-black) image decodes but has no detectable lines or labels."""
        # Encode a tiny blank image as PNG
        blank = np.zeros((100, 200, 3), dtype=np.uint8)
        import cv2

        ok, buf = cv2.imencode(".png", blank)
        assert ok
        result = extract_from_image(bytes(buf))
        assert result.image_decoded is True
        assert result.num_lines_detected == 0
        assert result.extraction_confidence < CONFIDENCE_LOW_THRESHOLD

    def test_image_with_green_line_decoded_and_line_detected(self):
        """An image with a green line should have image_decoded=True and ≥1 line."""
        img = np.zeros((400, 800, 3), dtype=np.uint8)
        img[200, 80:700] = [129, 153, 8]  # TradingView green
        import cv2

        ok, buf = cv2.imencode(".png", img)
        assert ok
        result = extract_from_image(bytes(buf))
        assert result.image_decoded is True
        assert result.num_lines_detected >= 1
