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
        """A segment shorter than 3 % of chart width should not be detected."""
        img = self._blank()
        # ~1.5 % of 800 px image width → below the 3 % _LINE_MIN_WIDTH_FRACTION.
        # chart_w ≈ 680 (after excluding 15 % price-axis); min_line_width = 20 px.
        img[200, 0:12] = [129, 153, 8]  # 12 px < 20 px minimum
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

    def test_debug_false_produces_no_debug_info(self):
        """When debug=False (default), debug_info must be None."""
        img = np.zeros((400, 800, 3), dtype=np.uint8)
        img[200, 80:700] = [129, 153, 8]
        import cv2

        ok, buf = cv2.imencode(".png", img)
        assert ok
        result = extract_from_image(bytes(buf), debug=False)
        assert result.debug_info is None

    def test_debug_true_populates_debug_info(self):
        """When debug=True, debug_info must contain the required keys."""
        img = np.zeros((400, 800, 3), dtype=np.uint8)
        img[200, 80:700] = [129, 153, 8]
        import cv2

        ok, buf = cv2.imencode(".png", img)
        assert ok
        result = extract_from_image(bytes(buf), debug=True)
        assert result.debug_info is not None
        di = result.debug_info
        assert "image_size" in di
        assert di["image_size"]["width"] == 800
        assert di["image_size"]["height"] == 400
        assert "chart_roi" in di
        assert "green_mask_pixels" in di
        assert "red_mask_pixels" in di
        assert "segments_before_dedup" in di
        assert "segments_after_dedup" in di


# ---------------------------------------------------------------------------
# Anti-aliased line detection
# ---------------------------------------------------------------------------


class TestAntiAliasedDetection:
    """Verify that lines blended with a dark background are still detected."""

    _BG = [34, 23, 19]   # TradingView dark theme background ≈ #131722
    _GREEN = [129, 153, 8]
    _RED = [69, 54, 242]

    @staticmethod
    def _blend(color, bg, alpha: float) -> list:
        return [int(alpha * c + (1 - alpha) * b) for c, b in zip(color, bg)]

    @staticmethod
    def _blank(h: int = 400, w: int = 800) -> np.ndarray:
        return np.zeros((h, w, 3), dtype=np.uint8)

    def test_detects_80pct_antialiased_green(self):
        """80 % blend of TradingView green with dark bg must be detected."""
        img = self._blank()
        blended = self._blend(self._GREEN, self._BG, 0.8)
        img[200, 80:700] = blended
        lines = _detect_horizontal_lines(img)
        assert len(lines) > 0, f"Expected line but got none (color={blended})"
        assert any(abs(y - 200) <= 5 for y, _ in lines)

    def test_detects_80pct_antialiased_red(self):
        """80 % blend of TradingView red with dark bg must be detected."""
        img = self._blank()
        blended = self._blend(self._RED, self._BG, 0.8)
        img[150, 80:700] = blended
        lines = _detect_horizontal_lines(img)
        assert len(lines) > 0, f"Expected line but got none (color={blended})"
        assert any(abs(y - 150) <= 5 for y, _ in lines)

    def test_3pixel_antialiased_green_line_detected(self):
        """A 3-row anti-aliased green line (core + 60 % blended rows) is detected."""
        img = self._blank()
        core_y = 200
        img[core_y, 80:700] = self._GREEN
        img[core_y - 1, 80:700] = self._blend(self._GREEN, self._BG, 0.6)
        img[core_y + 1, 80:700] = self._blend(self._GREEN, self._BG, 0.6)
        lines = _detect_horizontal_lines(img)
        assert len(lines) > 0
        assert any(abs(y - core_y) <= 5 for y, _ in lines)


# ---------------------------------------------------------------------------
# Chart ROI exclusion
# ---------------------------------------------------------------------------


class TestChartROI:
    """Lines outside the chart plot area should not be detected."""

    _GREEN = [129, 153, 8]

    @staticmethod
    def _blank(h: int = 400, w: int = 800) -> np.ndarray:
        return np.zeros((h, w, 3), dtype=np.uint8)

    def test_line_in_price_axis_not_detected(self):
        """A line drawn entirely in the rightmost 15 % (price axis) is excluded."""
        img = self._blank()
        axis_start = int(800 * 0.85) + 5  # well inside axis strip
        img[200, axis_start:795] = self._GREEN
        lines = _detect_horizontal_lines(img)
        assert lines == [], f"Expected no line from axis area, got {lines}"

    def test_line_in_title_area_not_detected(self):
        """A line drawn in the top 8 % (title/watermark) is excluded."""
        img = self._blank()
        title_y = int(400 * 0.04)  # 4 %, well inside top exclusion zone
        img[title_y, 80:600] = self._GREEN
        lines = _detect_horizontal_lines(img)
        assert lines == [], f"Expected no line from title area, got {lines}"

    def test_line_in_bottom_panel_not_detected(self):
        """A line drawn below 90 % of image height (session/ATR panel) is excluded."""
        img = self._blank()
        bottom_y = int(400 * 0.95)  # 95 %, inside bottom exclusion zone
        img[bottom_y, 80:600] = self._GREEN
        lines = _detect_horizontal_lines(img)
        assert lines == [], f"Expected no line from bottom panel, got {lines}"

    def test_line_in_chart_area_detected(self):
        """A line within the chart plot area is detected."""
        img = self._blank()
        chart_y = int(400 * 0.50)  # 50 %, centre of chart
        img[chart_y, 80:600] = self._GREEN
        lines = _detect_horizontal_lines(img)
        assert len(lines) > 0


# ---------------------------------------------------------------------------
# Candle-body suppression regression (real-world root-cause fix)
# ---------------------------------------------------------------------------


class TestCandleBodyMergedLines:
    """
    Lines of the same colour as a candlestick body merge into one connected
    component in the colour mask.  The old box-suppression logic incorrectly
    erased the entire component (including the level line) when the merged
    blob's height exceeded ``box_threshold_h``, producing zero detections.

    These tests verify that the fixed suppression preserves horizontally
    dominant blobs so that the Hough pass can still recover the level line.
    """

    _GREEN = [129, 153, 8]   # TradingView #089981 BGR
    _RED = [69, 54, 242]     # TradingView #f23645 BGR

    @staticmethod
    def _blank(h: int = 614, w: int = 1540) -> np.ndarray:
        """Full-HD-ish image matching common TradingView screenshot dimensions."""
        return np.zeros((h, w, 3), dtype=np.uint8)

    def test_green_line_merged_with_candle_body_detected(self):
        """A green level line that physically overlaps a green candle body is found."""
        img = self._blank()
        img[200, 80:1200] = self._GREEN          # horizontal level line
        img[180:220, 390:410] = self._GREEN      # candle body crossing y=200
        lines = _detect_horizontal_lines(img)
        assert len(lines) > 0, (
            "Green level line was erased by box suppression due to candle overlap"
        )
        y_vals = [y for y, _ in lines]
        assert any(abs(y - 200) <= 10 for y in y_vals)

    def test_red_line_merged_with_candle_body_detected(self):
        """A red level line that physically overlaps a red candle body is found."""
        img = self._blank()
        img[300, 80:1200] = self._RED
        img[280:320, 600:620] = self._RED
        lines = _detect_horizontal_lines(img)
        assert len(lines) > 0, (
            "Red level line was erased by box suppression due to candle overlap"
        )
        y_vals = [y for y, _ in lines]
        assert any(abs(y - 300) <= 10 for y in y_vals)

    def test_multiple_lines_each_merged_with_candles_all_detected(self):
        """Three green level lines each merged with multiple candle bodies are all found."""
        img = self._blank()
        for line_y in [150, 300, 450]:
            img[line_y, 80:1200] = self._GREEN
            for cx in [200, 450, 750, 1000]:
                img[line_y - 15 : line_y + 16, cx : cx + 12] = self._GREEN
        lines = _detect_horizontal_lines(img)
        assert len(lines) >= 3, (
            f"Expected ≥3 lines but got {len(lines)}; "
            "candle bodies may be suppressing level lines"
        )

    def test_narrow_candle_body_alone_not_detected_as_line(self):
        """A tall narrow candle body with NO horizontal line must NOT be reported."""
        img = self._blank()
        # Narrow candle body (12 px wide, 60 px tall) — no horizontal line
        img[180:240, 400:412] = self._GREEN
        lines = _detect_horizontal_lines(img)
        assert lines == [], (
            f"Standalone candle body should not be detected as a line; got {lines}"
        )


# ---------------------------------------------------------------------------
# Debug output from _detect_horizontal_lines
# ---------------------------------------------------------------------------


class TestDetectLinesDebugOut:
    @staticmethod
    def _green_image(h: int = 400, w: int = 800) -> np.ndarray:
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[200, 80:700] = [129, 153, 8]
        return img

    def test_debug_out_populated(self):
        img = self._green_image()
        debug: dict = {}
        lines = _detect_horizontal_lines(img, debug_out=debug)
        assert "chart_roi" in debug
        assert "green_mask_pixels" in debug
        assert "red_mask_pixels" in debug
        assert "segments_before_dedup" in debug
        assert "segments_after_dedup" in debug
        assert debug["green_mask_pixels"] > 0

    def test_debug_out_none_by_default(self):
        """Calling without debug_out must not raise and must return lines."""
        img = self._green_image()
        lines = _detect_horizontal_lines(img)
        assert len(lines) > 0


# ---------------------------------------------------------------------------
# _filter_outlier_prices — mechanical outlier rejection
# ---------------------------------------------------------------------------


from packages.core.image_extractor import _filter_outlier_prices


class TestFilterOutlierPrices:
    """Verify the 3-tier mechanical outlier rejection."""

    # A realistic NQ cluster around 28_000
    _NQ_CLUSTER = [27_900.0, 28_000.0, 28_100.0, 28_200.0, 28_300.0]
    _NQ_CURRENT = 28_000.0
    _NQ_ATR = 110.0  # ~realistic daily ATR for NQ

    def test_no_outliers_unchanged(self):
        clean, rejected = _filter_outlier_prices(
            self._NQ_CLUSTER, self._NQ_CURRENT, self._NQ_ATR
        )
        assert rejected == []
        assert set(clean) == set(self._NQ_CLUSTER)

    def test_atr_envelope_rejects_distant_price(self):
        """16 072 is ~110 ATR units away — must be rejected by Tier 1."""
        prices = self._NQ_CLUSTER + [16_072.0]
        clean, rejected = _filter_outlier_prices(prices, self._NQ_CURRENT, self._NQ_ATR)
        rejected_prices = {p for p, _ in rejected}
        assert 16_072.0 in rejected_prices, "16 072 should be rejected by ATR envelope"
        assert 16_072.0 not in clean

    def test_iqr_rejects_moderate_outlier_when_no_atr(self):
        """Without ATR, IQR tier still catches values far from the cluster."""
        # Inject a value 3× the cluster spread away from the median
        prices = [28_000.0, 28_100.0, 28_200.0, 28_300.0, 32_000.0]
        clean, rejected = _filter_outlier_prices(prices, 28_200.0, atr=None)
        rejected_prices = {p for p, _ in rejected}
        assert 32_000.0 in rejected_prices, "32 000 should be rejected by IQR filter"

    def test_peer_vote_isolation_rejects_singleton(self):
        """A lone value 25 % below current with no peers should be rejected."""
        # Cluster at 28_000; rogue value at 20_000 (28 % below current)
        prices = [28_000.0, 28_100.0, 28_200.0, 20_000.0]
        clean, rejected = _filter_outlier_prices(prices, 28_100.0, atr=None)
        rejected_prices = {p for p, _ in rejected}
        assert 20_000.0 in rejected_prices

    def test_empty_prices_returns_empty(self):
        clean, rejected = _filter_outlier_prices([], 28_000.0, 110.0)
        assert clean == []
        assert rejected == []

    def test_single_price_no_rejection(self):
        """A single price cannot be statistically rejected — return unchanged."""
        clean, rejected = _filter_outlier_prices([28_000.0], 28_000.0, 110.0)
        assert clean == [28_000.0]
        assert rejected == []

    def test_two_prices_atr_filter_still_runs(self):
        """With only 2 prices, the ATR tier should still fire for the outlier."""
        prices = [28_000.0, 5_000.0]
        clean, rejected = _filter_outlier_prices(prices, 28_000.0, 110.0)
        rejected_prices = {p for p, _ in rejected}
        assert 5_000.0 in rejected_prices

    def test_rejection_reason_contains_useful_text(self):
        """Rejection reason must contain diagnostic information."""
        prices = self._NQ_CLUSTER + [16_072.0]
        _, rejected = _filter_outlier_prices(prices, self._NQ_CURRENT, self._NQ_ATR)
        reasons = [r for _, r in rejected if abs(_ - 16_072.0) < 1]
        # Find the 16_072 rejection
        matching = [(p, r) for p, r in rejected if abs(p - 16_072.0) < 1]
        assert matching, "16 072 should appear in rejected list"
        _, reason = matching[0]
        assert "16072" in reason or "16" in reason  # price should be in reason

    def test_valid_far_support_not_rejected_when_atr_large(self):
        """A genuinely distant structural low should survive when ATR is large enough."""
        # ATR = 3 000 means ±15 000 envelope; 26 000 is within that range
        prices = [28_000.0, 28_200.0, 26_000.0]
        clean, rejected = _filter_outlier_prices(prices, 28_000.0, atr=3_000.0)
        # 26_000 is within 5× ATR of 28_000, so should not be rejected by Tier 1
        rejected_prices = {p for p, _ in rejected}
        assert 26_000.0 not in rejected_prices, (
            "26 000 is within 5× ATR envelope and should not be rejected"
        )


# ---------------------------------------------------------------------------
# _label_to_fields — multi-field compound label mapping  (PR 23)
# ---------------------------------------------------------------------------


from packages.core.image_extractor import _label_to_fields


class TestLabelToFields:
    """Verify that compound labels map to ALL matching LevelsPayload fields."""

    def test_simple_label_maps_to_one_field(self):
        assert _label_to_fields("New York Low") == ["ny_low"]

    def test_simple_label_case_insensitive(self):
        assert _label_to_fields("new york low") == ["ny_low"]

    def test_compound_label_maps_to_two_fields(self):
        fields = _label_to_fields("Prev Day High / London High")
        assert "pdh" in fields
        assert "london_high" in fields
        assert len(fields) == 2

    def test_compound_pdl_london_low(self):
        fields = _label_to_fields("Prev Day Low / London Low")
        assert "pdl" in fields
        assert "london_low" in fields

    def test_unknown_label_returns_empty(self):
        assert _label_to_fields("Garbage Label XYZ") == []

    def test_empty_string_returns_empty(self):
        assert _label_to_fields("") == []

    def test_no_duplicate_fields(self):
        """Even if label text matches the same field twice, no duplicates."""
        fields = _label_to_fields("Asia High")
        assert fields.count("asia_high") == 1

    def test_prev_week_high(self):
        assert _label_to_fields("Prev Week High") == ["globex_high"]

    def test_new_york_open(self):
        assert _label_to_fields("New York Open") == ["rth_open"]

    def test_london_open(self):
        assert _label_to_fields("London Open") == ["london_open"]

    def test_prev_4h_high(self):
        assert _label_to_fields("Prev 4H High") == ["ny_high"]

    def test_prev_4h_high_slash_new_york_high(self):
        """Compound 'Prev 4H High / New York High' -> ny_high (deduped)."""
        fields = _label_to_fields("Prev 4H High / New York High")
        assert "ny_high" in fields
        assert fields.count("ny_high") == 1


# ---------------------------------------------------------------------------
# _assign_interior_lines_to_ib_slots  (PR 23)
# ---------------------------------------------------------------------------


from packages.core.image_extractor import _assign_interior_lines_to_ib_slots


class TestAssignInteriorLinesToIbSlots:
    """Verify IB-slot assignment from unlabeled interior lines."""

    def test_assigns_ny_ib_when_ny_bounds_known(self):
        labeled = {"ny_low": 28_100.0, "ny_high": 28_500.0}
        unlabeled = [28_200.0, 28_400.0]
        slots = _assign_interior_lines_to_ib_slots(unlabeled, labeled)
        assert "ny_ib_low" in slots
        assert "ny_ib_high" in slots
        assert slots["ny_ib_low"] == pytest.approx(28_200.0)
        assert slots["ny_ib_high"] == pytest.approx(28_400.0)

    def test_no_assignment_when_bounds_missing(self):
        labeled = {"ny_high": 28_500.0}
        slots = _assign_interior_lines_to_ib_slots([28_200.0], labeled)
        assert "ny_ib_low" not in slots
        assert "ny_ib_high" not in slots

    def test_no_assignment_for_exterior_prices(self):
        labeled = {"ny_low": 28_100.0, "ny_high": 28_500.0}
        slots = _assign_interior_lines_to_ib_slots([28_000.0, 28_600.0], labeled)
        assert slots == {}

    def test_does_not_overwrite_existing_labeled_slot(self):
        labeled = {"ny_low": 28_100.0, "ny_high": 28_500.0, "ny_ib_low": 28_250.0}
        slots = _assign_interior_lines_to_ib_slots([28_200.0], labeled)
        assert "ny_ib_low" not in slots

    def test_london_ib_slots_assigned(self):
        labeled = {"london_low": 28_000.0, "london_high": 28_400.0}
        unlabeled = [28_150.0, 28_300.0]
        slots = _assign_interior_lines_to_ib_slots(unlabeled, labeled)
        assert "london_ib_low" in slots
        assert "london_ib_high" in slots

    def test_empty_unlabeled_returns_empty(self):
        labeled = {"ny_low": 28_100.0, "ny_high": 28_500.0}
        assert _assign_interior_lines_to_ib_slots([], labeled) == {}


# ---------------------------------------------------------------------------
# _compute_interior_confluence_zones  (PR 23)
# ---------------------------------------------------------------------------


from packages.core.image_extractor import _compute_interior_confluence_zones


class TestComputeInteriorConfluenceZones:
    """Verify clustering of line prices into confluence zones."""

    def test_single_price_yields_one_weak_zone(self):
        zones = _compute_interior_confluence_zones([28_000.0])
        assert len(zones) == 1
        assert zones[0]["strength"] == "weak"
        assert zones[0]["count"] == 1
        assert zones[0]["price"] == pytest.approx(28_000.0)

    def test_two_nearby_prices_merged_into_moderate(self):
        zones = _compute_interior_confluence_zones([28_000.0, 28_010.0], cluster_radius=20.0)
        assert len(zones) == 1
        assert zones[0]["strength"] == "moderate"
        assert zones[0]["count"] == 2

    def test_three_nearby_prices_create_strong_zone(self):
        zones = _compute_interior_confluence_zones(
            [28_000.0, 28_010.0, 28_015.0], cluster_radius=20.0
        )
        assert len(zones) == 1
        assert zones[0]["strength"] == "strong"
        assert zones[0]["count"] == 3

    def test_distant_prices_create_separate_zones(self):
        zones = _compute_interior_confluence_zones([28_000.0, 28_200.0], cluster_radius=20.0)
        assert len(zones) == 2

    def test_representative_price_is_mean(self):
        zones = _compute_interior_confluence_zones([28_000.0, 28_020.0], cluster_radius=30.0)
        assert zones[0]["price"] == pytest.approx(28_010.0)

    def test_sorted_by_count_then_price_desc(self):
        zones = _compute_interior_confluence_zones(
            [28_000.0, 28_005.0, 28_500.0], cluster_radius=20.0
        )
        assert len(zones) >= 2
        assert zones[0]["count"] >= zones[-1]["count"]

    def test_empty_prices_returns_empty(self):
        assert _compute_interior_confluence_zones([]) == []

    def test_interior_confluence_zones_on_extraction_result(self):
        """ExtractionResult default has interior_confluence_zones as empty list."""
        result = ExtractionResult()
        assert result.interior_confluence_zones == []
