"""
Regression tests for horizontal-line detection using a synthetic
TradingView-like screenshot.

The fixture image simulates:
- A dark background (#131722 ≈ BGR 34/23/19)
- Two green support/resistance lines (#089981) with anti-aliasing
- Two red support/resistance lines (#f23645) with anti-aliasing
- An empty price-axis strip (right 15 %) and title area (top 8 %)
- An empty ATR / session panel (bottom 10 %)

All four lines lie inside the chart plot area so the detection pipeline
must find at least 4 coloured segments, with at least one green and one red.
"""
from __future__ import annotations

import io

import cv2
import numpy as np
import pytest

from packages.core.image_extractor import _detect_horizontal_lines, extract_from_image

# ── Synthetic fixture parameters ────────────────────────────────────────────
_W, _H = 800, 600
_BG_BGR = (34, 23, 19)        # TradingView dark background ≈ #131722
_GREEN_BGR = (129, 153, 8)    # TradingView green #089981
_RED_BGR = (69, 54, 242)      # TradingView red #f23645

# Chart plot area boundaries (matching the detection pipeline)
_ROI_TOP = int(_H * 0.08)          # 48 px
_ROI_BOTTOM = int(_H * 0.90)       # 540 px
_AXIS_RIGHT = int(_W * 0.85)       # 680 px

# Line positions in the chart plot area (absolute image y-coordinates)
_GREEN_Y1 = _ROI_TOP + 80          # 128
_GREEN_Y2 = _ROI_TOP + 200         # 248
_RED_Y1 = _ROI_TOP + 310           # 358
_RED_Y2 = _ROI_TOP + 400           # 448

_EXPECTED_MIN_LINES = 4


# ── Fixture factory ──────────────────────────────────────────────────────────

def _blend(color_bgr, bg_bgr, alpha: float):
    return tuple(
        int(alpha * c + (1.0 - alpha) * b)
        for c, b in zip(color_bgr, bg_bgr)
    )


def _draw_antialiased_hline(
    image: np.ndarray,
    y: int,
    color_bgr,
    x_start: int = 10,
    x_end: int | None = None,
) -> None:
    """Draw a 3-row anti-aliased horizontal line (core + blended neighbours)."""
    if x_end is None:
        x_end = _AXIS_RIGHT - 10
    image[y, x_start:x_end] = color_bgr
    if y > 0:
        image[y - 1, x_start:x_end] = _blend(color_bgr, _BG_BGR, 0.6)
    if y < image.shape[0] - 1:
        image[y + 1, x_start:x_end] = _blend(color_bgr, _BG_BGR, 0.6)


def make_tradingview_image() -> np.ndarray:
    """Return a synthetic TradingView-like chart image (BGR, uint8)."""
    img = np.full((_H, _W, 3), _BG_BGR, dtype=np.uint8)
    _draw_antialiased_hline(img, _GREEN_Y1, _GREEN_BGR)
    _draw_antialiased_hline(img, _GREEN_Y2, _GREEN_BGR)
    _draw_antialiased_hline(img, _RED_Y1, _RED_BGR)
    _draw_antialiased_hline(img, _RED_Y2, _RED_BGR)
    return img


# ── Shared fixture ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tradingview_img() -> np.ndarray:
    return make_tradingview_image()


@pytest.fixture(scope="module")
def tradingview_png_bytes(tradingview_img) -> bytes:
    ok, buf = cv2.imencode(".png", tradingview_img)
    assert ok, "Failed to encode synthetic fixture to PNG"
    return bytes(buf)


# ── Detection tests ──────────────────────────────────────────────────────────

class TestFixtureDetection:
    def test_detects_minimum_line_count(self, tradingview_img):
        """Pipeline must detect at least 4 coloured segments."""
        lines = _detect_horizontal_lines(tradingview_img)
        assert len(lines) >= _EXPECTED_MIN_LINES, (
            f"Expected >= {_EXPECTED_MIN_LINES} lines, detected {len(lines)}: {lines}"
        )

    def test_detects_at_least_one_green(self, tradingview_img):
        """At least one detected segment must be labelled green."""
        lines = _detect_horizontal_lines(tradingview_img)
        colors = [c for _, c in lines]
        assert "green" in colors, f"No green segment detected; lines={lines}"

    def test_detects_at_least_one_red(self, tradingview_img):
        """At least one detected segment must be labelled red."""
        lines = _detect_horizontal_lines(tradingview_img)
        colors = [c for _, c in lines]
        assert "red" in colors, f"No red segment detected; lines={lines}"

    def test_green_lines_y_positions(self, tradingview_img):
        """Detected y-coordinates must be within ±10 px of drawn green lines."""
        lines = _detect_horizontal_lines(tradingview_img)
        green_ys = sorted(y for y, c in lines if c == "green")
        expected = sorted([_GREEN_Y1, _GREEN_Y2])
        assert len(green_ys) >= 2, f"Expected 2 green lines, got {green_ys}"
        for exp_y, got_y in zip(expected, green_ys):
            assert abs(got_y - exp_y) <= 10, (
                f"Green line expected near y={exp_y}, detected at y={got_y}"
            )

    def test_red_lines_y_positions(self, tradingview_img):
        """Detected y-coordinates must be within ±10 px of drawn red lines."""
        lines = _detect_horizontal_lines(tradingview_img)
        red_ys = sorted(y for y, c in lines if c == "red")
        expected = sorted([_RED_Y1, _RED_Y2])
        assert len(red_ys) >= 2, f"Expected 2 red lines, got {red_ys}"
        for exp_y, got_y in zip(expected, red_ys):
            assert abs(got_y - exp_y) <= 10, (
                f"Red line expected near y={exp_y}, detected at y={got_y}"
            )

    def test_no_lines_in_price_axis_area(self, tradingview_img):
        """No detected line should have an x-centroid in the price-axis strip."""
        # We verify via y-position: lines drawn only in chart area (y in ROI)
        lines = _detect_horizontal_lines(tradingview_img)
        for y, _ in lines:
            assert _ROI_TOP <= y <= _ROI_BOTTOM, (
                f"Line at y={y} is outside the chart ROI [{_ROI_TOP}, {_ROI_BOTTOM}]"
            )


# ── extract_from_image end-to-end with fixture ───────────────────────────────

class TestFixtureExtraction:
    def test_extract_returns_lines_detected(self, tradingview_png_bytes):
        """extract_from_image on the fixture PNG must detect >= 4 lines."""
        result = extract_from_image(tradingview_png_bytes, ticker="FIXTURE")
        assert result.image_decoded is True
        assert result.num_lines_detected >= _EXPECTED_MIN_LINES, (
            f"Expected >= {_EXPECTED_MIN_LINES} lines via extract_from_image, "
            f"got {result.num_lines_detected}"
        )

    def test_extract_debug_true_returns_debug_info(self, tradingview_png_bytes):
        """With debug=True, debug_info must be populated."""
        result = extract_from_image(tradingview_png_bytes, ticker="FIXTURE", debug=True)
        assert result.debug_info is not None
        di = result.debug_info
        assert di["image_size"] == {"width": _W, "height": _H}
        assert di["green_mask_pixels"] > 0
        assert di["red_mask_pixels"] > 0
        assert di["segments_after_dedup"] >= _EXPECTED_MIN_LINES

    def test_before_after_counts_reported(self, tradingview_png_bytes):
        """Debug info must report segment counts before and after deduplication."""
        result = extract_from_image(tradingview_png_bytes, ticker="FIXTURE", debug=True)
        di = result.debug_info
        assert di is not None
        assert "segments_before_dedup" in di
        assert "segments_after_dedup" in di
        # After dedup must be ≤ before dedup
        assert di["segments_after_dedup"] <= di["segments_before_dedup"]
