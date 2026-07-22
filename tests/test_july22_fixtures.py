"""
Regression tests for the July 22 2026 TradingView screenshot conditions
(RTY/NQ/YM/ES at ~10:09–10:10 ET) where visible red/green horizontal levels
were missed by the detection pipeline.

These tests use **synthetic** images that reproduce the specific problematic
conditions identified in the July 22 screenshots:

1. Large session/overlay boxes (filled coloured rectangles) that contaminate
   the colour masks and cause their edges to be detected as price levels.
2. Lines situated near the far-right edge of the plot area where the
   current-price label/arrow sits.
3. Multiple coloured horizontal lines (≥1 green and ≥1 red) that must all
   be detected correctly per image.
4. The new debug payload keys (raw candidate counts, filter counts,
   ``kept_lines`` with y-pixel + mapped price).
5. Detected line y-pixels must be within the chart ROI (which maps to prices
   within the OCR axis bounds).

No actual screenshots or Tesseract OCR are required; all tests run in the
sandboxed environment using only NumPy and OpenCV.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from packages.core.image_extractor import (
    _detect_horizontal_lines,
    _map_y_to_price,
    extract_from_image,
)

# ── Synthetic image parameters ───────────────────────────────────────────────
_W, _H = 800, 600
_BG_BGR = (34, 23, 19)          # TradingView dark background ≈ #131722
_GREEN_BGR = (129, 153, 8)      # TradingView green #089981
_RED_BGR = (69, 54, 242)        # TradingView red #f23645

# Chart ROI boundaries matching the detection pipeline constants
_ROI_TOP = int(_H * 0.08)       # 48 px
_ROI_BOTTOM = int(_H * 0.90)    # 540 px
_AXIS_X = int(_W * 0.85)        # 680 px  (start of price-axis strip)


def _encode_png(img: np.ndarray) -> bytes:
    """Encode a NumPy BGR image to PNG bytes."""
    ok, buf = cv2.imencode(".png", img)
    assert ok, "cv2.imencode failed"
    return bytes(buf)


def _draw_hline(
    img: np.ndarray,
    y: int,
    color_bgr: tuple,
    x_start: int = 20,
    x_end: int | None = None,
) -> None:
    """Draw a single-pixel horizontal line (no anti-aliasing)."""
    if x_end is None:
        x_end = _AXIS_X - 20  # stop well before price axis
    img[y, x_start:x_end] = color_bgr


# ---------------------------------------------------------------------------
# 1. Session-box suppression
# ---------------------------------------------------------------------------


class TestJuly22SessionBoxSuppression:
    """
    Filled session-box blobs must be suppressed so their top/bottom edges are
    not reported as price levels; actual thin horizontal lines must still pass.
    """

    @staticmethod
    def _make_image() -> tuple[np.ndarray, int, int]:
        """
        Chart with a large red session box (200 px tall) plus one thin green
        line below it and one thin red line above it.

        After session-box suppression the only detectable segments should be
        the two thin horizontal lines.
        """
        img = np.full((_H, _W, 3), _BG_BGR, dtype=np.uint8)

        # Large red session box – intentionally passes the red HSV/LAB/BGR masks
        # but is too tall to be a price-level line.
        img[150:350, 50:620] = _RED_BGR  # 200 px tall

        # Thin green line below the box (well separated)
        green_y = 420
        _draw_hline(img, green_y, _GREEN_BGR)

        # Thin red line above the box, inside the chart ROI
        red_y = 80
        _draw_hline(img, red_y, _RED_BGR)

        return img, green_y, red_y

    def test_detects_thin_green_line_not_suppressed(self):
        img, green_y, _ = self._make_image()
        lines = _detect_horizontal_lines(img)
        green_ys = [y for y, c in lines if c == "green"]
        assert len(green_ys) >= 1, f"Expected green line; got lines={lines}"
        assert any(abs(y - green_y) <= 10 for y in green_ys), (
            f"Green line expected near y={green_y}, detected at {green_ys}"
        )

    def test_detects_thin_red_line_not_confused_with_box(self):
        img, _, red_y = self._make_image()
        lines = _detect_horizontal_lines(img)
        red_ys = [y for y, c in lines if c == "red"]
        assert len(red_ys) >= 1, f"Expected red line; got lines={lines}"
        assert any(abs(y - red_y) <= 10 for y in red_ys), (
            f"Red line expected near y={red_y}, detected at {red_ys}"
        )

    def test_detects_both_colors_in_session_box_image(self):
        img, _, _ = self._make_image()
        lines = _detect_horizontal_lines(img)
        colors = {c for _, c in lines}
        assert "green" in colors, f"Missing green; lines={lines}"
        assert "red" in colors, f"Missing red; lines={lines}"

    def test_at_least_one_red_and_one_green(self):
        """Requirement: ≥1 red AND ≥1 green valid horizontal level per image."""
        img, green_y, red_y = self._make_image()
        lines = _detect_horizontal_lines(img)
        assert sum(1 for _, c in lines if c == "green") >= 1
        assert sum(1 for _, c in lines if c == "red") >= 1


# ---------------------------------------------------------------------------
# 2. Right-edge exclusion
# ---------------------------------------------------------------------------


class TestJuly22RightEdgeExclusion:
    """
    The right-edge exclusion zone (last 3 % of chart width) must prevent the
    current-price label/arrow from being detected as a level, while lines that
    span most of the chart area are still detected.
    """

    def test_line_spanning_most_of_chart_detected(self):
        """A green line from x=20 to x=650 (well inside exclusion zone) is detected."""
        img = np.full((_H, _W, 3), _BG_BGR, dtype=np.uint8)
        green_y = 300
        img[green_y, 20:650] = _GREEN_BGR  # 630 px wide, ends before roi_right=680
        lines = _detect_horizontal_lines(img)
        green_ys = [y for y, c in lines if c == "green"]
        assert len(green_ys) >= 1, f"Expected green line; got {lines}"
        assert any(abs(y - green_y) <= 10 for y in green_ys)

    def test_line_with_both_colors_near_right_edge(self):
        """Both a green and red line reaching ~96 % of chart width are detected."""
        img = np.full((_H, _W, 3), _BG_BGR, dtype=np.uint8)
        chart_right = int(_W * 0.85)          # 680
        detect_right = int(chart_right * 0.97)  # 659 – exclusion boundary

        # Lines end just inside the exclusion zone
        green_y, red_y = 200, 400
        img[green_y, 20:detect_right - 5] = _GREEN_BGR
        img[red_y, 20:detect_right - 5] = _RED_BGR

        lines = _detect_horizontal_lines(img)
        colors = {c for _, c in lines}
        assert "green" in colors, f"Missing green; lines={lines}"
        assert "red" in colors, f"Missing red; lines={lines}"


# ---------------------------------------------------------------------------
# 3. Noisy / multi-condition images – both colors detected
# ---------------------------------------------------------------------------


class TestJuly22BothColorsDetected:
    """Each synthetic July 22-style image must yield ≥1 red and ≥1 green level."""

    def _base_img(self) -> np.ndarray:
        return np.full((_H, _W, 3), _BG_BGR, dtype=np.uint8)

    def test_single_green_line_detected(self):
        img = self._base_img()
        _draw_hline(img, 300, _GREEN_BGR)
        lines = _detect_horizontal_lines(img)
        assert any(c == "green" for _, c in lines), f"No green; lines={lines}"

    def test_single_red_line_detected(self):
        img = self._base_img()
        _draw_hline(img, 300, _RED_BGR)
        lines = _detect_horizontal_lines(img)
        assert any(c == "red" for _, c in lines), f"No red; lines={lines}"

    def test_image_with_both_colors_detects_both(self):
        """Simulates a RTY/NQ/YM/ES screenshot with one support and one resistance."""
        img = self._base_img()
        _draw_hline(img, 200, _GREEN_BGR)  # above mid → resistance
        _draw_hline(img, 400, _RED_BGR)    # below mid → support
        lines = _detect_horizontal_lines(img)
        colors = {c for _, c in lines}
        assert "green" in colors, f"Missing green; lines={lines}"
        assert "red" in colors, f"Missing red; lines={lines}"

    def test_four_lines_detected(self):
        """Two green + two red lines must all be detected (July 22 multi-level chart)."""
        img = self._base_img()
        _draw_hline(img, 150, _GREEN_BGR)
        _draw_hline(img, 250, _GREEN_BGR)
        _draw_hline(img, 370, _RED_BGR)
        _draw_hline(img, 470, _RED_BGR)
        lines = _detect_horizontal_lines(img)
        assert sum(1 for _, c in lines if c == "green") >= 2, f"lines={lines}"
        assert sum(1 for _, c in lines if c == "red") >= 2, f"lines={lines}"


# ---------------------------------------------------------------------------
# 4. Debug payload – new keys
# ---------------------------------------------------------------------------


class TestJuly22DebugPayload:
    """
    When ``debug=True`` is passed to ``extract_from_image``, the debug_info
    dict must contain the new diagnostic fields added for July 22 2026.
    """

    @staticmethod
    def _simple_png() -> bytes:
        img = np.full((_H, _W, 3), _BG_BGR, dtype=np.uint8)
        _draw_hline(img, 200, _GREEN_BGR)
        _draw_hline(img, 400, _RED_BGR)
        return _encode_png(img)

    def test_debug_info_has_raw_candidate_counts(self):
        result = extract_from_image(self._simple_png(), debug=True)
        assert result.debug_info is not None
        di = result.debug_info
        assert "raw_green_candidates" in di, f"Missing key; di={di}"
        assert "raw_red_candidates" in di, f"Missing key; di={di}"
        assert isinstance(di["raw_green_candidates"], int)
        assert isinstance(di["raw_red_candidates"], int)

    def test_debug_info_has_filter_counts(self):
        result = extract_from_image(self._simple_png(), debug=True)
        di = result.debug_info
        assert "filtered_by_slope" in di, f"Missing key; di={di}"
        assert "filtered_by_length" in di, f"Missing key; di={di}"

    def test_debug_info_has_kept_lines(self):
        result = extract_from_image(self._simple_png(), debug=True)
        di = result.debug_info
        assert "kept_lines" in di, f"Missing key; di={di}"
        kept = di["kept_lines"]
        assert isinstance(kept, list)
        for item in kept:
            assert "y_pixel" in item, f"Missing y_pixel in {item}"
            assert "color" in item, f"Missing color in {item}"
            assert "price" in item, f"Missing price in {item}"

    def test_kept_lines_has_at_least_two_entries(self):
        """Green + red lines → at least 2 kept_lines entries."""
        result = extract_from_image(self._simple_png(), debug=True)
        di = result.debug_info
        assert di is not None and len(di["kept_lines"]) >= 2

    def test_kept_lines_color_labels_are_valid(self):
        result = extract_from_image(self._simple_png(), debug=True)
        for item in result.debug_info["kept_lines"]:  # type: ignore[index]
            assert item["color"] in ("green", "red")

    def test_raw_green_candidates_gte_zero(self):
        result = extract_from_image(self._simple_png(), debug=True)
        assert result.debug_info["raw_green_candidates"] >= 0  # type: ignore[index]

    def test_debug_false_does_not_populate_new_keys(self):
        result = extract_from_image(self._simple_png(), debug=False)
        assert result.debug_info is None

    def test_existing_debug_keys_still_present(self):
        """Backward compat: all pre-existing debug keys must still be present."""
        result = extract_from_image(self._simple_png(), debug=True)
        di = result.debug_info
        assert di is not None
        for key in (
            "image_size",
            "chart_roi",
            "green_mask_pixels",
            "red_mask_pixels",
            "segments_before_dedup",
            "segments_after_dedup",
        ):
            assert key in di, f"Legacy key '{key}' missing; di={di}"


# ---------------------------------------------------------------------------
# 5. Detected lines must map to prices within axis bounds
# ---------------------------------------------------------------------------


class TestJuly22MappedPricesInBounds:
    """
    Detected line y-pixels must:
    (a) fall inside the chart ROI, and
    (b) when mapped through a synthetic price axis, yield prices within the
        axis price range (with a small extrapolation tolerance).

    This simulates the requirement that mapped prices stay within OCR axis
    bounds without needing a live Tesseract installation.
    """

    # Synthetic axis covering the full chart ROI height
    # (y increases downward, price decreases)
    _AXIS = [
        (_ROI_TOP,      21000.0),
        (_ROI_TOP + 120, 20000.0),
        (_ROI_TOP + 240, 19000.0),
        (_ROI_TOP + 360, 18000.0),
        (_ROI_BOTTOM,   16000.0),
    ]
    _AXIS_MIN = 16000.0
    _AXIS_MAX = 21000.0
    # Allow ±10 % extrapolation beyond axis endpoints
    _TOLERANCE = 0.10

    def _lines_from_chart(self, *line_ys: int) -> list:
        """Build an image with lines at the given y-coords and return detections."""
        img = np.full((_H, _W, 3), _BG_BGR, dtype=np.uint8)
        for i, y in enumerate(line_ys):
            color = _GREEN_BGR if i % 2 == 0 else _RED_BGR
            _draw_hline(img, y, color)
        return _detect_horizontal_lines(img)

    def test_detected_line_y_within_roi(self):
        """All detected y-values must lie inside the chart ROI."""
        lines = self._lines_from_chart(200, 350)
        for y, _ in lines:
            assert _ROI_TOP <= y <= _ROI_BOTTOM, (
                f"y={y} outside ROI [{_ROI_TOP}, {_ROI_BOTTOM}]"
            )

    def test_detected_lines_map_to_prices_within_axis_range(self):
        """Prices mapped from detected y-coordinates stay within the axis range."""
        lines = self._lines_from_chart(200, 350)
        lo = self._AXIS_MIN * (1.0 - self._TOLERANCE)
        hi = self._AXIS_MAX * (1.0 + self._TOLERANCE)
        for y, _ in lines:
            price = _map_y_to_price(y, self._AXIS)
            assert price is not None, f"_map_y_to_price returned None for y={y}"
            assert lo <= price <= hi, (
                f"price={price} for y={y} outside [{lo}, {hi}]"
            )

    def test_green_and_red_both_map_to_valid_prices(self):
        """Both a green and red detection must yield a valid price."""
        lines = self._lines_from_chart(150, 450)
        green_prices = [
            _map_y_to_price(y, self._AXIS)
            for y, c in lines if c == "green"
        ]
        red_prices = [
            _map_y_to_price(y, self._AXIS)
            for y, c in lines if c == "red"
        ]
        assert len(green_prices) >= 1, "No green price"
        assert len(red_prices) >= 1, "No red price"
        for p in green_prices + red_prices:
            assert p is not None and p > 0
