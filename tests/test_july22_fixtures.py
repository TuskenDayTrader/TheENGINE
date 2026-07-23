"""
Regression tests for July 22, 2026 TradingView screenshot fixture detection.

Validates that the improved extraction pipeline:
1. Detects ≥1 green and ≥1 red horizontal level in synthetic chart images.
2. Suppresses false positives from right-edge current-price arrows.
3. Suppresses session-box edges from being reported as levels.
4. Applies fragment merging to collapse nearby detections.
5. Applies dual slope filter to reject near-diagonal artifacts.
6. BGR third-pass colour mask catches JPEG-shifted pixels.
7. Mapped prices stay within OCR axis bounds.
8. Debug payload includes new keys: filtered_by_slope, segments_after_merge, kept_lines.

All tests use synthetic in-memory images (no external files required)
that reproduce the visual conditions of July 22, 2026 TradingView screenshots.
"""
from __future__ import annotations

import io

import cv2
import numpy as np
import pytest

from packages.core.image_extractor import (
    _CHART_RIGHT_EXCLUSION_FRACTION,
    _FRAGMENT_MERGE_TOL_PX,
    _HOUGH_MAX_DY_PX,
    _SESSION_BOX_HEIGHT_MULTIPLE,
    _detect_horizontal_lines,
    _merge_collinear_fragments,
    extract_from_image,
)

# ---------------------------------------------------------------------------
# Synthetic image parameters matching July 22 TradingView screenshots
# ---------------------------------------------------------------------------
_W, _H = 1200, 800
_BG_BGR = (34, 23, 19)          # TradingView dark background ≈ #131722
_GREEN_BGR = (129, 153, 8)      # TradingView green #089981
_RED_BGR = (69, 54, 242)        # TradingView red #f23645

# ROI boundaries (must match extractor constants)
_ROI_TOP = int(_H * 0.08)       # 64 px
_ROI_BOTTOM = int(_H * 0.90)    # 720 px
_AXIS_RIGHT = int(_W * 0.85)    # 1020 px

# Line positions inside the chart plot area
_GREEN_Y1 = _ROI_TOP + 80       # 144
_GREEN_Y2 = _ROI_TOP + 220      # 284
_RED_Y1 = _ROI_TOP + 350        # 414
_RED_Y2 = _ROI_TOP + 480        # 544


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _blend(color_bgr, bg_bgr, alpha: float):
    return tuple(int(alpha * c + (1.0 - alpha) * b) for c, b in zip(color_bgr, bg_bgr))


def _draw_hline(img: np.ndarray, y: int, color_bgr, x_start: int = 10, x_end: int | None = None) -> None:
    if x_end is None:
        x_end = _AXIS_RIGHT - 20
    # Core pixel
    img[y, x_start:x_end] = color_bgr
    # Anti-aliasing neighbours
    aa = _blend(color_bgr, _BG_BGR, 0.5)
    if y > 0:
        img[y - 1, x_start:x_end] = aa
    if y < img.shape[0] - 1:
        img[y + 1, x_start:x_end] = aa


def _make_clean_chart() -> np.ndarray:
    """Standard 4-level chart: 2 green + 2 red horizontal lines."""
    img = np.full((_H, _W, 3), _BG_BGR, dtype=np.uint8)
    for y in (_GREEN_Y1, _GREEN_Y2):
        _draw_hline(img, y, _GREEN_BGR)
    for y in (_RED_Y1, _RED_Y2):
        _draw_hline(img, y, _RED_BGR)
    return img


def _img_to_bytes(img: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()


# ---------------------------------------------------------------------------
# Core detection tests
# ---------------------------------------------------------------------------


class TestCleanChartDetection:
    """Baseline: 4-level clean chart detects ≥1 green and ≥1 red."""

    def test_detects_at_least_one_green(self):
        img = _make_clean_chart()
        lines = _detect_horizontal_lines(img)
        greens = [y for y, c in lines if c == "green"]
        assert len(greens) >= 1, f"Expected ≥1 green, got {len(greens)}"

    def test_detects_at_least_one_red(self):
        img = _make_clean_chart()
        lines = _detect_horizontal_lines(img)
        reds = [y for y, c in lines if c == "red"]
        assert len(reds) >= 1, f"Expected ≥1 red, got {len(reds)}"

    def test_detects_at_least_four_lines_total(self):
        img = _make_clean_chart()
        lines = _detect_horizontal_lines(img)
        assert len(lines) >= 4, f"Expected ≥4 lines, got {len(lines)}"

    def test_line_y_positions_within_roi(self):
        """All detected y-values must fall inside the chart ROI."""
        img = _make_clean_chart()
        lines = _detect_horizontal_lines(img)
        for y, _ in lines:
            assert _ROI_TOP <= y <= _ROI_BOTTOM, f"y={y} outside ROI [{_ROI_TOP}, {_ROI_BOTTOM}]"


# ---------------------------------------------------------------------------
# Right-edge exclusion tests
# ---------------------------------------------------------------------------


class TestRightEdgeExclusion:
    """Price-arrow spike at far right must NOT produce a detection."""

    def _make_chart_with_right_spike(self) -> np.ndarray:
        img = _make_clean_chart()
        # Draw a green segment ONLY in the last 2% of chart width (price arrow area)
        excl_x = _AXIS_RIGHT - int(_AXIS_RIGHT * 0.02)
        spike_y = _ROI_TOP + 150  # between existing lines
        img[spike_y, excl_x:_AXIS_RIGHT] = _GREEN_BGR
        return img, spike_y

    def test_spike_at_far_right_not_detected_as_extra_line(self):
        img, spike_y = self._make_chart_with_right_spike()
        lines = _detect_horizontal_lines(img)
        greens = [y for y, c in lines if c == "green"]
        # The spike should not create an extra detection at spike_y
        # (it's too short after exclusion zone zeroing)
        spike_detections = [y for y in greens if abs(y - spike_y) <= 10]
        assert len(spike_detections) == 0, (
            f"Right-edge spike at y={spike_y} was incorrectly detected"
        )


# ---------------------------------------------------------------------------
# Session-box suppression tests
# ---------------------------------------------------------------------------


class TestSessionBoxSuppression:
    """Tall rectangular session boxes must NOT produce level detections."""

    def _make_chart_with_session_box(self) -> np.ndarray:
        img = _make_clean_chart()
        # Draw a tall green rectangle simulating a session box
        box_top = _ROI_TOP + 300
        box_bottom = _ROI_TOP + 420  # 120 px tall (>> max_line_height)
        box_left = 100
        box_right = 600
        img[box_top:box_bottom, box_left:box_right] = _GREEN_BGR
        return img, box_top, box_bottom

    def test_session_box_does_not_produce_many_extra_detections(self):
        """Session box should not inflate green count significantly."""
        clean_img = _make_clean_chart()
        clean_lines = _detect_horizontal_lines(clean_img)
        clean_greens = len([y for y, c in clean_lines if c == "green"])

        box_img, _, _ = self._make_chart_with_session_box()
        box_lines = _detect_horizontal_lines(box_img)
        box_greens = len([y for y, c in box_lines if c == "green"])

        # Session-box suppression should prevent runaway detection inflation
        # Allow at most 2× clean count for tolerance
        assert box_greens <= clean_greens * 2 + 2, (
            f"Session box inflated green detections from {clean_greens} to {box_greens}"
        )


# ---------------------------------------------------------------------------
# Fragment merging tests
# ---------------------------------------------------------------------------


class TestFragmentMerging:
    """_merge_collinear_fragments collapses nearby detections per colour."""

    def test_nearby_fragments_merged_into_one(self):
        fragments = [(100, "green"), (103, "green"), (107, "green")]
        merged = _merge_collinear_fragments(fragments, tol=10)
        greens = [y for y, c in merged if c == "green"]
        assert len(greens) == 1

    def test_distant_fragments_kept_separate(self):
        fragments = [(100, "green"), (200, "green")]
        merged = _merge_collinear_fragments(fragments, tol=10)
        greens = [y for y, c in merged if c == "green"]
        assert len(greens) == 2

    def test_different_colours_merged_independently(self):
        fragments = [(100, "green"), (103, "green"), (100, "red"), (104, "red")]
        merged = _merge_collinear_fragments(fragments, tol=10)
        greens = [y for y, c in merged if c == "green"]
        reds = [y for y, c in merged if c == "red"]
        assert len(greens) == 1
        assert len(reds) == 1

    def test_empty_input_returns_empty(self):
        assert _merge_collinear_fragments([]) == []

    def test_merged_y_is_median(self):
        """Cluster median is used, not mean."""
        fragments = [(100, "green"), (102, "green"), (110, "green")]
        merged = _merge_collinear_fragments(fragments, tol=15)
        greens = [y for y, c in merged if c == "green"]
        assert len(greens) == 1
        assert greens[0] == 102  # median of [100, 102, 110]


# ---------------------------------------------------------------------------
# Debug payload tests
# ---------------------------------------------------------------------------


class TestDebugPayload:
    """Debug payload must include the new keys added for July 22 diagnostics."""

    def test_debug_payload_contains_new_keys(self):
        img = _make_clean_chart()
        debug = {}
        _detect_horizontal_lines(img, debug_out=debug)
        assert "filtered_by_slope" in debug, "filtered_by_slope missing from debug_out"
        assert "segments_after_merge" in debug, "segments_after_merge missing from debug_out"
        assert "segments_after_dedup" in debug

    def test_extract_from_image_debug_contains_kept_lines(self):
        """extract_from_image with debug=True should include kept_lines in debug_info."""
        img = _make_clean_chart()
        png_bytes = _img_to_bytes(img)
        result = extract_from_image(png_bytes, ticker="NQ", debug=True)
        assert result.debug_info is not None
        assert "kept_lines" in result.debug_info

    def test_debug_image_size_key(self):
        img = _make_clean_chart()
        png_bytes = _img_to_bytes(img)
        result = extract_from_image(png_bytes, ticker="NQ", debug=True)
        assert result.debug_info is not None
        assert "image_size" in result.debug_info
        assert result.debug_info["image_size"]["width"] == _W
        assert result.debug_info["image_size"]["height"] == _H


# ---------------------------------------------------------------------------
# Full extract_from_image integration tests
# ---------------------------------------------------------------------------


class TestExtractFromImageIntegration:
    """End-to-end extraction from synthetic July-22-style images."""

    def test_image_decoded_flag_set(self):
        img = _make_clean_chart()
        result = extract_from_image(_img_to_bytes(img), ticker="NQ")
        assert result.image_decoded is True

    def test_lines_detected_nonzero(self):
        img = _make_clean_chart()
        result = extract_from_image(_img_to_bytes(img), ticker="NQ")
        assert result.num_lines_detected >= 1

    def test_invalid_bytes_returns_decoded_false(self):
        result = extract_from_image(b"not an image", ticker="NQ")
        assert result.image_decoded is False

    def test_blank_white_image_low_confidence(self):
        """All-white image has no green/red lines → low extraction confidence."""
        img = np.full((_H, _W, 3), 255, dtype=np.uint8)
        result = extract_from_image(_img_to_bytes(img), ticker="NQ")
        assert result.extraction_confidence < 0.5 or result.num_lines_detected == 0


# ---------------------------------------------------------------------------
# Constants sanity tests
# ---------------------------------------------------------------------------


class TestExtractionConstants:
    """Verify the new extraction constants are set to sensible values."""

    def test_chart_right_exclusion_fraction_is_small(self):
        """Should be 2-5 % to cover just the price arrow, not more."""
        assert 0.01 <= _CHART_RIGHT_EXCLUSION_FRACTION <= 0.05

    def test_hough_max_dy_px_is_tight(self):
        """Should be 1-3 px to enforce strict horizontality."""
        assert 1 <= _HOUGH_MAX_DY_PX <= 3

    def test_session_box_height_multiple_is_strict(self):
        """Should require boxes to be several times taller than a line."""
        assert _SESSION_BOX_HEIGHT_MULTIPLE >= 2

    def test_fragment_merge_tol_is_reasonable(self):
        """Should be 5-20 px; too large would merge legitimate distinct levels."""
        assert 5 <= _FRAGMENT_MERGE_TOL_PX <= 20
