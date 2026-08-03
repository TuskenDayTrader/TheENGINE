"""
Image extraction module for TheENGINE (MVP – TradingView dark theme).

Detects horizontal support/resistance lines and parses price-axis labels
from TradingView dark-theme screenshots to produce a ``LevelsPayload``
suitable for the scoring engine.

Public API
----------
- ``extract_from_image(image_bytes, ...)`` – main entry point.
- ``ExtractionResult`` – structured output dataclass.

Internal helpers (exported for unit testing)
---------------------------------------------
- ``_detect_horizontal_lines(img)``
- ``_parse_price_axis(img)``
- ``_extract_chart_labels(img)``
- ``_match_labels_to_lines(labels, lines, axis_points)``
- ``_extract_session_info(img)``
- ``_map_y_to_price(y, axis_points)``
- ``_deduplicate_lines(lines, tol)``
- ``_build_levels_payload(line_prices, current_price, labeled_levels)``
- ``_compute_confidence(...)``

Known limitations (MVP)
-----------------------
- Optimised for TradingView dark-theme screenshots only.
- Line-colour detection targets TradingView default green (#089981) and
  red (#f23645).  Custom colours may not be detected.
- Price-axis OCR requires ``tesseract-ocr`` to be installed on the host.
- ``current_price`` is estimated; it is *not* read from user input.
- Chart labels are read from the right-margin annotation zone; coverage
  depends on the TradingView indicator/study layout used.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import cv2
    import numpy as np

    _CV2_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CV2_AVAILABLE = False
    cv2 = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]

try:
    import pytesseract
    from pytesseract import TesseractNotFoundError  # type: ignore[attr-defined]

    _TESSERACT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TESSERACT_AVAILABLE = False
    pytesseract = None  # type: ignore[assignment]
    TesseractNotFoundError = Exception  # type: ignore[assignment,misc]

from .models import LevelsPayload

# ---------------------------------------------------------------------------
# TradingView dark-theme HSV colour ranges (OpenCV scale: H 0-179, S/V 0-255)
# Ranges are intentionally wide to capture anti-aliased line variants.
# ---------------------------------------------------------------------------

# Green #089981 → BGR(129, 153, 8) → HSV ≈ (85, 242, 153)
# Range widened (H 50-100, S/V ≥ 20) to tolerate display-calibration shifts
# and custom ORB indicator colours across the yellow-green→cyan-green spectrum.
_GREEN_LOWER: Tuple[int, int, int] = (50, 20, 20)
_GREEN_UPPER: Tuple[int, int, int] = (100, 255, 255)

# BGR pixel value used for LAB-space second-pass detection
_GREEN_BGR_TARGET: Tuple[int, int, int] = (129, 153, 8)

# Red #f23645 → BGR(69, 54, 242) → HSV ≈ (178, 200, 242) and wraps near H=0
# Ranges widened to H 0-25 / 150-180 and S/V ≥ 20/30 for the same reason.
_RED_LOWER_A: Tuple[int, int, int] = (0, 20, 30)
_RED_UPPER_A: Tuple[int, int, int] = (25, 255, 255)
_RED_LOWER_B: Tuple[int, int, int] = (150, 20, 30)
_RED_UPPER_B: Tuple[int, int, int] = (180, 255, 255)

# BGR pixel value used for LAB-space second-pass detection
_RED_BGR_TARGET: Tuple[int, int, int] = (69, 54, 242)

# LAB-distance threshold for second-pass near-neon colour detection.
# Covers anti-aliased pixels blended up to ~80 % with a dark background.
_LAB_DISTANCE_THRESHOLD: float = 50.0

# Minimum fraction of CHART width (plot area only) a segment must span.
# 3 % catches short recent levels while still rejecting stray pixels.
_LINE_MIN_WIDTH_FRACTION: float = 0.03

# Rightmost fraction of the image treated as the price axis
_PRICE_AXIS_FRACTION: float = 0.15

# Chart ROI fractions – areas excluded from line detection:
_ROI_TOP_FRACTION: float = 0.08     # top 8 %: watermark / title area
_ROI_BOTTOM_FRACTION: float = 0.90  # keep up to 90 %; bottom 10 %: ATR / session panel

# Hough near-horizontal tolerance in degrees
_HOUGH_ANGLE_MAX_DEG: float = 5.0

# ---------------------------------------------------------------------------
# July 22 2026 TradingView detection-improvement constants
# ---------------------------------------------------------------------------

# Rightmost fraction of CHART AREA (plot area) excluded from line candidate
# detection.  Avoids the current-price label/arrow artefact that sits at the
# far-right edge of the plot.  Axis OCR logic (which uses the rightmost 15 %
# of the full image) is deliberately kept unchanged.
_CHART_RIGHT_EXCLUSION_FRACTION: float = 0.03  # last 3 %

# Minimum width/height aspect ratio for contour bounding boxes.  Prevents
# wide-but-tall session-box blobs from being misclassified as horizontal lines.
_CONTOUR_MIN_ASPECT: float = 3.0

# Maximum absolute |dy| in pixels for a Hough segment to pass the slope
# filter.  Complements the angular tolerance for short or diagonal segments.
_HOUGH_MAX_SLOPE_DY_PX: int = 2

# BGR-space Euclidean distance threshold for the RGB-distance colour fallback.
# Covers neon green/red pixels that survive JPEG compression artefacts or
# colour-profile shifts outside the HSV and LAB detection windows.
_BGR_DISTANCE_THRESHOLD: float = 60.0

# A mask blob whose height exceeds this multiple of ``max_line_height`` is
# classified as a filled session/overlay rectangle and suppressed before
# Hough detection so its top/bottom edges are not reported as price levels.
_BOX_HEIGHT_FACTOR: int = 3

# Synthetic level offsets when no extracted levels exist on one side
_SYNTHETIC_RESISTANCE_OFFSET: float = 1.001
_SYNTHETIC_SUPPORT_OFFSET: float = 0.999

# ---------------------------------------------------------------------------
# Confidence weights (each component contributes equally)
# ---------------------------------------------------------------------------
CONFIDENCE_AXIS_POINTS: float = 0.25
CONFIDENCE_LINES_DETECTED: float = 0.25
CONFIDENCE_CURRENT_PRICE: float = 0.25
CONFIDENCE_PRICES_MAPPED: float = 0.25

# Confidence below this threshold → extraction_warning is set
CONFIDENCE_LOW_THRESHOLD: float = 0.50

# Minimum axis points required to establish a reliable price scale
_MIN_AXIS_POINTS_FOR_SCALE: int = 3

# ---------------------------------------------------------------------------
# Label extraction constants
# ---------------------------------------------------------------------------

# The chart-label zone is in the right portion of the chart image, between
# the plot area right edge and the numeric price axis strip.
# These fractions define the x-range for label OCR (relative to full image width).
# Right boundary extended to 0.93 so that far-right margin annotations
# (e.g. "Asia High", "New York Low") placed near the price-axis edge are captured.
_LABEL_ZONE_LEFT_FRACTION: float = 0.60   # start of label zone
_LABEL_ZONE_RIGHT_FRACTION: float = 0.93  # end of label zone (captures far-right labels)

# Binary threshold used for label OCR.  Lowered from 140 to 100 so that
# TradingView's dim/gray annotation text (value ~110-160) is also detected,
# not only bright-white labels.
_LABEL_BINARY_THRESHOLD: int = 100

# Maximum y-pixel distance between a chart label and its nearest horizontal
# line for the two to be associated.
_LABEL_LINE_MATCH_TOLERANCE_PX: int = 20

# Session-info panel is in the bottom-right corner.
_SESSION_PANEL_TOP_FRACTION: float = 0.82
_SESSION_PANEL_LEFT_FRACTION: float = 0.72

# ---------------------------------------------------------------------------
# Mechanical outlier-rejection constants
# ---------------------------------------------------------------------------

# Tier-2: reject any mapped price whose distance from current_price exceeds
# this multiple of the detected ATR.  A 5× ATR envelope is deliberately wide
# to avoid rejecting genuine far-away structural levels while still catching
# wild OCR mis-reads (e.g. 16 072 vs 28 072).
_OUTLIER_ATR_ENVELOPE: float = 5.0

# Tier-3: statistical IQR multiplier.  With a typical set of 4-10 chart
# levels, 2.5× IQR is tight enough to reject isolated garbage values but
# permissive enough to keep spread-out structural levels.
_OUTLIER_IQR_MULTIPLIER: float = 2.5

# Minimum number of non-outlier peer prices required before the IQR filter
# is applied.  With fewer peers the IQR is unreliable.
_OUTLIER_IQR_MIN_PEERS: int = 3

# Canonical label → LevelsPayload field mapping.
# Normalised label text (lowercase, stripped) → field name.
# Compound labels like "Prev Day High / London High" are split on "/" and
# each part is tried independently; the first match wins.
_LABEL_FIELD_MAP: Dict[str, str] = {
    # Previous day
    "prev day high": "pdh",
    "previous day high": "pdh",
    "prior day high": "pdh",
    "prevdayhigh": "pdh",
    "pdh": "pdh",
    "prev day low": "pdl",
    "previous day low": "pdl",
    "prior day low": "pdl",
    "prevdaylow": "pdl",
    "pdl": "pdl",
    # Prior settle / close
    "prior settle": "prior_settle",
    "prev settle": "prior_settle",
    "prior close": "prior_settle",
    "prev close": "prior_settle",
    # Globex session
    "globex high": "globex_high",
    "globex low": "globex_low",
    "globex open": "globex_open",
    # Asian / Tokyo session
    "asia high": "asia_high",
    "asian high": "asia_high",
    "asia low": "asia_low",
    "asian low": "asia_low",
    "asia open": "asia_open",
    "asian open": "asia_open",
    "tokyo open": "asia_open",
    # London session
    "london high": "london_high",
    "london low": "london_low",
    "london open": "london_open",
    # New York session
    "new york high": "ny_high",
    "newyork high": "ny_high",
    "ny high": "ny_high",
    "new york low": "ny_low",
    "newyork low": "ny_low",
    "ny low": "ny_low",
    "new york open": "rth_open",
    "newyork open": "rth_open",
    "ny open": "rth_open",
    # 4H references mapped to session equivalents where possible
    "prev 4h high": "ny_high",
    "prev 4h low": "ny_low",
    # Previous week — map to globex session slots (nearest equivalent)
    "prev week high": "globex_high",
    "previous week high": "globex_high",
    "prior week high": "globex_high",
    "prev week low": "globex_low",
    "previous week low": "globex_low",
    "prior week low": "globex_low",
    # Previous month — map to pdh/pdl (broadest prior reference)
    "prev month high": "pdh",
    "previous month high": "pdh",
    "prior month high": "pdh",
    "prev month low": "pdl",
    "previous month low": "pdl",
    "prior month low": "pdl",
}


def _normalise_label(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation for label matching."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9 /]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _label_to_field(raw_label: str) -> Optional[str]:
    """
    Map a raw OCR label string to a ``LevelsPayload`` field name.

    Handles compound labels separated by ``/`` by trying each part in order.
    Returns ``None`` when no mapping is found.
    """
    normalised = _normalise_label(raw_label)
    # Try the full label first
    if normalised in _LABEL_FIELD_MAP:
        return _LABEL_FIELD_MAP[normalised]
    # Split compound labels on "/"
    for part in normalised.split("/"):
        part = part.strip()
        if part in _LABEL_FIELD_MAP:
            return _LABEL_FIELD_MAP[part]
    return None


# ---------------------------------------------------------------------------
# Public data model
# ---------------------------------------------------------------------------


@dataclass
class ExtractionResult:
    """Structured result returned by :func:`extract_from_image`."""

    #: True when the image was decoded successfully (regardless of whether
    #: any levels were found).
    image_decoded: bool = False

    #: A ready-to-score ``LevelsPayload``; ``None`` when confidence is too low.
    levels_payload: Optional[LevelsPayload] = None

    #: Estimated current price; ``None`` when it could not be determined.
    current_price: Optional[float] = None

    #: Number of coloured horizontal lines found in the chart area.
    num_lines_detected: int = 0

    #: Number of price labels successfully parsed from the axis via OCR.
    num_axis_points: int = 0

    #: Overall extraction quality in [0.0, 1.0].
    extraction_confidence: float = 0.0

    #: Human-readable warning when confidence < :data:`CONFIDENCE_LOW_THRESHOLD`.
    warning: Optional[str] = None

    #: Label-to-price mapping extracted from chart right-margin annotations.
    #: Keys are LevelsPayload field names (e.g. "ny_low"), values are prices.
    #: Empty dict when no labels were readable.
    labeled_levels: Dict[str, float] = field(default_factory=dict)

    #: Session context parsed from the bottom-right panel (e.g. "Globex",
    #: "New York").  ``None`` when the panel could not be read.
    detected_session: Optional[str] = None

    #: ATR value parsed from the session panel.  ``None`` when unavailable.
    detected_atr: Optional[float] = None

    #: Lowest price label parsed from the price axis (bottom of visible range).
    #: Used by the quality-gate axis-bounds check.
    axis_price_min: Optional[float] = None

    #: Highest price label parsed from the price axis (top of visible range).
    #: Used by the quality-gate axis-bounds check.
    axis_price_max: Optional[float] = None

    #: Prices mechanically rejected by :func:`_filter_outlier_prices` together
    #: with a short human-readable reason.  Empty when no outliers were found.
    rejected_outliers: List[Tuple[float, str]] = field(default_factory=list)

    #: Optional diagnostic counters populated when ``debug=True`` is passed to
    #: :func:`extract_from_image`.  Keys: ``image_size``, ``chart_roi``,
    #: ``green_mask_pixels``, ``red_mask_pixels``, ``contour_segments_raw``,
    #: ``hough_segments_raw``, ``raw_green_candidates``, ``raw_red_candidates``,
    #: ``filtered_by_slope``, ``filtered_by_length``,
    #: ``segments_before_dedup``, ``segments_after_dedup``,
    #: ``kept_lines`` (list of dicts with ``y_pixel``, ``color``, ``price``),
    #: ``chart_labels`` (list of dicts with ``label``, ``y``, ``field``,
    #: ``price``).
    debug_info: Optional[dict] = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _deduplicate_lines(
    lines: List[Tuple[int, str]], tol: int = 5
) -> List[Tuple[int, str]]:
    """Remove duplicate lines whose y-positions are within *tol* pixels."""
    if not lines:
        return []
    sorted_lines = sorted(lines, key=lambda x: x[0])
    result: List[Tuple[int, str]] = [sorted_lines[0]]
    for y, color in sorted_lines[1:]:
        if y - result[-1][0] > tol:
            result.append((y, color))
    return result


def _compute_lab_mask(
    chart_lab: "np.ndarray",
    target_bgr: Tuple[int, int, int],
    threshold: float,
) -> "np.ndarray":
    """
    Return a binary mask where each pixel's OpenCV-LAB distance to
    *target_bgr* is ≤ *threshold*.

    Performs a single-channel Euclidean distance test in CIE L*a*b* space
    (OpenCV scale), which is more perceptually uniform than BGR and reliably
    captures anti-aliased variants of the target colour blended with a dark
    background.

    Parameters
    ----------
    chart_lab:
        Image already converted to OpenCV LAB (float32, same shape as source).
    target_bgr:
        Target colour in BGR channel order.
    threshold:
        Maximum acceptable OpenCV-LAB Euclidean distance (≈ CIE ΔE units
        multiplied by 2.55 for the L channel).
    """
    target_px = np.array([[list(target_bgr)]], dtype=np.uint8)
    target_lab = cv2.cvtColor(target_px, cv2.COLOR_BGR2Lab).astype(np.float32)[0, 0]
    diff = chart_lab - target_lab
    dist = np.sqrt((diff * diff).sum(axis=2))
    return (dist <= threshold).astype(np.uint8) * 255


def _compute_bgr_mask(
    chart_bgr: "np.ndarray",
    target_bgr: Tuple[int, int, int],
    threshold: float,
) -> "np.ndarray":
    """
    Return a binary mask where each pixel's BGR Euclidean distance to
    *target_bgr* is ≤ *threshold*.

    Provides an RGB-space fallback that complements the HSV and LAB passes
    for detecting neon green/red line pixels on TradingView's dark theme
    background, including pixels affected by JPEG compression artefacts or
    colour-profile shifts outside the HSV and LAB detection windows.

    Parameters
    ----------
    chart_bgr:
        Source image in BGR channel order (uint8, same shape as source).
    target_bgr:
        Target colour in BGR channel order.
    threshold:
        Maximum acceptable Euclidean distance in BGR space.
    """
    target_arr = np.array(list(target_bgr), dtype=np.float32)
    diff = chart_bgr.astype(np.float32) - target_arr
    dist = np.sqrt((diff * diff).sum(axis=2))
    return (dist <= threshold).astype(np.uint8) * 255


def _merge_collinear_fragments(
    segments: List[Tuple[int, str]], tol: int = 10
) -> List[Tuple[int, str]]:
    """Merge nearby collinear fragment y-positions into a single representative.

    Segments whose y-coordinates lie within *tol* pixels of each other are
    grouped into a cluster.  Each cluster is collapsed into a single
    ``(median_y, dominant_color)`` tuple.  This handles the common case where
    a single horizontal line generates multiple nearby detections from the
    contour and Hough detection paths.

    Parameters
    ----------
    segments:
        List of ``(y_pixel, color_label)`` tuples (unsorted is fine).
    tol:
        Maximum pixel gap between consecutive y-values within a cluster.

    Returns
    -------
    Merged list with at most ``len(segments)`` entries.
    """
    if not segments:
        return []
    sorted_segs = sorted(segments, key=lambda x: x[0])
    groups: List[List[Tuple[int, str]]] = [[sorted_segs[0]]]
    for y, color in sorted_segs[1:]:
        if y - groups[-1][-1][0] <= tol:
            groups[-1].append((y, color))
        else:
            groups.append([(y, color)])
    result: List[Tuple[int, str]] = []
    for group in groups:
        ys = [y for y, _ in group]
        color_counts: dict = {}
        for _, c in group:
            color_counts[c] = color_counts.get(c, 0) + 1
        dominant = max(color_counts, key=lambda k: color_counts[k])
        result.append((sorted(ys)[len(ys) // 2], dominant))
    return result


def _detect_horizontal_lines(
    img: "np.ndarray",
    debug_out: Optional[dict] = None,
) -> List[Tuple[int, str]]:
    """
    Detect coloured horizontal lines in a TradingView dark-theme chart image.

    Uses two complementary detection methods whose results are merged:

    * **Contour bounding boxes** – wide, flat bounding rectangles after
      morphological clean-up.
    * **Probabilistic Hough lines** – near-horizontal line segments.

    Detection is restricted to the chart plot area (right price-axis strip,
    top title/watermark area and bottom session/ATR panel are excluded).
    An additional right-edge exclusion zone (last ``_CHART_RIGHT_EXCLUSION_FRACTION``
    of the chart width) prevents the current-price label/arrow from being
    reported as a price level.

    Large filled blobs (session/overlay boxes) are suppressed before Hough
    detection so their top and bottom edges are not mistaken for horizontal
    levels.

    Three colour passes are used: HSV, LAB-distance, and BGR-distance (each
    OR-combined) for maximum robustness against anti-aliasing and compression.

    Parameters
    ----------
    img:
        BGR image array from ``cv2.imdecode``.
    debug_out:
        Optional dict populated in-place with diagnostic counters.  New keys
        populated by this function: ``raw_green_candidates``,
        ``raw_red_candidates``, ``filtered_by_slope``, ``filtered_by_length``.

    Returns
    -------
    list of (y_pixel, color_label) tuples, where ``color_label`` is
    ``"green"`` or ``"red"``.  y values are in full-image coordinates.
    """
    h, w = img.shape[:2]

    # ── Chart ROI: exclude price axis (right), title (top), bottom panel ─────
    roi_top = int(h * _ROI_TOP_FRACTION)
    roi_bottom = int(h * _ROI_BOTTOM_FRACTION)
    roi_right = int(w * (1.0 - _PRICE_AXIS_FRACTION))
    chart = img[roi_top:roi_bottom, 0:roi_right]
    chart_h, chart_w = chart.shape[:2]

    if debug_out is not None:
        debug_out["chart_roi"] = {
            "top": roi_top,
            "bottom": roi_bottom,
            "left": 0,
            "right": roi_right,
        }

    min_line_width = max(10, int(chart_w * _LINE_MIN_WIDTH_FRACTION))
    max_line_height = max(5, chart_h // 100)

    # Right-edge exclusion: column index beyond which pixels are zeroed out in
    # the processed mask.  This stops the current-price label/arrow (drawn at
    # the far-right of the plot area) from being detected as a price level.
    # Axis OCR uses the full image and is not affected by this value.
    detect_right = int(chart_w * (1.0 - _CHART_RIGHT_EXCLUSION_FRACTION))

    hsv = cv2.cvtColor(chart, cv2.COLOR_BGR2HSV)

    # Pre-compute LAB image once for LAB second-pass detection
    chart_lab = cv2.cvtColor(chart, cv2.COLOR_BGR2Lab).astype(np.float32)

    # ── Green mask: HSV  OR  LAB  OR  BGR (triple-pass) ──────────────────────
    green_mask_hsv = cv2.inRange(
        hsv,
        np.array(_GREEN_LOWER, dtype=np.uint8),
        np.array(_GREEN_UPPER, dtype=np.uint8),
    )
    green_mask_lab = _compute_lab_mask(chart_lab, _GREEN_BGR_TARGET, _LAB_DISTANCE_THRESHOLD)
    green_mask_bgr = _compute_bgr_mask(chart, _GREEN_BGR_TARGET, _BGR_DISTANCE_THRESHOLD)
    green_mask = cv2.bitwise_or(
        cv2.bitwise_or(green_mask_hsv, green_mask_lab), green_mask_bgr
    )

    # ── Red mask: HSV (two hue ranges)  OR  LAB  OR  BGR (triple-pass) ───────
    red_mask_a = cv2.inRange(
        hsv,
        np.array(_RED_LOWER_A, dtype=np.uint8),
        np.array(_RED_UPPER_A, dtype=np.uint8),
    )
    red_mask_b = cv2.inRange(
        hsv,
        np.array(_RED_LOWER_B, dtype=np.uint8),
        np.array(_RED_UPPER_B, dtype=np.uint8),
    )
    red_mask_lab = _compute_lab_mask(chart_lab, _RED_BGR_TARGET, _LAB_DISTANCE_THRESHOLD)
    red_mask_bgr = _compute_bgr_mask(chart, _RED_BGR_TARGET, _BGR_DISTANCE_THRESHOLD)
    red_mask = cv2.bitwise_or(
        cv2.bitwise_or(cv2.bitwise_or(red_mask_a, red_mask_b), red_mask_lab),
        red_mask_bgr,
    )

    if debug_out is not None:
        debug_out["green_mask_pixels"] = int(cv2.countNonZero(green_mask))
        debug_out["red_mask_pixels"] = int(cv2.countNonZero(red_mask))

    # ── Morphology kernels ────────────────────────────────────────────────────
    # Horizontal close: bridges anti-aliased gaps (2 % of chart width, ≥ 15 px)
    close_w = max(15, int(chart_w * 0.02))
    h_close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_w, 1))
    # Horizontal open: strips tiny isolated pixel noise without killing thin lines
    noise_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))

    # ── Hough parameters ─────────────────────────────────────────────────────
    hough_angle_sin = np.sin(np.radians(_HOUGH_ANGLE_MAX_DEG))
    hough_gap = max(5, int(chart_w * 0.02))        # 2 % gap tolerance
    hough_threshold = max(20, min_line_width // 3)

    results: List[Tuple[int, str]] = []
    contour_raw = 0
    hough_raw = 0
    raw_green_candidates = 0
    raw_red_candidates = 0
    filtered_by_slope = 0
    filtered_by_length = 0

    # Height threshold above which a mask blob is classified as a filled
    # session/overlay rectangle and suppressed before Hough detection.
    box_threshold_h = max_line_height * _BOX_HEIGHT_FACTOR

    for mask, color_label in [(green_mask, "green"), (red_mask, "red")]:
        color_raw = 0
        color_slope_filtered = 0
        color_length_filtered = 0

        # Morphology: close to reconnect fragments, open to strip noise
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, h_close_kernel)
        processed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, noise_kernel)

        # ── Session-box suppression ───────────────────────────────────────────
        # Suppress tall blobs that are NOT predominantly horizontal.
        # Pure candle bodies (narrow) and session overlay boxes (square-ish)
        # are removed so Hough does not misclassify their edges as price levels.
        #
        # IMPORTANT: blobs that are both tall AND wide with a high
        # width-to-height ratio are horizontal level lines that have been
        # merged with a same-colour candlestick body (they share pixels where
        # the candle crosses the level line).  Suppressing these entire blobs
        # erases the level line itself from the mask.  We preserve them so
        # that the Hough pass (Method B) can still detect the horizontal
        # segment within the merged blob.
        cnts_box, _ = cv2.findContours(
            processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt in cnts_box:
            _, _, cw_b, ch_b = cv2.boundingRect(cnt)
            if ch_b > box_threshold_h:
                # A merged "line + candle" blob spans the chart width and
                # keeps a favourable width:height ratio.  Leave it intact.
                is_line_like = (
                    cw_b >= min_line_width
                    and cw_b / ch_b >= _CONTOUR_MIN_ASPECT
                )
                if not is_line_like:
                    cv2.drawContours(processed, [cnt], -1, 0, thickness=-1)

        # ── Right-edge exclusion ──────────────────────────────────────────────
        if detect_right < chart_w:
            processed[:, detect_right:] = 0

        # Method A: contour bounding boxes (wide & flat = horizontal segment)
        contours, _ = cv2.findContours(
            processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            color_raw += 1
            contour_raw += 1
            # Aspect-ratio guard: width must be at least _CONTOUR_MIN_ASPECT × height.
            # This rejects any remaining wide-but-tall blobs (partial session boxes).
            aspect_ok = ch == 0 or (cw / ch >= _CONTOUR_MIN_ASPECT)
            if cw >= min_line_width and ch <= max_line_height and aspect_ok:
                center_y = y + ch // 2
                results.append((center_y + roi_top, color_label))
            else:
                if cw < min_line_width:
                    color_length_filtered += 1
                else:
                    color_slope_filtered += 1

        # Method B: probabilistic Hough lines (near-horizontal)
        hough_result = cv2.HoughLinesP(
            processed,
            rho=1,
            theta=np.pi / 180,
            threshold=hough_threshold,
            minLineLength=min_line_width,
            maxLineGap=hough_gap,
        )
        if hough_result is not None:
            for seg in hough_result:
                x1, y1, x2, y2 = seg.tolist()
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                length = np.hypot(dx, dy)
                color_raw += 1
                hough_raw += 1
                # Dual slope filter: angular threshold AND absolute |dy| limit.
                # The absolute limit prevents short segments with acceptable angle
                # but non-trivial pixel slope from contaminating the results.
                slope_ok = (
                    length > 0
                    and dy / length <= hough_angle_sin
                    and dy <= _HOUGH_MAX_SLOPE_DY_PX
                )
                if slope_ok:
                    center_y = (y1 + y2) // 2
                    results.append((center_y + roi_top, color_label))
                else:
                    color_slope_filtered += 1

        if color_label == "green":
            raw_green_candidates = color_raw
        else:
            raw_red_candidates = color_raw
        filtered_by_slope += color_slope_filtered
        filtered_by_length += color_length_filtered

    if debug_out is not None:
        debug_out["contour_segments_raw"] = contour_raw
        debug_out["hough_segments_raw"] = hough_raw
        debug_out["raw_green_candidates"] = raw_green_candidates
        debug_out["raw_red_candidates"] = raw_red_candidates
        debug_out["filtered_by_slope"] = filtered_by_slope
        debug_out["filtered_by_length"] = filtered_by_length
        debug_out["segments_before_dedup"] = len(results)

    # Merge nearby collinear fragments before deduplication.  Clusters of
    # detections within 10 px are collapsed to their median y-position so that
    # a single physical line generates exactly one output entry.
    merged = _merge_collinear_fragments(results)

    deduped = _deduplicate_lines(merged)

    if debug_out is not None:
        debug_out["segments_after_dedup"] = len(deduped)

    return deduped


def _parse_price_axis(img: "np.ndarray") -> List[Tuple[int, float]]:
    """
    OCR the price-axis labels on the right side of the chart.

    Parameters
    ----------
    img:
        BGR image array (full screenshot).

    Returns
    -------
    list of (y_pixel, price) pairs sorted by y ascending.
    An empty list is returned when OCR is unavailable or produces no results.
    """
    if not (_CV2_AVAILABLE and _TESSERACT_AVAILABLE):
        return []

    h, w = img.shape[:2]
    axis_x = int(w * (1.0 - _PRICE_AXIS_FRACTION))
    axis_crop = img[:, axis_x:, :]

    gray = cv2.cvtColor(axis_crop, cv2.COLOR_BGR2GRAY)
    # TradingView: bright text on dark background → binary threshold for white text
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    try:
        data = pytesseract.image_to_data(
            thresh,
            config="--psm 11 -c tessedit_char_whitelist=0123456789.,",
            output_type=pytesseract.Output.DICT,
        )
    except (TesseractNotFoundError, OSError, Exception) as exc:
        logger.warning("Price-axis OCR failed: %s", exc)
        return []

    axis_points: List[Tuple[int, float]] = []
    for i, text in enumerate(data["text"]):
        raw = (text or "").strip()
        if not raw:
            continue
        clean = raw.replace(",", "")
        # Require at least 2 digits to avoid stray characters
        if not re.fullmatch(r"\d+(\.\d+)?", clean):
            continue
        try:
            price = float(clean)
        except ValueError:
            continue
        if price <= 0:
            continue
        word_y = int(data["top"][i]) + int(data["height"][i]) // 2
        axis_points.append((word_y, price))

    # Sort ascending by y-pixel (top of image = highest price)
    axis_points.sort(key=lambda p: p[0])

    # Sanity-filter: in a chart, price decreases as y increases.
    # Remove any points that violate this monotonicity to protect interpolation.
    filtered: List[Tuple[int, float]] = []
    for pt in axis_points:
        if not filtered or pt[1] < filtered[-1][1]:
            filtered.append(pt)

    return filtered


def _map_y_to_price(
    y: int, axis_points: List[Tuple[int, float]]
) -> Optional[float]:
    """
    Map a y-pixel coordinate to a price using linear interpolation (or
    extrapolation) from *axis_points*.

    Parameters
    ----------
    y:
        Y-pixel position (0 = top of image).
    axis_points:
        Sorted list of (y_pixel, price) pairs with y ascending and price
        strictly decreasing.

    Returns
    -------
    Interpolated/extrapolated price, or ``None`` when fewer than 2 axis
    points are available.
    """
    if len(axis_points) < 2:
        return None

    # Search for the bracketing interval
    for i in range(len(axis_points) - 1):
        y0, p0 = axis_points[i]
        y1, p1 = axis_points[i + 1]
        if y0 <= y <= y1:
            if y1 == y0:
                return (p0 + p1) / 2.0
            t = (y - y0) / (y1 - y0)
            return p0 + t * (p1 - p0)

    # Extrapolate beyond the ends using the nearest pair
    if y < axis_points[0][0]:
        y0, p0 = axis_points[0]
        y1, p1 = axis_points[1]
    else:
        y0, p0 = axis_points[-2]
        y1, p1 = axis_points[-1]

    if y1 == y0:
        return (p0 + p1) / 2.0
    t = (y - y0) / (y1 - y0)
    return p0 + t * (p1 - p0)


def _extract_chart_labels(
    img: "np.ndarray",
) -> List[Tuple[int, str]]:
    """
    OCR the right-margin label zone of a TradingView chart to find named
    level annotations (e.g. "Asia High", "New York Low").

    TradingView renders these as white text directly on the dark chart
    background, to the left of the numeric price axis.

    Parameters
    ----------
    img:
        Full BGR screenshot.

    Returns
    -------
    List of ``(y_pixel, raw_label_text)`` tuples for all recognised text
    tokens in the label zone.  The y-pixel is the vertical centre of the
    text word in full-image coordinates.  Returns an empty list when OCR
    is unavailable.
    """
    if not (_CV2_AVAILABLE and _TESSERACT_AVAILABLE):
        return []

    h, w = img.shape[:2]
    label_x0 = int(w * _LABEL_ZONE_LEFT_FRACTION)
    label_x1 = int(w * _LABEL_ZONE_RIGHT_FRACTION)
    label_crop = img[:, label_x0:label_x1, :]

    gray = cv2.cvtColor(label_crop, cv2.COLOR_BGR2GRAY)
    # TradingView renders annotations in white and dim-gray text on a dark
    # background.  A threshold of 100 (vs the previous 140) ensures gray
    # labels (e.g. "Prev Day Low / London Low") are binarised along with the
    # bright-white ones.
    _, thresh = cv2.threshold(gray, _LABEL_BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)

    try:
        data = pytesseract.image_to_data(
            thresh,
            config="--psm 11",
            output_type=pytesseract.Output.DICT,
        )
    except (TesseractNotFoundError, OSError, Exception) as exc:
        logger.debug("Chart-label OCR failed: %s", exc)
        return []

    tokens: List[Tuple[int, str]] = []
    for i, text in enumerate(data["text"]):
        raw = (text or "").strip()
        if not raw or len(raw) < 2:
            continue
        # Filter out numeric-only tokens (those belong to the price axis)
        if re.fullmatch(r"[\d.,]+", raw):
            continue
        # Skip very low-confidence tokens
        conf = int(data.get("conf", [-1] * len(data["text"]))[i] or -1)
        if conf < 20 and conf != -1:
            continue
        word_y = int(data["top"][i]) + int(data["height"][i]) // 2
        tokens.append((word_y, raw))

    return tokens


def _match_labels_to_lines(
    label_tokens: List[Tuple[int, str]],
    lines: List[Tuple[int, str]],
    axis_points: List[Tuple[int, float]],
    tol: int = _LABEL_LINE_MATCH_TOLERANCE_PX,
) -> Dict[str, float]:
    """
    Associate OCR label tokens with detected horizontal lines and compute
    the price for each recognised field.

    Strategy
    --------
    1. Group nearby label tokens (within ``tol`` px) into phrases.
    2. For each phrase, resolve it to a ``LevelsPayload`` field via
       :func:`_label_to_field`.
    3. Find the nearest horizontal line within ``tol`` pixels.
    4. Map that line's y-coordinate to a price via :func:`_map_y_to_price`.

    Parameters
    ----------
    label_tokens:
        ``(y_pixel, text)`` pairs from :func:`_extract_chart_labels`.
    lines:
        ``(y_pixel, color)`` pairs from :func:`_detect_horizontal_lines`.
    axis_points:
        Calibrated price axis for price mapping.
    tol:
        Maximum pixel gap for grouping tokens into phrases AND for
        matching a phrase to its nearest line.

    Returns
    -------
    Dict mapping LevelsPayload field names to their extracted price values.
    """
    if not label_tokens or not lines or len(axis_points) < 2:
        return {}

    # ── Step 1: group nearby tokens into phrases ──────────────────────────────
    # Tokens within tol px vertically are on the same label row; concatenate
    # their text in x-order (approximate by token order from Tesseract).
    sorted_tokens = sorted(label_tokens, key=lambda t: t[0])
    phrases: List[Tuple[int, str]] = []  # (median_y, full_phrase)
    group: List[Tuple[int, str]] = [sorted_tokens[0]]
    for token in sorted_tokens[1:]:
        if token[0] - group[-1][0] <= tol:
            group.append(token)
        else:
            median_y = sorted(g[0] for g in group)[len(group) // 2]
            phrase = " ".join(t[1] for t in group)
            phrases.append((median_y, phrase))
            group = [token]
    if group:
        median_y = sorted(g[0] for g in group)[len(group) // 2]
        phrase = " ".join(t[1] for t in group)
        phrases.append((median_y, phrase))

    # ── Step 2 & 3: resolve each phrase to a field and nearest line ───────────
    labeled: Dict[str, float] = {}
    line_ys = [y for y, _ in lines]

    for phrase_y, phrase_text in phrases:
        field = _label_to_field(phrase_text)
        if field is None:
            continue

        # Find nearest detected line
        if not line_ys:
            continue
        nearest_y = min(line_ys, key=lambda ly: abs(ly - phrase_y))
        if abs(nearest_y - phrase_y) > tol * 3:
            # No line close enough — try mapping directly from the phrase y
            nearest_y = phrase_y

        price = _map_y_to_price(nearest_y, axis_points)
        if price is None or price <= 0:
            continue

        # Only overwrite a field if this match is closer to its line
        if field not in labeled:
            labeled[field] = round(price, 4)
        else:
            # Keep whichever was matched with a closer line distance
            existing_y = min(line_ys, key=lambda ly: abs(ly - phrase_y))
            if abs(nearest_y - phrase_y) < abs(existing_y - phrase_y):
                labeled[field] = round(price, 4)

    return labeled


def _extract_session_info(
    img: "np.ndarray",
) -> Tuple[Optional[str], Optional[float]]:
    """
    Parse the session info panel in the bottom-right corner of the chart.

    TradingView renders a small table: ``Session``, ``ATR (Chart)``,
    ``IB Volume``.  This function extracts the session name and ATR value.

    Returns
    -------
    ``(session_name, atr_value)`` — either or both may be ``None`` when
    OCR cannot read the panel.
    """
    if not (_CV2_AVAILABLE and _TESSERACT_AVAILABLE):
        return None, None

    h, w = img.shape[:2]
    panel_y0 = int(h * _SESSION_PANEL_TOP_FRACTION)
    panel_x0 = int(w * _SESSION_PANEL_LEFT_FRACTION)
    panel_crop = img[panel_y0:, panel_x0:, :]

    gray = cv2.cvtColor(panel_crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

    try:
        text = pytesseract.image_to_string(thresh, config="--psm 6").strip()
    except (TesseractNotFoundError, OSError, Exception) as exc:
        logger.debug("Session panel OCR failed: %s", exc)
        return None, None

    session: Optional[str] = None
    atr_val: Optional[float] = None

    for line in text.splitlines():
        line = line.strip()
        # Session line: "Session   Globex" or "Session New York"
        m_sess = re.search(
            r"session[:\s]+([a-z ()\-]+)",
            line,
            re.IGNORECASE,
        )
        if m_sess and session is None:
            session = m_sess.group(1).strip().title()

        # ATR line: "ATR (Chart) 113.00" or "ATR   107.50"
        m_atr = re.search(
            r"atr[^0-9]*([0-9]+\.?[0-9]*)",
            line,
            re.IGNORECASE,
        )
        if m_atr and atr_val is None:
            try:
                atr_val = float(m_atr.group(1))
            except ValueError:
                pass

    return session, atr_val


def _filter_outlier_prices(
    prices: List[float],
    current_price: float,
    atr: Optional[float],
) -> Tuple[List[float], List[Tuple[float, str]]]:
    """
    Mechanically reject implausible prices using a 3-tier fact-finding approach.

    This prevents wild OCR mis-reads (e.g. 16 072 when all other levels
    cluster around 28 000) from contaminating the scoring payload.

    Tier 1 — ATR envelope
        Any price further than ``_OUTLIER_ATR_ENVELOPE × ATR`` from the
        current price is rejected immediately.  This tier is only applied
        when ATR is available.

    Tier 2 — IQR statistical filter
        With ≥ ``_OUTLIER_IQR_MIN_PEERS`` surviving prices, compute Q1/Q3
        and reject any price outside
        ``[Q1 - IQR × _OUTLIER_IQR_MULTIPLIER, Q3 + IQR × _OUTLIER_IQR_MULTIPLIER]``.

    Tier 3 — Peer-vote isolation check
        A price is rejected if it differs from every remaining peer by more
        than 20 % of the current price.  This catches the case where a
        single rogue value survives Tier 1 and Tier 2 but is clearly
        disconnected from the cluster.

    Parameters
    ----------
    prices:
        Raw mapped prices from line detection (may include OCR garbage).
    current_price:
        Estimated last-bar close price used as the ATR anchor.
    atr:
        Detected ATR value; ``None`` skips the ATR-envelope tier.

    Returns
    -------
    ``(clean_prices, rejected)`` where *rejected* is a list of
    ``(price, reason)`` tuples documenting every removal.
    """
    if not prices:
        return [], []

    rejected_prices: Dict[float, str] = {}

    # ── Tier 1: ATR envelope ──────────────────────────────────────────────────
    if atr and atr > 0:
        envelope = atr * _OUTLIER_ATR_ENVELOPE
        for p in prices:
            if p not in rejected_prices and abs(p - current_price) > envelope:
                atr_ratio = abs(p - current_price) / atr
                rejected_prices[p] = (
                    f"ATR-envelope rejection: {p:.2f} is {atr_ratio:.1f}× ATR "
                    f"({atr:.2f}) from current price {current_price:.2f} "
                    f"(limit {_OUTLIER_ATR_ENVELOPE}× ATR = {envelope:.2f})"
                )

    surviving = [p for p in prices if p not in rejected_prices]

    # ── Tier 2: IQR statistical filter ───────────────────────────────────────
    if len(surviving) >= _OUTLIER_IQR_MIN_PEERS:
        sorted_s = sorted(surviving)
        n = len(sorted_s)
        q1 = sorted_s[n // 4]
        q3 = sorted_s[(3 * n) // 4]
        iqr = q3 - q1
        if iqr > 0:
            lo = q1 - _OUTLIER_IQR_MULTIPLIER * iqr
            hi = q3 + _OUTLIER_IQR_MULTIPLIER * iqr
            for p in surviving[:]:
                if p not in rejected_prices and not (lo <= p <= hi):
                    rejected_prices[p] = (
                        f"IQR-outlier rejection: {p:.2f} falls outside "
                        f"[{lo:.2f}, {hi:.2f}] "
                        f"(Q1={q1:.2f} Q3={q3:.2f} IQR={iqr:.2f} "
                        f"multiplier={_OUTLIER_IQR_MULTIPLIER})"
                    )

    surviving = [p for p in prices if p not in rejected_prices]

    # ── Tier 3: peer-vote isolation check ────────────────────────────────────
    isolation_threshold = current_price * 0.20
    if len(surviving) >= 2:
        for p in surviving[:]:
            if p in rejected_prices:
                continue
            peers = [q for q in surviving if q != p and q not in rejected_prices]
            if peers and all(abs(p - q) > isolation_threshold for q in peers):
                rejected_prices[p] = (
                    f"Isolation rejection: {p:.2f} differs from every peer "
                    f"by >{isolation_threshold:.2f} (20 % of current price "
                    f"{current_price:.2f})"
                )

    clean = [p for p in prices if p not in rejected_prices]
    rejected_list: List[Tuple[float, str]] = [
        (p, reason) for p, reason in rejected_prices.items()
    ]
    if rejected_list:
        logger.info(
            "Outlier filter removed %d price(s): %s",
            len(rejected_list),
            [(round(p, 2), r[:60]) for p, r in rejected_list],
        )
    return clean, rejected_list


def _build_levels_payload(
    line_prices: List[float],
    current_price: float,
    labeled_levels: Optional[Dict[str, float]] = None,
) -> LevelsPayload:
    """
    Build a ``LevelsPayload`` from extracted line prices and an estimated
    current price, preferring label-mapped values when available.

    When *labeled_levels* is provided and non-empty, those field assignments
    take precedence over the positional (blind) slot-filling.  Any remaining
    unnamed line prices are used to fill empty optional slots.

    An ATR14 estimate is derived from the full detected price range; if
    ``atr14`` is already present in *labeled_levels* it is kept.
    """
    labeled = labeled_levels or {}

    resistance = sorted(
        [p for p in line_prices if p > current_price], reverse=True
    )
    support = sorted([p for p in line_prices if p < current_price])

    # Named field slots in priority order (used when label mapping misses a slot)
    _RES_FIELDS = ["pdh", "globex_high", "asia_high", "london_high", "ny_high"]
    _SUP_FIELDS = ["pdl", "globex_low", "asia_low", "london_low", "ny_low"]

    # Start with label-mapped values (authoritative)
    kwargs: dict = {}
    kwargs.update({k: v for k, v in labeled.items() if v is not None})

    # Required fields — fill from labels or fall back to positional heuristic
    if "pdh" not in kwargs:
        kwargs["pdh"] = (
            resistance[0]
            if resistance
            else round(current_price * _SYNTHETIC_RESISTANCE_OFFSET, 4)
        )
    if "pdl" not in kwargs:
        kwargs["pdl"] = (
            support[0]
            if support
            else round(current_price * _SYNTHETIC_SUPPORT_OFFSET, 4)
        )
    if "prior_settle" not in kwargs:
        kwargs["prior_settle"] = current_price

    # Fill remaining optional slots with unnamed line prices
    res_used = {kwargs.get(f) for f in _RES_FIELDS if f in kwargs}
    sup_used = {kwargs.get(f) for f in _SUP_FIELDS if f in kwargs}

    res_unnamed = [p for p in resistance if p not in res_used]
    sup_unnamed = [p for p in support if p not in sup_used]

    for slot, fname in enumerate(_RES_FIELDS[1:], start=1):
        if fname not in kwargs and slot - 1 < len(res_unnamed):
            kwargs[fname] = res_unnamed[slot - 1]

    for slot, fname in enumerate(_SUP_FIELDS[1:], start=1):
        if fname not in kwargs and slot - 1 < len(sup_unnamed):
            kwargs[fname] = sup_unnamed[slot - 1]

    # ATR14: use detected ATR from session panel if present in labeled dict,
    # otherwise estimate from the full detected price range.
    if "atr14" not in kwargs:
        all_prices = list(resistance) + list(support) + [current_price]
        if len(all_prices) >= 2:
            price_range = max(all_prices) - min(all_prices)
            if price_range > 0:
                kwargs["atr14"] = round(price_range / 14.0, 4)

    return LevelsPayload(**kwargs)


def _compute_confidence(
    num_axis_points: int,
    num_lines: int,
    num_mapped: int,
    cp_from_axis: bool,
) -> float:
    """
    Compute an overall extraction confidence score in [0.0, 1.0].

    Each of four independent components contributes up to 0.25:

    - axis_points  – a reliable price scale (≥ 3 OCR labels)
    - lines        – at least one coloured horizontal line detected
    - mapped       – at least one line successfully mapped to a price
    - current_price from axis (rather than estimated as median/midpoint)
    """
    score = 0.0
    if num_axis_points >= _MIN_AXIS_POINTS_FOR_SCALE:
        score += CONFIDENCE_AXIS_POINTS
    if num_lines >= 1:
        score += CONFIDENCE_LINES_DETECTED
    if num_mapped >= 1:
        score += CONFIDENCE_PRICES_MAPPED
    if cp_from_axis:
        score += CONFIDENCE_CURRENT_PRICE
    return round(score, 4)


def _build_warning(
    num_axis_points: int,
    num_lines: int,
    num_mapped: int,
    current_price: Optional[float],
) -> str:
    """Compose a human-readable warning for low-confidence extraction."""
    parts: List[str] = []
    if num_axis_points < _MIN_AXIS_POINTS_FOR_SCALE:
        parts.append(
            f"only {num_axis_points} price label(s) found on the axis "
            f"(need ≥ {_MIN_AXIS_POINTS_FOR_SCALE} for reliable scale)"
        )
    if num_lines == 0:
        parts.append("no coloured horizontal lines detected in the chart area")
    elif num_mapped == 0:
        parts.append(
            "lines were detected but could not be mapped to prices "
            "(price scale unavailable)"
        )
    if current_price is None:
        parts.append("current price could not be estimated")

    body = "; ".join(parts) if parts else "extraction confidence is low"
    return f"Low confidence extraction: {body}."


def _estimate_current_price(
    img: "np.ndarray",
    axis_points: List[Tuple[int, float]],
    line_prices: List[float],
) -> Tuple[Optional[float], bool]:
    """
    Estimate the current (last-bar close) price from the chart image.

    Strategy
    --------
    1. Look for the TradingView "current price" marker: a horizontally
       elongated bright rectangle in the price-axis area at a distinct
       y-position.  Map that y-position through *axis_points*.
    2. Fallback: median of mapped *line_prices*.
    3. Fallback: midpoint of the price-axis range.

    Returns
    -------
    (price, from_axis) where *from_axis* is True for strategy 1.
    """
    if not (_CV2_AVAILABLE and len(axis_points) >= 2):
        # No reliable mapping → use median of line prices
        if line_prices:
            sorted_lp = sorted(line_prices)
            return sorted_lp[len(sorted_lp) // 2], False
        return None, False

    h, w = img.shape[:2]
    axis_x = int(w * (1.0 - _PRICE_AXIS_FRACTION))

    # The TradingView current-price box is a bright rectangle in the axis area.
    # Look for a horizontal bright band that is NOT part of the regular grid.
    axis_region = img[:, axis_x:, :]
    gray = cv2.cvtColor(axis_region, cv2.COLOR_BGR2GRAY)

    # Bright threshold (label box background is white or near-white)
    _, bright = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Horizontal projection: find rows with many bright pixels
    row_sums = bright.sum(axis=1)  # shape (h,)
    axis_w = w - axis_x
    bright_rows = [
        y
        for y, s in enumerate(row_sums)
        if s > axis_w * 0.3 * 255  # at least 30 % of axis width is bright
    ]

    if bright_rows:
        marker_y = int(sum(bright_rows) / len(bright_rows))
        price = _map_y_to_price(marker_y, axis_points)
        if price is not None and price > 0:
            return round(price, 4), True

    # Fallback 1: median of line prices
    if line_prices:
        sorted_lp = sorted(line_prices)
        return sorted_lp[len(sorted_lp) // 2], False

    # Fallback 2: midpoint of axis range
    if len(axis_points) >= 2:
        prices = [p for _, p in axis_points]
        return round((max(prices) + min(prices)) / 2.0, 4), False

    return None, False


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def extract_from_image(
    image_bytes: bytes,
    ticker: str = "UNKNOWN",
    date_et: Optional[str] = None,
    timeframe: str = "30m",
    lookback_days: int = 5,
    debug: bool = False,
) -> ExtractionResult:
    """
    Extract price levels from a chart screenshot.

    Parameters
    ----------
    image_bytes:
        Raw bytes of a PNG, JPEG, or WebP file.
    ticker:
        Ticker symbol (used only for logging).
    date_et, timeframe, lookback_days:
        Passed through for context logging; not used in extraction.
    debug:
        When ``True``, populate ``ExtractionResult.debug_info`` with
        diagnostic counters (image size, chart ROI, mask pixel counts,
        raw/filtered candidate counts, segment counts before and after
        filtering, and ``kept_lines`` – a list of dicts with ``y_pixel``,
        ``color``, and ``price`` for each detected level, and
        ``chart_labels`` – a list of dicts with ``label``, ``y``,
        ``field``, and ``price`` for each resolved chart annotation).

    Returns
    -------
    An :class:`ExtractionResult`.  When extraction fails completely
    (bad image, missing libraries, etc.) all numeric fields are zero/None
    and ``image_decoded`` is ``False``.
    """
    if not _CV2_AVAILABLE:
        logger.warning("OpenCV not available; image extraction skipped.")
        return ExtractionResult()

    # Decode image ────────────────────────────────────────────────────────────
    try:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as exc:
        logger.warning("Image decode error for ticker=%s: %s", ticker, exc)
        return ExtractionResult()

    if img is None:
        logger.info("cv2.imdecode returned None for ticker=%s (invalid image)", ticker)
        return ExtractionResult()

    # Collect debug telemetry when requested
    _debug_info: Optional[dict] = None
    if debug:
        h_img, w_img = img.shape[:2]
        _debug_info = {"image_size": {"width": w_img, "height": h_img}}

    # Step 1: Parse price axis ────────────────────────────────────────────────
    axis_points = _parse_price_axis(img)

    # Step 2: Detect horizontal lines ─────────────────────────────────────────
    lines = _detect_horizontal_lines(img, debug_out=_debug_info)

    # Step 3: Map line y-positions to prices ──────────────────────────────────
    line_prices: List[float] = []
    kept_lines_debug: List[dict] = []
    for y, color in lines:
        raw_price = _map_y_to_price(y, axis_points)
        if raw_price is not None and raw_price > 0:
            mapped_price: Optional[float] = round(raw_price, 4)
        else:
            mapped_price = None
        if mapped_price is not None:
            line_prices.append(mapped_price)
        if debug and _debug_info is not None:
            kept_lines_debug.append({"y_pixel": y, "color": color, "price": mapped_price})

    if debug and _debug_info is not None:
        _debug_info["kept_lines"] = kept_lines_debug

    # Step 3b: Extract chart-margin labels and match to lines ─────────────────
    labeled_levels: Dict[str, float] = {}
    try:
        label_tokens = _extract_chart_labels(img)
        labeled_levels = _match_labels_to_lines(label_tokens, lines, axis_points)
        if labeled_levels:
            logger.info(
                "Label extraction: ticker=%s found %d labeled levels: %s",
                ticker,
                len(labeled_levels),
                {k: v for k, v in labeled_levels.items()},
            )
        if debug and _debug_info is not None:
            _debug_info["chart_labels"] = [
                {"label": txt, "y": y, "field": _label_to_field(txt), "price": None}
                for y, txt in label_tokens
            ]
            # Enrich with resolved prices where available
            for entry in _debug_info["chart_labels"]:
                f = entry["field"]
                if f and f in labeled_levels:
                    entry["price"] = labeled_levels[f]
    except Exception as exc:
        logger.debug("Chart label extraction failed for ticker=%s: %s", ticker, exc)

    # Step 3c: Extract session info from bottom-right panel ───────────────────
    detected_session: Optional[str] = None
    detected_atr: Optional[float] = None
    try:
        detected_session, detected_atr = _extract_session_info(img)
        if detected_session:
            logger.info("Session panel: ticker=%s session=%s atr=%s", ticker, detected_session, detected_atr)
        # If the panel gave us an ATR, inject it into labeled_levels for payload building
        if detected_atr is not None and detected_atr > 0:
            labeled_levels["atr14"] = round(detected_atr, 4)
    except Exception as exc:
        logger.debug("Session info extraction failed for ticker=%s: %s", ticker, exc)

    # Step 3d: Extract axis price bounds for quality-gate axis-bounds check ───
    axis_price_min: Optional[float] = None
    axis_price_max: Optional[float] = None
    if axis_points:
        axis_prices = [p for _, p in axis_points]
        axis_price_min = round(min(axis_prices), 4)
        axis_price_max = round(max(axis_prices), 4)

    # Step 4: Estimate current price ──────────────────────────────────────────
    current_price, cp_from_axis = _estimate_current_price(img, axis_points, line_prices)

    # If we have prior_settle from labels use it to improve current_price estimate
    if current_price is None and "prior_settle" in labeled_levels:
        current_price = labeled_levels["prior_settle"]
        cp_from_axis = False

    # Step 4b: Mechanical outlier rejection ───────────────────────────────────
    # Run the 3-tier filter on line_prices and on labeled_levels prices alike.
    # Use the detected ATR (from the session panel) as the primary envelope
    # anchor; fall back to the ATR stored in labeled_levels if the panel OCR
    # was unavailable.
    rejected_outliers: List[Tuple[float, str]] = []
    if current_price is not None and (line_prices or labeled_levels):
        # Resolve the best ATR estimate available at this stage
        atr_for_filter: Optional[float] = detected_atr
        if atr_for_filter is None:
            atr_for_filter = labeled_levels.get("atr14")

        # Filter unlabeled line prices
        if line_prices:
            line_prices, line_rejected = _filter_outlier_prices(
                line_prices, current_price, atr_for_filter
            )
            rejected_outliers.extend(line_rejected)

        # Filter labeled level prices (skip non-price keys like atr14)
        if labeled_levels:
            _skip_keys = {"atr14"}
            _labeled_prices = {
                k: v for k, v in labeled_levels.items()
                if k not in _skip_keys and v is not None
            }
            _all_labeled = list(_labeled_prices.values())
            _clean_labeled, _labeled_rejected = _filter_outlier_prices(
                _all_labeled, current_price, atr_for_filter
            )
            rejected_outliers.extend(_labeled_rejected)
            # Remove rejected prices from labeled_levels
            _rejected_set = {p for p, _ in _labeled_rejected}
            if _rejected_set:
                labeled_levels = {
                    k: v for k, v in labeled_levels.items()
                    if k in _skip_keys or v not in _rejected_set
                }

        if rejected_outliers:
            logger.warning(
                "ticker=%s: mechanical outlier filter rejected %d value(s) — %s",
                ticker,
                len(rejected_outliers),
                [(round(p, 2), r.split(":")[0]) for p, r in rejected_outliers],
            )

    # Step 5: Compute confidence ──────────────────────────────────────────────
    # Bonus: label-matched levels improve confidence
    label_bonus = min(0.10, len(labeled_levels) * 0.02) if labeled_levels else 0.0
    confidence = min(
        1.0,
        _compute_confidence(
            num_axis_points=len(axis_points),
            num_lines=len(lines),
            num_mapped=len(line_prices),
            cp_from_axis=cp_from_axis,
        ) + label_bonus,
    )

    # Step 6: Build payload when we have the minimum viable data ──────────────
    levels_payload: Optional[LevelsPayload] = None
    if current_price is not None and (line_prices or labeled_levels):
        try:
            levels_payload = _build_levels_payload(
                line_prices, current_price, labeled_levels=labeled_levels
            )
        except Exception as exc:
            logger.warning("Failed to build LevelsPayload for ticker=%s: %s", ticker, exc)

    warning: Optional[str] = None
    if confidence < CONFIDENCE_LOW_THRESHOLD:
        warning = _build_warning(
            len(axis_points), len(lines), len(line_prices), current_price
        )

    logger.info(
        "Extraction complete: ticker=%s axis_points=%d lines=%d "
        "mapped=%d labels=%d session=%s confidence=%.2f outliers_rejected=%d",
        ticker,
        len(axis_points),
        len(lines),
        len(line_prices),
        len(labeled_levels),
        detected_session or "unknown",
        confidence,
        len(rejected_outliers),
    )

    return ExtractionResult(
        image_decoded=True,
        levels_payload=levels_payload,
        current_price=current_price,
        num_lines_detected=len(lines),
        num_axis_points=len(axis_points),
        extraction_confidence=confidence,
        warning=warning,
        labeled_levels=labeled_levels,
        detected_session=detected_session,
        detected_atr=detected_atr,
        axis_price_min=axis_price_min,
        axis_price_max=axis_price_max,
        rejected_outliers=rejected_outliers,
        debug_info=_debug_info,
    )
