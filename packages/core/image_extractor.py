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
- ``_map_y_to_price(y, axis_points)``
- ``_deduplicate_lines(lines, tol)``
- ``_build_levels_payload(line_prices, current_price)``
- ``_compute_confidence(...)``

Known limitations (MVP)
-----------------------
- Optimised for TradingView dark-theme screenshots only.
- Line-colour detection targets TradingView default green (#089981) and
  red (#f23645).  Custom colours may not be detected.
- Price-axis OCR requires ``tesseract-ocr`` to be installed on the host.
- ``current_price`` is estimated; it is *not* read from user input.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

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
_GREEN_LOWER: Tuple[int, int, int] = (60, 30, 30)
_GREEN_UPPER: Tuple[int, int, int] = (95, 255, 255)

# BGR pixel value used for LAB-space second-pass detection
_GREEN_BGR_TARGET: Tuple[int, int, int] = (129, 153, 8)

# Red #f23645 → BGR(69, 54, 242) → HSV ≈ (178, 200, 242) and wraps near H=0
_RED_LOWER_A: Tuple[int, int, int] = (0, 30, 40)
_RED_UPPER_A: Tuple[int, int, int] = (20, 255, 255)
_RED_LOWER_B: Tuple[int, int, int] = (155, 30, 40)
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

# Right-edge exclusion: last 3 % of chart width is zeroed before line
# detection to prevent the current-price arrow/label from being picked up
# as a false horizontal segment candidate.
_CHART_RIGHT_EXCLUSION_FRACTION: float = 0.03

# Hough near-horizontal tolerance in degrees
_HOUGH_ANGLE_MAX_DEG: float = 5.0

# Additional pixel-level slope gate: detected segments must also satisfy
# |dy| ≤ this many pixels regardless of the angular threshold above.
_HOUGH_MAX_DY_PX: int = 2

# Session-box suppression: connected components taller than this multiple
# of max_line_height are erased before Hough runs (prevents session-box
# edges from being reported as levels).
_SESSION_BOX_HEIGHT_MULTIPLE: int = 3

# Fragment-merging tolerance: collinear fragments within this many y-pixels
# of each other are collapsed to a single representative line.
_FRAGMENT_MERGE_TOL_PX: int = 10

# BGR-distance threshold for third-pass colour detection.
# Accepts pixels within this L2 distance of the target colour in BGR space.
_BGR_DISTANCE_THRESHOLD: float = 80.0

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

    #: Optional diagnostic counters populated when ``debug=True`` is passed to
    #: :func:`extract_from_image`.  Keys: ``image_size``, ``chart_roi``,
    #: ``green_mask_pixels``, ``red_mask_pixels``, ``contour_segments_raw``,
    #: ``hough_segments_raw``, ``filtered_by_slope``,
    #: ``segments_before_dedup``, ``segments_after_merge``,
    #: ``segments_after_dedup``, ``kept_lines``.
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

    Acts as the third-pass colour detector, catching JPEG-compressed or
    lightly shifted neon pixels that slip past the HSV and LAB masks.
    """
    target = np.array(target_bgr, dtype=np.float32)
    diff = chart_bgr.astype(np.float32) - target
    dist = np.sqrt((diff * diff).sum(axis=2))
    return (dist <= threshold).astype(np.uint8) * 255


def _merge_collinear_fragments(
    lines: List[Tuple[int, str]], tol: int = _FRAGMENT_MERGE_TOL_PX
) -> List[Tuple[int, str]]:
    """
    Cluster line detections within *tol* pixels by colour, collapse each
    cluster to its median y, then return the merged list.

    This prevents a single drawn level from producing several detections
    when fragments are picked up at slightly different y-positions.
    """
    if not lines:
        return []

    by_color: dict = {}
    for y, color in lines:
        by_color.setdefault(color, []).append(y)

    result: List[Tuple[int, str]] = []
    for color, ys in by_color.items():
        ys_sorted = sorted(ys)
        clusters: List[List[int]] = [[ys_sorted[0]]]
        for y in ys_sorted[1:]:
            if y - clusters[-1][-1] <= tol:
                clusters[-1].append(y)
            else:
                clusters.append([y])
        for cluster in clusters:
            median_y = sorted(cluster)[len(cluster) // 2]
            result.append((median_y, color))

    return sorted(result, key=lambda t: t[0])


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

    Additional robustness features applied before Hough/contour detection:

    * **Right-edge exclusion** (``_CHART_RIGHT_EXCLUSION_FRACTION``): the
      rightmost fraction of the chart mask is zeroed to suppress the
      current-price arrow that TradingView draws at the far right.
    * **Session-box suppression**: connected components taller than
      ``_SESSION_BOX_HEIGHT_MULTIPLE × max_line_height`` are erased so that
      session-rectangle edges are not mistaken for horizontal levels.
    * **Dual slope filter**: Hough segments must pass both the angular
      threshold *and* the ``|dy| ≤ _HOUGH_MAX_DY_PX`` pixel-level check.
    * **Triple-pass colour mask**: HSV | LAB-distance | BGR-distance for
      maximum neon-colour coverage under JPEG compression.
    * **Fragment merging**: collinear fragments within
      ``_FRAGMENT_MERGE_TOL_PX`` pixels are collapsed to a single line.

    Parameters
    ----------
    img:
        BGR image array from ``cv2.imdecode``.
    debug_out:
        Optional dict populated in-place with diagnostic counters.

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

    # ── Right-edge exclusion zone (last _CHART_RIGHT_EXCLUSION_FRACTION) ──────
    # Zero the rightmost strip so the current-price arrow is not detected.
    exclusion_x = max(0, chart_w - int(chart_w * _CHART_RIGHT_EXCLUSION_FRACTION))

    hsv = cv2.cvtColor(chart, cv2.COLOR_BGR2HSV)

    # Pre-compute LAB and float BGR images once for second/third-pass detection
    chart_lab = cv2.cvtColor(chart, cv2.COLOR_BGR2Lab).astype(np.float32)
    chart_bgr_f = chart.astype(np.float32)

    # ── Green mask: HSV | LAB | BGR triple-pass ───────────────────────────────
    green_mask_hsv = cv2.inRange(
        hsv,
        np.array(_GREEN_LOWER, dtype=np.uint8),
        np.array(_GREEN_UPPER, dtype=np.uint8),
    )
    green_mask_lab = _compute_lab_mask(chart_lab, _GREEN_BGR_TARGET, _LAB_DISTANCE_THRESHOLD)
    green_mask_bgr = _compute_bgr_mask(chart_bgr_f, _GREEN_BGR_TARGET, _BGR_DISTANCE_THRESHOLD)
    green_mask = cv2.bitwise_or(cv2.bitwise_or(green_mask_hsv, green_mask_lab), green_mask_bgr)

    # ── Red mask: HSV (two hue ranges) | LAB | BGR triple-pass ───────────────
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
    red_mask_bgr = _compute_bgr_mask(chart_bgr_f, _RED_BGR_TARGET, _BGR_DISTANCE_THRESHOLD)
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
    filtered_by_slope = 0

    for mask, color_label in [(green_mask, "green"), (red_mask, "red")]:
        # ── Apply right-edge exclusion ────────────────────────────────────
        mask[:, exclusion_x:] = 0

        # Morphology: close to reconnect fragments, open to strip noise
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, h_close_kernel)
        processed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, noise_kernel)

        # ── Session-box suppression ───────────────────────────────────────
        # Erase tall connected components that are session rectangles,
        # not horizontal lines.
        box_thresh = max_line_height * _SESSION_BOX_HEIGHT_MULTIPLE
        contours_all, _ = cv2.findContours(
            processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt in contours_all:
            _, _, _, ch = cv2.boundingRect(cnt)
            if ch > box_thresh:
                cv2.drawContours(processed, [cnt], -1, 0, cv2.FILLED)

        # Method A: contour bounding boxes (wide & flat = horizontal segment)
        contours, _ = cv2.findContours(
            processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            contour_raw += 1
            if cw >= min_line_width and ch <= max_line_height:
                center_y = y + ch // 2
                results.append((center_y + roi_top, color_label))

        # Method B: probabilistic Hough lines (near-horizontal, dual slope gate)
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
                hough_raw += 1
                # Dual slope gate: angular threshold AND pixel-level |dy| cap
                angle_ok = length > 0 and dy / length <= hough_angle_sin
                pixel_ok = dy <= _HOUGH_MAX_DY_PX
                if angle_ok and pixel_ok:
                    center_y = (y1 + y2) // 2
                    results.append((center_y + roi_top, color_label))
                else:
                    filtered_by_slope += 1

    if debug_out is not None:
        debug_out["contour_segments_raw"] = contour_raw
        debug_out["hough_segments_raw"] = hough_raw
        debug_out["filtered_by_slope"] = filtered_by_slope
        debug_out["segments_before_dedup"] = len(results)

    # ── Fragment merging: collapse fragments within ±10 px per colour ─────────
    merged = _merge_collinear_fragments(results)

    if debug_out is not None:
        debug_out["segments_after_merge"] = len(merged)

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


def _build_levels_payload(
    line_prices: List[float], current_price: float
) -> LevelsPayload:
    """
    Build a ``LevelsPayload`` from a list of extracted line prices and an
    estimated current price.

    Resistance prices (above current) are mapped to named resistance fields
    in descending order.  Support prices (below current) are mapped to
    named support fields in ascending order from current.

    An ATR14 estimate is derived from the full detected price range.
    """
    resistance = sorted(
        [p for p in line_prices if p > current_price], reverse=True
    )
    support = sorted([p for p in line_prices if p < current_price])

    # Named field slots in priority order
    _RES_FIELDS = ["pdh", "globex_high", "asia_high", "london_high", "ny_high"]
    _SUP_FIELDS = ["pdl", "globex_low", "asia_low", "london_low", "ny_low"]

    kwargs: dict = {
        # Required fields; fall back to tiny synthetic offsets when no
        # extracted levels exist on that side
        "pdh": resistance[0] if resistance else round(current_price * _SYNTHETIC_RESISTANCE_OFFSET, 4),
        "pdl": support[0] if support else round(current_price * _SYNTHETIC_SUPPORT_OFFSET, 4),
        "prior_settle": current_price,
    }

    for slot, fname in enumerate(_RES_FIELDS[1:], start=1):
        if slot < len(resistance):
            kwargs[fname] = resistance[slot]

    for slot, fname in enumerate(_SUP_FIELDS[1:], start=1):
        if slot < len(support):
            kwargs[fname] = support[slot]

    # Rough ATR14 estimate: range of all detected prices divided by 14
    all_prices = resistance + support + [current_price]
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
        segment counts before and after filtering).

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
    for y, _color in lines:
        price = _map_y_to_price(y, axis_points)
        if price is not None and price > 0:
            line_prices.append(round(price, 4))
            kept_lines_debug.append({"y_pixel": y, "color": _color, "price": round(price, 4)})

    if _debug_info is not None:
        _debug_info["kept_lines"] = kept_lines_debug

    # Step 4: Estimate current price ──────────────────────────────────────────
    current_price, cp_from_axis = _estimate_current_price(img, axis_points, line_prices)

    # Step 5: Compute confidence ──────────────────────────────────────────────
    confidence = _compute_confidence(
        num_axis_points=len(axis_points),
        num_lines=len(lines),
        num_mapped=len(line_prices),
        cp_from_axis=cp_from_axis,
    )

    # Step 6: Build payload when we have the minimum viable data ──────────────
    levels_payload: Optional[LevelsPayload] = None
    if current_price is not None and line_prices:
        try:
            levels_payload = _build_levels_payload(line_prices, current_price)
        except Exception as exc:
            logger.warning("Failed to build LevelsPayload for ticker=%s: %s", ticker, exc)

    warning: Optional[str] = None
    if confidence < CONFIDENCE_LOW_THRESHOLD:
        warning = _build_warning(
            len(axis_points), len(lines), len(line_prices), current_price
        )

    logger.info(
        "Extraction complete: ticker=%s axis_points=%d lines=%d "
        "mapped=%d confidence=%.2f",
        ticker,
        len(axis_points),
        len(lines),
        len(line_prices),
        confidence,
    )

    return ExtractionResult(
        image_decoded=True,
        levels_payload=levels_payload,
        current_price=current_price,
        num_lines_detected=len(lines),
        num_axis_points=len(axis_points),
        extraction_confidence=confidence,
        warning=warning,
        debug_info=_debug_info,
    )
