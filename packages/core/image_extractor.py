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
# ---------------------------------------------------------------------------

# Green #089981  → BGR(129, 153, 8) → HSV ≈ (85, 242, 153)
_GREEN_LOWER: Tuple[int, int, int] = (75, 100, 80)
_GREEN_UPPER: Tuple[int, int, int] = (100, 255, 255)

# Red #f23645 → BGR(69, 54, 242) → HSV ≈ (178, 200, 242) and wraps to (0-5)
_RED_LOWER_A: Tuple[int, int, int] = (0, 80, 100)
_RED_UPPER_A: Tuple[int, int, int] = (15, 255, 255)
_RED_LOWER_B: Tuple[int, int, int] = (165, 80, 100)
_RED_UPPER_B: Tuple[int, int, int] = (180, 255, 255)

# Minimum fraction of image width a colour span must cover to be a "line"
_LINE_MIN_WIDTH_FRACTION: float = 0.10

# Rightmost fraction of the image treated as the price axis
_PRICE_AXIS_FRACTION: float = 0.15

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


def _detect_horizontal_lines(
    img: "np.ndarray",
) -> List[Tuple[int, str]]:
    """
    Detect coloured horizontal lines in a TradingView dark-theme chart image.

    Parameters
    ----------
    img:
        BGR image array from ``cv2.imdecode``.

    Returns
    -------
    list of (y_pixel, color_label) tuples, where ``color_label`` is
    ``"green"`` or ``"red"``.
    """
    h, w = img.shape[:2]
    min_line_width = max(10, int(w * _LINE_MIN_WIDTH_FRACTION))
    max_line_height = max(5, h // 100)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    green_mask = cv2.inRange(
        hsv,
        np.array(_GREEN_LOWER, dtype=np.uint8),
        np.array(_GREEN_UPPER, dtype=np.uint8),
    )

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
    red_mask = cv2.bitwise_or(red_mask_a, red_mask_b)

    # Horizontal dilation kernel to bridge short gaps within a line segment
    h_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (min_line_width // 2 + 1, 1)
    )

    results: List[Tuple[int, str]] = []

    for mask, color_label in [(green_mask, "green"), (red_mask, "red")]:
        dilated = cv2.dilate(mask, h_kernel, iterations=1)
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if cw >= min_line_width and ch <= max_line_height:
                center_y = y + ch // 2
                results.append((center_y, color_label))

    return _deduplicate_lines(results)


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

    # Step 1: Parse price axis ────────────────────────────────────────────────
    axis_points = _parse_price_axis(img)

    # Step 2: Detect horizontal lines ─────────────────────────────────────────
    lines = _detect_horizontal_lines(img)

    # Step 3: Map line y-positions to prices ──────────────────────────────────
    line_prices: List[float] = []
    for y, _color in lines:
        price = _map_y_to_price(y, axis_points)
        if price is not None and price > 0:
            line_prices.append(round(price, 4))

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
    )
