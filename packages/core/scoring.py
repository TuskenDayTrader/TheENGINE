"""
<<<<<<< HEAD
2+2+2+2 confluence scoring engine.

Business rules
--------------
All named price levels from LevelsPayload are classified as either
resistance (above current_price) or support (below current_price).

Each level receives a composite score based on:
  1. Proximity weight   – levels closer to current_price score higher
  2. Category weight    – prior-day and settle levels outrank IB/session extremes
  3. Confluence bonus   – levels within 0.1 ATR of each other earn a bonus

The scored lists are sorted and sliced into four buckets:
  strongest_resistance  – top-2 closest resistance levels
  weakest_resistance    – next-2 resistance levels
  strongest_support     – top-2 closest support levels
  weakest_support       – next-2 support levels

ActionState is determined by the structural position of current_price
relative to the closest resistance and support:
  ACTIVE_LONG   – price cleared PDH or prior_settle with clear air above
  ACTIVE_SHORT  – price broke below PDL or prior_settle with clear air below
  STAND_DOWN    – price is sandwiched or signals are mixed
=======
Deterministic 2+2+2+2 scoring engine for TheENGINE.

Given an ``AnalysisPayload`` the engine:

1. Extracts every named price level from ``LevelsPayload``.
2. Groups confluent levels (within ``CONFLUENCE_THRESHOLD_ATR_FRACTION``
   of ATR14, or a percentage fallback) into a single *level group*.
3. Scores each group using a weighted composite of:
   - confluence count (stacked sources)
   - session relevance (NY > London > Globex > Asia > prior-day)
   - recency weighting  (same ordering)
   - proximity to current price (normalised by ATR14 when available)
4. Classifies groups as **resistance** (above current price) or
   **support** (below current price).
5. Applies deterministic tie-break sort: score → confluence → recency →
   price proximity direction.
6. Returns exactly **2 strongest** and **2 weakest** items per side.

Default weight rationale
------------------------
+--------------------+--------+----------------------------------------------+
| Factor             | Weight | Rationale                                    |
+====================+========+==============================================+
| Confluence count   |  0.40  | Most reliable signal: multiple independent   |
|                    |        | sources converging on the same price zone    |
|                    |        | dramatically increases probability of a      |
|                    |        | meaningful reaction.                         |
+--------------------+--------+----------------------------------------------+
| Session relevance  |  0.25  | Not all sessions carry equal institutional  |
|                    |        | weight.  NY (primary) > London > Globex >   |
|                    |        | Asia > prior-day reference.                  |
+--------------------+--------+----------------------------------------------+
| Recency weighting  |  0.20  | Levels formed in the most recent session    |
|                    |        | have had less time to be absorbed by the    |
|                    |        | market and therefore retain their edge.     |
+--------------------+--------+----------------------------------------------+
| Proximity          |  0.15  | Levels closest to current price have the    |
|                    |        | highest *immediate* relevance; distant      |
|                    |        | levels act as targets, not triggers.        |
+--------------------+--------+----------------------------------------------+
>>>>>>> origin/main
"""

from __future__ import annotations

<<<<<<< HEAD
import math
from typing import Dict, List, Tuple

from .models import (
    ActionState,
    AnalysisPayload,
    LevelDecision,
)

# ---------------------------------------------------------------------------
# Level metadata – category weights (higher = more institutionally significant)
# ---------------------------------------------------------------------------

_LEVEL_WEIGHTS: Dict[str, float] = {
    "PDH": 1.0,
    "PDL": 1.0,
    "Prior Settle": 0.95,
    "RTH Open": 0.85,
    "Globex High": 0.80,
    "Globex Low": 0.80,
    "Asia High": 0.65,
    "Asia Low": 0.65,
    "London High": 0.70,
    "London Low": 0.70,
    "NY High": 0.75,
    "NY Low": 0.75,
    "Asia IB High": 0.60,
    "Asia IB Low": 0.60,
    "London IB High": 0.65,
    "London IB Low": 0.65,
    "NY IB High": 0.70,
    "NY IB Low": 0.70,
}

# Confluence proximity threshold (fraction of ATR)
_CONFLUENCE_ATR_THRESHOLD = 0.10
_CONFLUENCE_BONUS = 0.15

# Minimum distinct levels required per side to build all four buckets
_MIN_LEVELS_PER_SIDE = 4


def _extract_named_levels(payload: AnalysisPayload) -> List[Tuple[str, float]]:
    """Return (label, price) pairs for every level in the payload."""
    lvl = payload.levels
    return [
        ("PDH", lvl.pdh),
        ("PDL", lvl.pdl),
        ("Prior Settle", lvl.prior_settle),
        ("RTH Open", lvl.rth_open),
        ("Globex High", lvl.globex_high),
        ("Globex Low", lvl.globex_low),
        ("Asia High", lvl.asia_high),
        ("Asia Low", lvl.asia_low),
        ("London High", lvl.london_high),
        ("London Low", lvl.london_low),
        ("NY High", lvl.ny_high),
        ("NY Low", lvl.ny_low),
        ("Asia IB High", lvl.asia_ib_high),
        ("Asia IB Low", lvl.asia_ib_low),
        ("London IB High", lvl.london_ib_high),
        ("London IB Low", lvl.london_ib_low),
        ("NY IB High", lvl.ny_ib_high),
        ("NY IB Low", lvl.ny_ib_low),
    ]


def _add_confluence_bonuses(
    scored: List[Tuple[str, float, float, float]],
    atr: float,
) -> List[Tuple[str, float, float, float]]:
    """
    Boost the score of levels that cluster within _CONFLUENCE_ATR_THRESHOLD ATR
    of at least one other level in the same list.

    Input/output tuple: (label, price, raw_distance_atr, score)
    """
    threshold = _CONFLUENCE_ATR_THRESHOLD * atr
    result = list(scored)
    boosted: set[int] = set()
    for i, (label_i, price_i, dist_i, score_i) in enumerate(result):
        if i in boosted:
            continue
        for j, (label_j, price_j, _dist_j, _score_j) in enumerate(result):
            if i == j:
                continue
            if abs(price_i - price_j) <= threshold:
                result[i] = (label_i, price_i, dist_i, score_i + _CONFLUENCE_BONUS)
                boosted.add(i)
                break
    return result


def _score_levels(
    levels: List[Tuple[str, float]],
    current_price: float,
    atr: float,
    above: bool,
) -> List[Tuple[str, float, float, float]]:
    """
    Filter levels to one side of current_price, compute scores, apply
    confluence bonuses, and return sorted list.

    Returns list of (label, price, distance_atr, score) sorted by score desc.
    """
    side_levels: List[Tuple[str, float]] = [
        (lbl, px)
        for lbl, px in levels
        if (px > current_price) == above and not math.isclose(px, current_price, rel_tol=1e-6)
    ]

    if not side_levels:
        return []

    # Proximity score: 1 / (1 + distance_in_atr_units)
    scored: List[Tuple[str, float, float, float]] = []
    for label, price in side_levels:
        dist_atr = abs(price - current_price) / atr
        proximity_score = 1.0 / (1.0 + dist_atr)
        category_weight = _LEVEL_WEIGHTS.get(label, 0.5)
        composite = proximity_score * category_weight
        scored.append((label, price, dist_atr, composite))

    # Apply confluence bonuses
    scored = _add_confluence_bonuses(scored, atr)

    # Sort by score descending (strongest first)
    scored.sort(key=lambda x: x[3], reverse=True)
    return scored


def _pad_levels(
    scored: List[Tuple[str, float, float, float]],
    current_price: float,
    atr: float,
    above: bool,
    needed: int,
) -> List[Tuple[str, float, float, float]]:
    """
    If there are fewer distinct scored levels than 'needed', synthesise
    placeholder entries offset by 0.5 ATR increments so the response
    contract is always satisfied.
    """
    result = list(scored)
    idx = len(result) + 1
    while len(result) < needed:
        offset = 0.5 * idx * atr
        price = current_price + offset if above else current_price - offset
        dist_atr = abs(price - current_price) / atr
        result.append((f"Synthetic-{idx}", price, dist_atr, 0.01))
        idx += 1
    return result


def _make_level_decision(
    label: str,
    price: float,
    dist_atr: float,
    score: float,
    current_price: float,
    atr: float,
) -> LevelDecision:
    side = "above" if price > current_price else "below"
    return LevelDecision(
        label=label,
        price=round(price, 4),
        score=round(score, 4),
        distance_atr=round(dist_atr, 4),
        rationale=(
            f"{label} is {dist_atr:.2f} ATR {side} current price "
            f"(score {score:.3f}; ATR={atr:.2f})"
        ),
    )


def _determine_action_state(
    payload: AnalysisPayload,
    resistance_scored: List[Tuple[str, float, float, float]],
    support_scored: List[Tuple[str, float, float, float]],
) -> Tuple[ActionState, float, str]:
    """
    Determine ActionState, aggregate confidence, and rationale string.

    Rules (conservative):
    - ACTIVE_LONG  : price > PDH or price > Prior Settle AND nearest
                     resistance is ≥ 1.0 ATR away AND nearest support ≤ 0.5 ATR
    - ACTIVE_SHORT : price < PDL or price < Prior Settle AND nearest
                     support is ≥ 1.0 ATR away AND nearest resistance ≤ 0.5 ATR
    - STAND_DOWN   : all other conditions (sandwiched, conflicting signals)
    """
    lvl = payload.levels
    cp = payload.current_price
    atr = lvl.atr14

    near_res_dist = resistance_scored[0][2] if resistance_scored else float("inf")
    near_sup_dist = support_scored[0][2] if support_scored else float("inf")

    above_pdh = cp > lvl.pdh
    above_settle = cp > lvl.prior_settle
    below_pdl = cp < lvl.pdl
    below_settle = cp < lvl.prior_settle

    long_structure = (above_pdh or above_settle) and near_res_dist >= 1.0 and near_sup_dist <= 0.75
    short_structure = (below_pdl or below_settle) and near_sup_dist >= 1.0 and near_res_dist <= 0.75

    if long_structure and not short_structure:
        state = ActionState.ACTIVE_LONG
        conf = min(1.0, 0.5 + (near_res_dist - 1.0) * 0.1)
        rationale = (
            f"Price ({cp:.2f}) is {'above PDH' if above_pdh else 'above Prior Settle'}. "
            f"Nearest resistance {near_res_dist:.2f} ATR away. "
            f"Nearest support {near_sup_dist:.2f} ATR away. Structure favours longs."
        )
    elif short_structure and not long_structure:
        state = ActionState.ACTIVE_SHORT
        conf = min(1.0, 0.5 + (near_sup_dist - 1.0) * 0.1)
        rationale = (
            f"Price ({cp:.2f}) is {'below PDL' if below_pdl else 'below Prior Settle'}. "
            f"Nearest support {near_sup_dist:.2f} ATR away. "
            f"Nearest resistance {near_res_dist:.2f} ATR away. Structure favours shorts."
        )
    else:
        state = ActionState.STAND_DOWN
        conf = max(0.0, 0.5 - abs(near_res_dist - near_sup_dist) * 0.05)
        rationale = (
            f"Price ({cp:.2f}) is sandwiched between support ({near_sup_dist:.2f} ATR) "
            f"and resistance ({near_res_dist:.2f} ATR). Mixed or insufficient structure – "
            f"stand down until a key level is cleared."
        )

    return state, round(conf, 4), rationale
=======
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .action_state import compute_action_state
from .models import (
    ActionState,
    AnalysisPayload,
    AnalysisResult,
    ConvictionTag,
    LevelDecision,
    LevelsPayload,
)

# ---------------------------------------------------------------------------
# Weights (conservative defaults — see module docstring for rationale)
# ---------------------------------------------------------------------------

CONFLUENCE_WEIGHT: float = 0.40
SESSION_WEIGHT: float = 0.25
RECENCY_WEIGHT: float = 0.20
PROXIMITY_WEIGHT: float = 0.15

# Session relevance scores (0.0–1.0)
SESSION_SCORES: Dict[str, float] = {
    "ny": 1.00,
    "london": 0.85,
    "globex": 0.70,
    "asia": 0.65,
    "prior_day": 0.50,
    "neutral": 0.55,
}

# Recency scores — same taxonomy as session but independent weighting
RECENCY_SCORES: Dict[str, float] = {
    "ny": 1.00,
    "london": 0.85,
    "globex": 0.80,
    "asia": 0.70,
    "prior_day": 0.50,
    "neutral": 0.55,
}

# Confluence grouping: group levels within this fraction of ATR14
CONFLUENCE_THRESHOLD_ATR_FRACTION: float = 0.25
# Fallback when ATR14 is absent: fraction of current price
CONFLUENCE_THRESHOLD_PCT: float = 0.001   # 0.1 %

# Maximum confluence count used for normalisation
MAX_CONFLUENCE: int = 6

# Map each named level field → session category
_SOURCE_SESSION: Dict[str, str] = {
    "pdh": "prior_day",
    "pdl": "prior_day",
    "prior_settle": "prior_day",
    "rth_open": "neutral",
    "globex_high": "globex",
    "globex_low": "globex",
    "asia_high": "asia",
    "asia_low": "asia",
    "london_high": "london",
    "london_low": "london",
    "ny_high": "ny",
    "ny_low": "ny",
    "asia_ib_high": "asia",
    "asia_ib_low": "asia",
    "london_ib_high": "london",
    "london_ib_low": "london",
    "ny_ib_high": "ny",
    "ny_ib_low": "ny",
}


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------


@dataclass
class _RawLevel:
    source: str
    price: float


@dataclass
class _LevelGroup:
    representative_price: float
    sources: List[str]          # sorted for determinism
    best_session: str
    best_recency: str


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _extract_raw_levels(levels: LevelsPayload) -> List[_RawLevel]:
    """Return all non-None, positive named levels from the payload."""
    raw: List[_RawLevel] = []
    for source in _SOURCE_SESSION:
        price = getattr(levels, source, None)
        if price is not None and price > 0:
            raw.append(_RawLevel(source=source, price=price))
    return raw


def _confluence_threshold(levels: LevelsPayload, current_price: float) -> float:
    """Price-distance threshold for grouping confluent levels."""
    if levels.atr14 and levels.atr14 > 0:
        return levels.atr14 * CONFLUENCE_THRESHOLD_ATR_FRACTION
    return current_price * CONFLUENCE_THRESHOLD_PCT


def _group_confluent_levels(
    raw_levels: List[_RawLevel], threshold: float
) -> List[_LevelGroup]:
    """
    Deterministically group levels that fall within ``threshold`` of each
    other in price.

    Algorithm:
    - Sort by (price, source) for a stable, reproducible ordering.
    - Greedy pass: the first unvisited level seeds a new group; all
      subsequent unvisited levels within ``threshold`` of that seed join
      the group.
    - Representative price = mean of grouped prices.
    - Best session / recency = the category with the highest score among
      all sources in the group.
    """
    if not raw_levels:
        return []

    sorted_levels = sorted(raw_levels, key=lambda x: (x.price, x.source))
    used = [False] * len(sorted_levels)
    groups: List[_LevelGroup] = []

    for i, seed in enumerate(sorted_levels):
        if used[i]:
            continue
        sources = [seed.source]
        prices = [seed.price]
        used[i] = True

        for j in range(i + 1, len(sorted_levels)):
            if used[j]:
                continue
            if abs(sorted_levels[j].price - seed.price) <= threshold:
                sources.append(sorted_levels[j].source)
                prices.append(sorted_levels[j].price)
                used[j] = True

        rep_price = sum(prices) / len(prices)
        best_session = max(
            (_SOURCE_SESSION.get(s, "neutral") for s in sources),
            key=lambda c: SESSION_SCORES.get(c, 0.0),
        )
        best_recency = max(
            (_SOURCE_SESSION.get(s, "neutral") for s in sources),
            key=lambda c: RECENCY_SCORES.get(c, 0.0),
        )
        groups.append(
            _LevelGroup(
                representative_price=round(rep_price, 4),
                sources=sorted(sources),
                best_session=best_session,
                best_recency=best_recency,
            )
        )

    return groups


def _score_group(
    group: _LevelGroup,
    current_price: float,
    atr14: Optional[float],
) -> float:
    """
    Composite score for a level group; result is in [0, 1].

    score = CONFLUENCE_WEIGHT * confluence_sub
          + SESSION_WEIGHT    * session_sub
          + RECENCY_WEIGHT    * recency_sub
          + PROXIMITY_WEIGHT  * proximity_sub
    """
    confluence_sub = min(len(group.sources), MAX_CONFLUENCE) / MAX_CONFLUENCE
    session_sub = SESSION_SCORES.get(group.best_session, 0.5)
    recency_sub = RECENCY_SCORES.get(group.best_recency, 0.5)

    distance = abs(group.representative_price - current_price)
    if atr14 and atr14 > 0:
        proximity_sub = 1.0 / (1.0 + distance / atr14)
    else:
        denom = max(current_price * 0.001, 1.0)
        proximity_sub = 1.0 / (1.0 + distance / denom)

    return (
        CONFLUENCE_WEIGHT * confluence_sub
        + SESSION_WEIGHT * session_sub
        + RECENCY_WEIGHT * recency_sub
        + PROXIMITY_WEIGHT * proximity_sub
    )


def _conviction_tag(score: float) -> ConvictionTag:
    if score >= 0.70:
        return ConvictionTag.HIGH
    if score >= 0.45:
        return ConvictionTag.MODERATE
    return ConvictionTag.LOW


def _trigger_note(group: _LevelGroup, level_type: str, conviction: ConvictionTag) -> str:
    source_str = "+".join(group.sources)
    return (
        f"{conviction.value} {level_type}: {source_str} "
        f"@ {group.representative_price:.2f}"
    )


def _build_level_decision(
    group: _LevelGroup, level_type: str, score: float
) -> LevelDecision:
    conviction = _conviction_tag(score)
    return LevelDecision(
        price=group.representative_price,
        sources=list(group.sources),
        level_type=level_type,
        conviction=conviction,
        score=round(score, 6),
        trigger_note=_trigger_note(group, level_type, conviction),
    )


# ---------------------------------------------------------------------------
# Tie-break sort keys
# Tie-break order: score ↓ → confluence count ↓ → recency ↓ → price (direction-aware)
# ---------------------------------------------------------------------------

def _sort_key_resistance(item: Tuple[_LevelGroup, float]) -> tuple:
    """Sort resistance groups: highest score first; ties broken deterministically."""
    group, score = item
    return (
        -score,
        -len(group.sources),
        -RECENCY_SCORES.get(group.best_recency, 0.0),
        group.representative_price,   # lower price = closer overhead = higher priority
    )


def _sort_key_support(item: Tuple[_LevelGroup, float]) -> tuple:
    """Sort support groups: highest score first; ties broken deterministically."""
    group, score = item
    return (
        -score,
        -len(group.sources),
        -RECENCY_SCORES.get(group.best_recency, 0.0),
        -group.representative_price,  # higher price = closer below = higher priority
    )


def _select_buckets(
    sorted_pairs: List[Tuple[_LevelGroup, float]],
    level_type: str,
) -> Tuple[List[LevelDecision], List[LevelDecision]]:
    """
    Return (strongest_2, weakest_2) from a scored, sorted list.

    With ≥ 4 items: strongest = first 2, weakest = last 2.
    With 3 items:   strongest = first 2, weakest = last 2 (overlaps at index 1).
    With 2 items:   strongest = weakest = both items.
    With 1 item:    both buckets contain that item twice (duplicated).
    With 0 items:   empty list returned (caller pads).
    """
    n = len(sorted_pairs)
    if n == 0:
        return [], []

    decisions = [_build_level_decision(g, level_type, s) for g, s in sorted_pairs]

    if n == 1:
        return [decisions[0], decisions[0]], [decisions[0], decisions[0]]

    strongest = decisions[:2]

    if n <= 3:
        weakest = decisions[-2:]
    else:
        weakest = decisions[-2:]

    # Guarantee exactly 2 items in each bucket
    while len(strongest) < 2:
        strongest.append(strongest[-1])
    while len(weakest) < 2:
        weakest.append(weakest[-1])

    return strongest[:2], weakest[:2]
>>>>>>> origin/main


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


<<<<<<< HEAD
def score(payload: AnalysisPayload) -> dict:
    """
    Run the 2+2+2+2 scoring engine on the provided payload.

    Returns a dict with keys:
      strongest_resistance, weakest_resistance,
      strongest_support,    weakest_support,
      action_state, confidence, rationale
    """
    named = _extract_named_levels(payload)
    cp = payload.current_price
    atr = payload.levels.atr14

    res_scored = _score_levels(named, cp, atr, above=True)
    sup_scored = _score_levels(named, cp, atr, above=False)

    # Ensure enough levels in each list for 2 strongest + 2 weakest
    res_scored = _pad_levels(res_scored, cp, atr, above=True, needed=_MIN_LEVELS_PER_SIDE)
    sup_scored = _pad_levels(sup_scored, cp, atr, above=False, needed=_MIN_LEVELS_PER_SIDE)

    def to_decisions(entries: List[Tuple[str, float, float, float]]) -> List[LevelDecision]:
        return [_make_level_decision(lbl, px, da, sc, cp, atr) for lbl, px, da, sc in entries]

    strongest_res = to_decisions(res_scored[:2])
    weakest_res = to_decisions(res_scored[2:4])
    strongest_sup = to_decisions(sup_scored[:2])
    weakest_sup = to_decisions(sup_scored[2:4])

    action_state, confidence, rationale = _determine_action_state(
        payload, res_scored, sup_scored
    )

    return {
        "strongest_resistance": strongest_res,
        "weakest_resistance": weakest_res,
        "strongest_support": strongest_sup,
        "weakest_support": weakest_sup,
        "action_state": action_state,
        "confidence": confidence,
        "rationale": rationale,
    }
=======
def score(payload: AnalysisPayload) -> AnalysisResult:
    """
    Score an ``AnalysisPayload`` and return a deterministic ``AnalysisResult``.

    Guarantees
    ----------
    - ``strongest_resistance``, ``weakest_resistance``,
      ``strongest_support``, and ``weakest_support`` each contain
      **exactly 2** ``LevelDecision`` objects.
    - Same input always produces the same ordered output (determinism).

    Parameters
    ----------
    payload:
        A fully-populated ``AnalysisPayload``.  At minimum ``pdh``,
        ``pdl``, and ``prior_settle`` must be non-zero positive values.

    Returns
    -------
    AnalysisResult
    """
    current_price = payload.current_price
    levels = payload.levels
    atr14 = levels.atr14

    raw_levels = _extract_raw_levels(levels)
    threshold = _confluence_threshold(levels, current_price)
    groups = _group_confluent_levels(raw_levels, threshold)

    resistance_pairs: List[Tuple[_LevelGroup, float]] = []
    support_pairs: List[Tuple[_LevelGroup, float]] = []

    for group in groups:
        s = _score_group(group, current_price, atr14)
        if group.representative_price > current_price:
            resistance_pairs.append((group, s))
        elif group.representative_price < current_price:
            support_pairs.append((group, s))
        # Levels exactly at current_price are ambiguous — excluded

    # Deterministic sort with tie-break rules
    resistance_sorted = sorted(resistance_pairs, key=_sort_key_resistance)
    support_sorted = sorted(support_pairs, key=_sort_key_support)

    strongest_r, weakest_r = _select_buckets(resistance_sorted, "resistance")
    strongest_s, weakest_s = _select_buckets(support_sorted, "support")

    # Action state
    action_state, rationale = compute_action_state(
        current_price=current_price,
        strongest_resistance=strongest_r,
        strongest_support=strongest_s,
        atr14=atr14,
    )

    # Overall confidence = conviction of the stronger top level
    top_scores = [d.score for d in (strongest_r + strongest_s) if d.score > 0]
    avg_top = sum(top_scores) / len(top_scores) if top_scores else 0.0
    overall_confidence = _conviction_tag(avg_top)

    return AnalysisResult(
        ticker=payload.ticker,
        date_et=payload.date_et,
        strongest_resistance=strongest_r,
        weakest_resistance=weakest_r,
        strongest_support=strongest_s,
        weakest_support=weakest_s,
        action_state=action_state,
        confidence=overall_confidence,
        rationale=rationale,
    )
>>>>>>> origin/main
