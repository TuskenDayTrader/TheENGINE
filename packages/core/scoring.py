"""
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
"""

from __future__ import annotations

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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


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
