"""
Action-state engine for TheENGINE.

Determines directional bias (ACTIVE_LONG / ACTIVE_SHORT / STAND_DOWN)
from the strongest scored resistance and support levels.

Conservative defaults
---------------------
PROXIMITY_THRESHOLD_ATR : float
    A level must be within this many ATR14 units of current price to
    trigger an active directional state.  Default 0.75 keeps the engine
    conservative – only levels that are *immediately* nearby matter.

SCORE_DOMINANCE_RATIO : float
    The stronger side's top score must exceed the weaker side's top score
    by at least this ratio before any active state is declared.  Default
    1.15 (15 % margin) prevents hair-trigger signals when both sides
    score similarly.

FALLBACK_PROXIMITY_PCT : float
    Used when ATR14 is unavailable.  Level must be within 0.5 % of
    current price.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from .models import ActionState, ConvictionTag, LevelDecision


# ---------------------------------------------------------------------------
# Configurable thresholds (conservative defaults)
# ---------------------------------------------------------------------------

PROXIMITY_THRESHOLD_ATR: float = 0.75
SCORE_DOMINANCE_RATIO: float = 1.15
FALLBACK_PROXIMITY_PCT: float = 0.005


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_action_state(
    current_price: float,
    strongest_resistance: List[LevelDecision],
    strongest_support: List[LevelDecision],
    atr14: Optional[float] = None,
) -> Tuple[ActionState, str]:
    """
    Derive action state from the top-scored resistance and support levels.

    Decision logic (applied in order)
    ----------------------------------
    1. If either bucket is empty → STAND_DOWN (insufficient data).
    2. If both top scores are within ``SCORE_DOMINANCE_RATIO`` of each
       other → STAND_DOWN (mixed signals).
    3. If nearest resistance is within proximity threshold **and**
       resistance score dominates → ACTIVE_SHORT.
    4. If nearest support is within proximity threshold **and**
       support score dominates → ACTIVE_LONG.
    5. Default → STAND_DOWN (no confirming proximity).

    Parameters
    ----------
    current_price:
        The price used as the reference point for proximity calculations.
    strongest_resistance:
        The two highest-conviction resistance ``LevelDecision`` objects.
    strongest_support:
        The two highest-conviction support ``LevelDecision`` objects.
    atr14:
        14-period ATR used for proximity normalisation.  When ``None``,
        a percentage-of-price fallback is used.

    Returns
    -------
    (ActionState, rationale_string)
    """

    # --- Guard: insufficient data -------------------------------------------
    if not strongest_resistance or not strongest_support:
        return (
            ActionState.STAND_DOWN,
            "Insufficient levels for active state determination.",
        )

    top_r = strongest_resistance[0]
    top_s = strongest_support[0]

    r_dist = abs(top_r.price - current_price)
    s_dist = abs(top_s.price - current_price)
    r_score = top_r.score
    s_score = top_s.score

    # --- Proximity thresholds -----------------------------------------------
    if atr14 and atr14 > 0:
        prox_threshold = PROXIMITY_THRESHOLD_ATR * atr14
    else:
        prox_threshold = current_price * FALLBACK_PROXIMITY_PCT

    r_near = r_dist <= prox_threshold
    s_near = s_dist <= prox_threshold

    # --- Mixed-signal check -------------------------------------------------
    if r_score > 0 and s_score > 0:
        high, low = (r_score, s_score) if r_score >= s_score else (s_score, r_score)
        ratio = high / low
        if ratio < SCORE_DOMINANCE_RATIO:
            return (
                ActionState.STAND_DOWN,
                (
                    f"Mixed signals: resistance score {r_score:.3f} vs "
                    f"support score {s_score:.3f} "
                    f"(ratio {ratio:.2f} < threshold {SCORE_DOMINANCE_RATIO:.2f}). "
                    "No dominant directional bias."
                ),
            )

    # --- Directional bias ---------------------------------------------------
    if r_near and r_score > s_score:
        return (
            ActionState.ACTIVE_SHORT,
            (
                f"Price within {r_dist:.2f} of strong resistance at "
                f"{top_r.price:.2f} (proximity threshold {prox_threshold:.2f}). "
                f"Resistance score {r_score:.3f} dominates support score {s_score:.3f}. "
                f"Bias: {top_r.trigger_note}."
            ),
        )

    if s_near and s_score > r_score:
        return (
            ActionState.ACTIVE_LONG,
            (
                f"Price within {s_dist:.2f} of strong support at "
                f"{top_s.price:.2f} (proximity threshold {prox_threshold:.2f}). "
                f"Support score {s_score:.3f} dominates resistance score {r_score:.3f}. "
                f"Bias: {top_s.trigger_note}."
            ),
        )

    # --- Default: no confirming proximity -----------------------------------
    return (
        ActionState.STAND_DOWN,
        (
            f"No dominant level in proximity. "
            f"Nearest resistance {top_r.price:.2f} ({r_dist:.2f} away), "
            f"nearest support {top_s.price:.2f} ({s_dist:.2f} away). "
            "Stand down until boundary acceptance or failure is confirmed."
        ),
    )
