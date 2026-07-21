"""
Canonical data models for TheENGINE analysis pipeline.

These models define the shared contract used by the scoring engine,
action-state logic, and output formatters.  PR1 will formalise the
schema with full Pydantic validation; this module provides the
dataclass-based internal interfaces until PR1 is merged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ActionState(str, Enum):
    """Directional bias produced by the action-state engine."""

    ACTIVE_LONG = "ACTIVE_LONG"
    ACTIVE_SHORT = "ACTIVE_SHORT"
    STAND_DOWN = "STAND_DOWN"


class ConvictionTag(str, Enum):
    """Confidence/conviction label attached to a scored level."""

    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"


# ---------------------------------------------------------------------------
# Input models
# ---------------------------------------------------------------------------


@dataclass
class LevelsPayload:
    """
    Named price levels for a single analysis session.

    All optional fields default to ``None``; missing levels are excluded
    from scoring.  ``atr14`` is used for confluence-threshold and
    proximity-score normalisation when provided.
    """

    pdh: float
    pdl: float
    prior_settle: float

    rth_open: Optional[float] = None

    globex_high: Optional[float] = None
    globex_low: Optional[float] = None

    asia_high: Optional[float] = None
    asia_low: Optional[float] = None

    london_high: Optional[float] = None
    london_low: Optional[float] = None

    ny_high: Optional[float] = None
    ny_low: Optional[float] = None

    asia_ib_high: Optional[float] = None
    asia_ib_low: Optional[float] = None

    london_ib_high: Optional[float] = None
    london_ib_low: Optional[float] = None

    ny_ib_high: Optional[float] = None
    ny_ib_low: Optional[float] = None

    atr14: Optional[float] = None


@dataclass
class AnalysisPayload:
    """Top-level input payload for a single ticker analysis."""

    date_et: str        # "YYYY-MM-DD" in US Eastern time
    ticker: str         # e.g. "NQU2026", "ESU2026"
    timeframe: str      # e.g. "30m", "1h"
    lookback_days: int
    current_price: float
    levels: LevelsPayload


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------


@dataclass
class LevelDecision:
    """A single scored level selected for inclusion in the 2+2+2+2 output."""

    price: float                # representative price of the level / confluent zone
    sources: List[str]          # named sources, e.g. ["pdh", "ny_high"]
    level_type: str             # "resistance" or "support"
    conviction: ConvictionTag
    score: float                # composite score in [0, 1]
    trigger_note: str           # short human-readable rationale


@dataclass
class AnalysisResult:
    """
    Full 2+2+2+2 output for a single ticker.

    Each bucket contains **exactly 2** ``LevelDecision`` items.
    """

    ticker: str
    date_et: str

    strongest_resistance: List[LevelDecision]   # 2 highest-conviction resistance
    weakest_resistance: List[LevelDecision]     # 2 lowest-conviction resistance
    strongest_support: List[LevelDecision]      # 2 highest-conviction support
    weakest_support: List[LevelDecision]        # 2 lowest-conviction support

    action_state: ActionState
    confidence: ConvictionTag
    rationale: str
