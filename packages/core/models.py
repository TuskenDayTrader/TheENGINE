"""
<<<<<<< HEAD
Canonical schema definitions for TheENGINE analysis pipeline.

AnalysisPayload  – input contract consumed by POST /analyze
AnalysisResult   – output contract returned by POST /analyze
LevelsPayload    – nested levels block inside AnalysisPayload
LevelDecision    – single level entry inside AnalysisResult bucket
=======
Canonical data models for TheENGINE analysis pipeline.

These models define the shared contract used by the scoring engine,
action-state logic, and output formatters.  PR1 will formalise the
schema with full Pydantic validation; this module provides the
dataclass-based internal interfaces until PR1 is merged.
>>>>>>> origin/main
"""

from __future__ import annotations

<<<<<<< HEAD
from enum import Enum
from typing import List

from pydantic import BaseModel, Field, field_validator, model_validator
=======
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
>>>>>>> origin/main


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ActionState(str, Enum):
<<<<<<< HEAD
=======
    """Directional bias produced by the action-state engine."""

>>>>>>> origin/main
    ACTIVE_LONG = "ACTIVE_LONG"
    ACTIVE_SHORT = "ACTIVE_SHORT"
    STAND_DOWN = "STAND_DOWN"


<<<<<<< HEAD
class Timeframe(str, Enum):
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    M60 = "60m"


# ---------------------------------------------------------------------------
# Levels sub-model
# ---------------------------------------------------------------------------


class LevelsPayload(BaseModel):
    """All key price levels required for scoring."""

    # Prior-day structure
    pdh: float = Field(..., description="Prior-day high")
    pdl: float = Field(..., description="Prior-day low")
    prior_settle: float = Field(..., description="Prior session settlement price")

    # RTH open
    rth_open: float = Field(..., description="Regular-trading-hours open")

    # Overnight / Globex
    globex_high: float = Field(..., description="Overnight (Globex) session high")
    globex_low: float = Field(..., description="Overnight (Globex) session low")

    # Asia session
    asia_high: float = Field(..., description="Asia session high")
    asia_low: float = Field(..., description="Asia session low")

    # London session
    london_high: float = Field(..., description="London session high")
    london_low: float = Field(..., description="London session low")

    # NY session
    ny_high: float = Field(..., description="NY session high")
    ny_low: float = Field(..., description="NY session low")

    # Initial-balance levels by session
    asia_ib_high: float = Field(..., description="Asia initial-balance high")
    asia_ib_low: float = Field(..., description="Asia initial-balance low")
    london_ib_high: float = Field(..., description="London initial-balance high")
    london_ib_low: float = Field(..., description="London initial-balance low")
    ny_ib_high: float = Field(..., description="NY initial-balance high")
    ny_ib_low: float = Field(..., description="NY initial-balance low")

    # Volatility reference
    atr14: float = Field(..., gt=0, description="14-period ATR (must be > 0)")

    @field_validator(
        "pdh", "pdl", "prior_settle", "rth_open",
        "globex_high", "globex_low",
        "asia_high", "asia_low",
        "london_high", "london_low",
        "ny_high", "ny_low",
        "asia_ib_high", "asia_ib_low",
        "london_ib_high", "london_ib_low",
        "ny_ib_high", "ny_ib_low",
        mode="before",
    )
    @classmethod
    def must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("price levels must be positive")
        return v

    @model_validator(mode="after")
    def high_must_exceed_low(self) -> "LevelsPayload":
        pairs = [
            ("pdh", "pdl"),
            ("globex_high", "globex_low"),
            ("asia_high", "asia_low"),
            ("london_high", "london_low"),
            ("ny_high", "ny_low"),
            ("asia_ib_high", "asia_ib_low"),
            ("london_ib_high", "london_ib_low"),
            ("ny_ib_high", "ny_ib_low"),
        ]
        for high_field, low_field in pairs:
            high = getattr(self, high_field)
            low = getattr(self, low_field)
            if high <= low:
                raise ValueError(
                    f"{high_field} ({high}) must be strictly greater than "
                    f"{low_field} ({low})"
                )
        return self


# ---------------------------------------------------------------------------
# Input payload
# ---------------------------------------------------------------------------


class AnalysisPayload(BaseModel):
    """Canonical input payload for POST /analyze."""

    date_et: str = Field(
        ...,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Analysis date in ET timezone (YYYY-MM-DD)",
    )
    ticker: str = Field(
        ...,
        min_length=1,
        max_length=20,
        description="Futures ticker symbol (e.g. NQU2026, ESU2026)",
    )
    timeframe: Timeframe = Field(
        ...,
        description="Analysis timeframe: 5m | 15m | 30m | 60m",
    )
    lookback_days: int = Field(
        ...,
        ge=1,
        le=365,
        description="Number of lookback days for context (1–365)",
    )
    current_price: float = Field(
        ...,
        gt=0,
        description="Current or reference price at analysis time",
    )
    levels: LevelsPayload = Field(
        ...,
        description="All required price levels for scoring",
    )
=======
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
>>>>>>> origin/main


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------


<<<<<<< HEAD
class LevelDecision(BaseModel):
    """A single scored level entry in the AnalysisResult."""

    label: str = Field(..., description="Human-readable level label (e.g. 'PDH')")
    price: float = Field(..., description="Price of this level")
    score: float = Field(..., description="Composite confluence score (higher = stronger)")
    distance_atr: float = Field(
        ...,
        description="Distance from current_price expressed in ATR units",
    )
    rationale: str = Field(..., description="Short explanation of why this level scored as it did")


class AnalysisResult(BaseModel):
    """Canonical output returned by POST /analyze."""

    ticker: str
    date_et: str
    timeframe: str
    current_price: float

    # 2 + 2 + 2 + 2 buckets
    strongest_resistance: List[LevelDecision] = Field(
        ...,
        min_length=2,
        max_length=2,
        description="Two highest-priority resistance levels (closest above current price)",
    )
    weakest_resistance: List[LevelDecision] = Field(
        ...,
        min_length=2,
        max_length=2,
        description="Two lower-priority resistance levels (farther above current price)",
    )
    strongest_support: List[LevelDecision] = Field(
        ...,
        min_length=2,
        max_length=2,
        description="Two highest-priority support levels (closest below current price)",
    )
    weakest_support: List[LevelDecision] = Field(
        ...,
        min_length=2,
        max_length=2,
        description="Two lower-priority support levels (farther below current price)",
    )

    action_state: ActionState = Field(
        ...,
        description="Directional bias: ACTIVE_LONG | ACTIVE_SHORT | STAND_DOWN",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Aggregate confidence score between 0 and 1",
    )
    rationale: str = Field(
        ...,
        description="Human-readable explanation of action_state and confidence",
    )
    poster_text: str = Field(
        ...,
        description="Branded SESSION CONFLUENCE MAP text block for dashboard graphics",
    )
=======
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
>>>>>>> origin/main
