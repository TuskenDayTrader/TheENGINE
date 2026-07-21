"""
Canonical data models for TheENGINE analysis pipeline.

AnalysisPayload — the input contract every module consumes.
AnalysisResult  — the output contract every module produces.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, List

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Payload — input side
# ---------------------------------------------------------------------------


class LevelsPayload(BaseModel):
    """All price levels required for confluence scoring."""

    pdh: float = Field(..., description="Prior day high")
    pdl: float = Field(..., description="Prior day low")
    prior_settle: float = Field(..., description="Prior session settlement price")
    rth_open: float = Field(..., description="Regular trading hours open")

    globex_high: float = Field(..., description="Globex (overnight) session high")
    globex_low: float = Field(..., description="Globex (overnight) session low")

    asia_high: float = Field(..., description="Asia session high")
    asia_low: float = Field(..., description="Asia session low")

    london_high: float = Field(..., description="London session high")
    london_low: float = Field(..., description="London session low")

    ny_high: float = Field(..., description="New York session high")
    ny_low: float = Field(..., description="New York session low")

    asia_ib_high: float = Field(..., description="Asia initial balance high")
    asia_ib_low: float = Field(..., description="Asia initial balance low")

    london_ib_high: float = Field(..., description="London initial balance high")
    london_ib_low: float = Field(..., description="London initial balance low")

    ny_ib_high: float = Field(..., description="New York initial balance high")
    ny_ib_low: float = Field(..., description="New York initial balance low")

    atr14: float = Field(..., gt=0, description="14-period ATR (must be positive)")

    @model_validator(mode="after")
    def _validate_high_low_pairs(self) -> "LevelsPayload":
        """Every H/L pair must have high >= low."""
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
        errors: list[str] = []
        for high_field, low_field in pairs:
            high = getattr(self, high_field)
            low = getattr(self, low_field)
            if high < low:
                errors.append(
                    f"levels.{high_field} ({high}) must be >= levels.{low_field} ({low})"
                )
        if errors:
            raise ValueError("; ".join(errors))
        return self


class AnalysisPayload(BaseModel):
    """Canonical input payload for a single-ticker analysis request."""

    date_et: str = Field(
        ...,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Analysis date in ET timezone (YYYY-MM-DD)",
    )
    ticker: str = Field(..., min_length=1, description="Instrument symbol (e.g. NQU2026)")
    timeframe: str = Field(
        ...,
        min_length=1,
        description="Chart timeframe (e.g. 30m, 1h, 1d)",
    )
    lookback_days: int = Field(
        ...,
        ge=1,
        description="Number of calendar days of history to include (>= 1)",
    )
    current_price: float = Field(
        ...,
        gt=0,
        description="Current market price at time of analysis (must be positive)",
    )
    levels: LevelsPayload = Field(..., description="All required price levels")

    @field_validator("ticker", "timeframe", mode="before")
    @classmethod
    def _strip_and_nonempty(cls, v: object) -> str:
        if not isinstance(v, str):
            raise ValueError("must be a string")
        stripped = v.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped


# ---------------------------------------------------------------------------
# Result — output side
# ---------------------------------------------------------------------------


class ActionState(str, Enum):
    """Directional bias for the current session."""

    ACTIVE_LONG = "ACTIVE_LONG"
    ACTIVE_SHORT = "ACTIVE_SHORT"
    STAND_DOWN = "STAND_DOWN"


class LevelItem(BaseModel):
    """A single resolved support or resistance decision."""

    level: Annotated[
        str,
        Field(description="Price level or range string (e.g. '29195-29205' or '29195.25')"),
    ]
    source: str = Field(
        ..., description="Origin label (e.g. 'PDH', 'Globex High', 'NY IB High')"
    )
    conviction: str = Field(
        ...,
        description="Conviction/confidence tag (e.g. 'HIGH', 'MEDIUM', 'LOW')",
    )
    trigger_note: str = Field(
        ...,
        description="Brief note on trigger conditions or invalidation scenario",
    )


class AnalysisResult(BaseModel):
    """Canonical 2+2+2+2 output produced by the scoring engine."""

    strongest_resistance: Annotated[
        List[LevelItem],
        Field(min_length=2, max_length=2, description="Two levels most likely to reject price"),
    ]
    weakest_resistance: Annotated[
        List[LevelItem],
        Field(
            min_length=2,
            max_length=2,
            description="Two resistance levels most likely to break up through",
        ),
    ]
    strongest_support: Annotated[
        List[LevelItem],
        Field(min_length=2, max_length=2, description="Two levels most likely to bounce price"),
    ]
    weakest_support: Annotated[
        List[LevelItem],
        Field(
            min_length=2,
            max_length=2,
            description="Two support levels most likely to fail down through",
        ),
    ]
    action_state: ActionState = Field(
        ..., description="Session directional bias"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Aggregate confidence score in [0.0, 1.0]",
    )
    rationale: List[str] = Field(
        ...,
        min_length=1,
        description="Ordered list of reasoning strings supporting this result",
    )
