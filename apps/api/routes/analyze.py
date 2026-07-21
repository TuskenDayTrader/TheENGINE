from __future__ import annotations

import datetime as dt
import logging
from enum import Enum

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from packages.core.models import AnalysisPayload, AnalysisResult, ConvictionTag, LevelDecision, LevelsPayload
from packages.core.output import build_poster
from packages.core.scoring import score

logger = logging.getLogger(__name__)
router = APIRouter()


class Timeframe(str, Enum):
    M30 = "30m"
    H1 = "1h"


class LevelsInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    pdh: float
    pdl: float
    prior_settle: float
    rth_open: float | None = None
    globex_high: float | None = None
    globex_low: float | None = None
    asia_high: float | None = None
    asia_low: float | None = None
    london_high: float | None = None
    london_low: float | None = None
    ny_high: float | None = None
    ny_low: float | None = None
    asia_ib_high: float | None = None
    asia_ib_low: float | None = None
    london_ib_high: float | None = None
    london_ib_low: float | None = None
    ny_ib_high: float | None = None
    ny_ib_low: float | None = None
    atr14: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_high_low(self) -> "LevelsInput":
        if self.pdh < self.pdl:
            raise ValueError("pdh must be greater than or equal to pdl")
        return self


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    date_et: str
    ticker: str
    timeframe: Timeframe
    lookback_days: int = Field(ge=1)
    current_price: float = Field(gt=0)
    levels: LevelsInput

    @field_validator("date_et")
    @classmethod
    def validate_date(cls, value: str) -> str:
        dt.datetime.strptime(value, "%Y-%m-%d")
        return value


class ResponseLevel(BaseModel):
    label: str
    price: float
    score: float
    distance_atr: float | None
    rationale: str


class AnalyzeResponse(BaseModel):
    ticker: str
    date_et: str
    strongest_resistance: list[ResponseLevel]
    weakest_resistance: list[ResponseLevel]
    strongest_support: list[ResponseLevel]
    weakest_support: list[ResponseLevel]
    action_state: str
    confidence: float
    poster_text: str


def _confidence_to_float(tag: ConvictionTag) -> float:
    if tag == ConvictionTag.HIGH:
        return 0.9
    if tag == ConvictionTag.MODERATE:
        return 0.6
    return 0.3


def _map_level(level: LevelDecision, current_price: float, atr14: float | None) -> ResponseLevel:
    distance_atr = None
    if atr14 and atr14 > 0:
        distance_atr = abs(level.price - current_price) / atr14
    return ResponseLevel(
        label="+".join(level.sources),
        price=level.price,
        score=level.score,
        distance_atr=distance_atr,
        rationale=level.trigger_note,
    )


def _map_response(result: AnalysisResult, current_price: float, atr14: float | None) -> AnalyzeResponse:
    return AnalyzeResponse(
        ticker=result.ticker,
        date_et=result.date_et,
        strongest_resistance=[_map_level(x, current_price, atr14) for x in result.strongest_resistance],
        weakest_resistance=[_map_level(x, current_price, atr14) for x in result.weakest_resistance],
        strongest_support=[_map_level(x, current_price, atr14) for x in result.strongest_support],
        weakest_support=[_map_level(x, current_price, atr14) for x in result.weakest_support],
        action_state=result.action_state.value,
        confidence=_confidence_to_float(result.confidence),
        poster_text=build_poster(result),
    )


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(payload: AnalyzeRequest) -> AnalyzeResponse:
    try:
        result = score(
            AnalysisPayload(
                date_et=payload.date_et,
                ticker=payload.ticker,
                timeframe=payload.timeframe.value,
                lookback_days=payload.lookback_days,
                current_price=payload.current_price,
                levels=LevelsPayload(**payload.levels.model_dump()),
            )
        )
    except Exception:
        logger.exception("Scoring failed for ticker=%s date=%s", payload.ticker, payload.date_et)
        return JSONResponse(
            status_code=500,
            content={"detail": "An internal error occurred during analysis. Please try again."},
        )
    return _map_response(result, payload.current_price, payload.levels.atr14)
