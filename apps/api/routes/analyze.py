from __future__ import annotations

import datetime as dt
import logging
import uuid
from enum import Enum

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from packages.core.models import AnalysisPayload, AnalysisResult, ConvictionTag, LevelDecision, LevelsPayload
from packages.core.output import build_poster
from packages.core.scoring import score

logger = logging.getLogger(__name__)
router = APIRouter()
ALLOWED_IMAGE_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp"}


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


class AnalyzeImageResponse(BaseModel):
    filename: str
    content_type: str
    size_bytes: int
    message: str


def _confidence_to_float(tag: ConvictionTag) -> float:
    if tag == ConvictionTag.HIGH:
        return 0.9
    if tag == ConvictionTag.MODERATE:
        return 0.6
    return 0.3


def _map_level(level: LevelDecision, current_price: float, atr_14: float | None) -> ResponseLevel:
    """Map a core level using a precomputed 14-period ATR value for normalized distance."""
    distance_atr = None
    if atr_14 and atr_14 > 0:
        distance_atr = abs(level.price - current_price) / atr_14
    return ResponseLevel(
        label="+".join(level.sources),
        price=level.price,
        score=level.score,
        distance_atr=distance_atr,
        rationale=level.trigger_note,
    )


def _map_response(result: AnalysisResult, current_price: float, atr_14: float | None) -> AnalyzeResponse:
    return AnalyzeResponse(
        ticker=result.ticker,
        date_et=result.date_et,
        strongest_resistance=[_map_level(x, current_price, atr_14) for x in result.strongest_resistance],
        weakest_resistance=[_map_level(x, current_price, atr_14) for x in result.weakest_resistance],
        strongest_support=[_map_level(x, current_price, atr_14) for x in result.strongest_support],
        weakest_support=[_map_level(x, current_price, atr_14) for x in result.weakest_support],
        action_state=result.action_state.value,
        confidence=_confidence_to_float(result.confidence),
        poster_text=build_poster(result),
    )


@router.post("/analyze-image", response_model=AnalyzeImageResponse)
async def analyze_image(
    file: UploadFile = File(...),
    ticker: str | None = Form(default=None),
    timeframe: str | None = Form(default=None),
    lookback_days: int | None = Form(default=None),
    date_et: str | None = Form(default=None),
) -> AnalyzeImageResponse:
    # Reserved for future OCR/context handling; accepted now as part of the upload contract.
    _ = (ticker, timeframe, lookback_days, date_et)
    content_type = file.content_type or ""
    if content_type not in ALLOWED_IMAGE_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file content type. Allowed types: image/png, image/jpeg, image/jpg, image/webp",
        )

    contents = await file.read()
    return AnalyzeImageResponse(
        filename=file.filename or "",
        content_type=content_type,
        size_bytes=len(contents),
        message="upload received",
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
        error_id = uuid.uuid4().hex[:8]
        logger.exception(
            "Scoring failed for ticker=%s date=%s error_id=%s",
            payload.ticker,
            payload.date_et,
            error_id,
        )
        raise HTTPException(
            status_code=500,
            detail=f"An internal error occurred during analysis. Please try again. Ref: {error_id}",
        )
    return _map_response(result, payload.current_price, payload.levels.atr14)
