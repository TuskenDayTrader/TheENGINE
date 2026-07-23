from __future__ import annotations

import datetime as dt
import logging
import uuid
from enum import Enum

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from packages.core.image_extractor import ExtractionResult, extract_from_image
from packages.core.models import ActionState, AnalysisPayload, AnalysisResult, ConvictionTag, LevelDecision, LevelsPayload
from packages.core.output import build_poster
from packages.core.policy import ExtractionQualityResult, PolicyDecision, check_extraction_quality_gates, enforce_scalper_policy
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
    realized_pnl_usd: float | None = None
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
    policy: dict


class AnalyzeImageResponse(BaseModel):
    filename: str
    content_type: str
    size_bytes: int
    message: str
    extraction_confidence: float | None = None
    extraction_warning: str | None = None
    analysis: AnalyzeResponse | None = None
    debug_info: dict | None = None


def _confidence_to_float(tag: ConvictionTag) -> float:
    if tag == ConvictionTag.HIGH:
        return 0.9
    if tag == ConvictionTag.MODERATE:
        return 0.6
    return 0.3


def _levels_to_price_list(payload: LevelsPayload) -> list[float]:
    """Return all non-None, non-ATR price values from a LevelsPayload."""
    import dataclasses

    skip = {"atr14"}
    prices: list[float] = []
    for f in dataclasses.fields(payload):
        if f.name in skip:
            continue
        val = getattr(payload, f.name)
        if isinstance(val, (int, float)) and val is not None:
            prices.append(float(val))
    return prices


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


def _map_response(
    result: AnalysisResult,
    current_price: float,
    atr_14: float | None,
    policy: PolicyDecision | None = None,
) -> AnalyzeResponse:
    policy_payload = _policy_to_payload(
        policy=policy,
        fallback_state=result.action_state,
        fallback_confidence=result.confidence,
    )
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
        policy=policy_payload,
    )


def _mutate_result_with_policy(
    result: AnalysisResult, policy_reasons: list[str], enforced_state: ActionState
) -> None:
    """Mutate the scoring result in-place so poster output reflects policy enforcement."""
    result.action_state = enforced_state
    if policy_reasons and (not result.rationale or "Policy:" not in result.rationale):
        result.rationale = f"{result.rationale or ''} Policy: {' '.join(policy_reasons)}"


def _policy_to_payload(
    policy: PolicyDecision | None,
    fallback_state: ActionState,
    fallback_confidence: ConvictionTag,
) -> dict:
    if policy is None:
        return {
            "original_action_state": fallback_state.value,
            "enforced_action_state": fallback_state.value,
            "lockout_active": False,
            "daily_profit_cap_usd": None,
            "lockout_reset_timezone": None,
            "lockout_reset_time": None,
            "rr_target": None,
            "confidence_value": _confidence_to_float(fallback_confidence),
            "min_confidence_for_action": None,
            "nearby_structure_threshold": None,
            "stand_down_reasons": [],
            "template_350": None,
        }
    return {
        "original_action_state": policy.original_action_state.value,
        "enforced_action_state": policy.enforced_action_state.value,
        "lockout_active": policy.lockout_active,
        "daily_profit_cap_usd": policy.daily_profit_cap_usd,
        "lockout_reset_timezone": policy.lockout_reset_timezone,
        "lockout_reset_time": policy.lockout_reset_time,
        "rr_target": policy.rr_target,
        "confidence_value": policy.confidence_value,
        "min_confidence_for_action": policy.confidence_threshold,
        "nearby_structure_threshold": policy.nearby_structure_threshold,
        "stand_down_reasons": policy.stand_down_reasons,
        "template_350": {
            "symbol": policy.template.symbol,
            "value": policy.template.value,
            "unit": policy.template.unit,
            "tick_size": policy.template.tick_size,
            "ticks_per_point": policy.template.ticks_per_point,
            "dollars_per_tick": policy.template.dollars_per_tick,
            "template_ticks": policy.template.template_ticks,
            "template_price_distance": policy.template.template_price_distance,
            "estimated_risk_usd": policy.template.estimated_risk_usd,
            "estimated_reward_usd": policy.template.estimated_reward_usd,
        },
    }


@router.post("/analyze-image", response_model=AnalyzeImageResponse, response_model_exclude_none=True)
async def analyze_image(
    file: UploadFile = File(...),
    ticker: str | None = Form(default=None),
    timeframe: str | None = Form(default=None),
    lookback_days: int | None = Form(default=None),
    date_et: str | None = Form(default=None),
    realized_pnl_usd: float | None = Form(default=None),
    debug: bool = Query(default=False),
) -> AnalyzeImageResponse:
    content_type = file.content_type or ""
    if content_type not in ALLOWED_IMAGE_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file content type. Allowed types: image/png, image/jpeg, image/jpg, image/webp",
        )

    contents = await file.read()

    # Attempt extraction from image bytes
    extraction: ExtractionResult = extract_from_image(
        image_bytes=contents,
        ticker=ticker or "UNKNOWN",
        date_et=date_et,
        timeframe=timeframe or "30m",
        lookback_days=lookback_days or 5,
        debug=debug,
    )

    # When the image could not be decoded at all, return the minimal response
    # so that existing callers that only care about upload acknowledgement
    # continue to receive the same four-field payload.
    if not extraction.image_decoded:
        return AnalyzeImageResponse(
            filename=file.filename or "",
            content_type=content_type,
            size_bytes=len(contents),
            message="upload received",
        )

    # Run the scoring pipeline when extraction produced usable data
    analysis: AnalyzeResponse | None = None
    if extraction.levels_payload is not None and extraction.current_price is not None:
        try:
            # Quality gates: reject impossible/out-of-range extraction results
            _level_prices = _levels_to_price_list(extraction.levels_payload)
            qg: ExtractionQualityResult = check_extraction_quality_gates(
                ticker=ticker or "UNKNOWN",
                level_prices=_level_prices,
                current_price=extraction.current_price,
            )
            if not qg.passed:
                extraction.warning = (
                    (extraction.warning + " | " if extraction.warning else "")
                    + "Quality gate failed: "
                    + "; ".join(qg.rejection_reasons)
                )
            else:
                _date_et = date_et or dt.date.today().isoformat()
                try:
                    _timeframe_enum = Timeframe(timeframe or "30m")
                except ValueError:
                    _timeframe_enum = Timeframe.M30

                scored = score(
                    AnalysisPayload(
                        date_et=_date_et,
                        ticker=ticker or "UNKNOWN",
                        timeframe=_timeframe_enum.value,
                        lookback_days=lookback_days or 5,
                        current_price=extraction.current_price,
                        levels=extraction.levels_payload,
                    )
                )
                policy_decision = enforce_scalper_policy(
                    result=scored,
                    current_price=extraction.current_price,
                    contract_ticker=ticker or "UNKNOWN",
                    realized_pnl_usd=realized_pnl_usd,
                    atr14=extraction.levels_payload.atr14,
                )
                _mutate_result_with_policy(
                    result=scored,
                    policy_reasons=policy_decision.stand_down_reasons,
                    enforced_state=policy_decision.enforced_action_state,
                )
                analysis = _map_response(
                    scored,
                    extraction.current_price,
                    extraction.levels_payload.atr14,
                    policy_decision,
                )
        except Exception:
            error_id = uuid.uuid4().hex[:8]
            logger.exception(
                "Scoring from image extraction failed for ticker=%s error_id=%s",
                ticker,
                error_id,
            )
            extraction.warning = f"Analysis failed (error {error_id})"

    return AnalyzeImageResponse(
        filename=file.filename or "",
        content_type=content_type,
        size_bytes=len(contents),
        message="upload received",
        extraction_confidence=extraction.extraction_confidence,
        extraction_warning=extraction.warning,
        analysis=analysis,
        debug_info=extraction.debug_info if debug else None,
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
        policy_decision = enforce_scalper_policy(
            result=result,
            current_price=payload.current_price,
            contract_ticker=payload.ticker,
            realized_pnl_usd=payload.realized_pnl_usd,
            atr14=payload.levels.atr14,
        )
        _mutate_result_with_policy(
            result=result,
            policy_reasons=policy_decision.stand_down_reasons,
            enforced_state=policy_decision.enforced_action_state,
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
    return _map_response(result, payload.current_price, payload.levels.atr14, policy_decision)
