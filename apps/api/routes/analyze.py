from __future__ import annotations

import dataclasses
import datetime as dt
import logging
import uuid
from enum import Enum
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from packages.core.data_store import AnalysisRecord, ExtractionRecord, get_store
from packages.core.image_extractor import ExtractionResult, extract_from_image
from packages.core.insight_engine import find_confluence_zones, generate_ticker_report
from packages.core.models import ActionState, AnalysisPayload, AnalysisResult, ConvictionTag, LevelDecision, LevelsPayload
from packages.core.output import build_poster
from packages.core.policy import ExtractionQualityResult, PolicyDecision, check_extraction_quality_gates, enforce_scalper_policy
from packages.core.scoring import score

logger = logging.getLogger(__name__)
router = APIRouter()
ALLOWED_IMAGE_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp"}
DAILY_PNL_CAP: float = 550.0  # USD — stop trading once realized P&L reaches this amount


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
    daily_pnl: float | None = Field(default=None, description="Realized P&L for the trading day in USD. If >= $550, returns STOP_TRADING_DAY.")

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
    labeled_levels: dict | None = None
    detected_session: str | None = None
    rejected_outliers: list[dict] | None = None
    interior_confluence_zones: list[dict] | None = None


class HistoryEntryResponse(BaseModel):
    id: int
    created_at: str
    ticker: str
    date_et: str | None
    session: str | None
    filename: str | None
    confidence: float
    current_price: float | None
    quality_passed: bool
    action_state: str | None
    analysis_confidence: float | None
    labeled_levels: dict | None
    warning: str | None


class HistoryResponse(BaseModel):
    ticker: str
    entries: list[HistoryEntryResponse]
    total: int


class ConfluenceZoneResponse(BaseModel):
    price: float
    hit_count: int
    fields: list[str]
    last_seen: str | None
    confluence: bool


class InsightsResponse(BaseModel):
    ticker: str
    generated_at: str
    total_extractions: int
    quality_gate_pass_rate: float
    avg_confidence: float
    avg_labeled_levels: float
    common_sessions: list[str]
    recurring_levels: list[ConfluenceZoneResponse]
    gap_analysis: list[str]
    suggested_improvements: list[str]


def _confidence_to_float(tag: ConvictionTag) -> float:
    if tag == ConvictionTag.HIGH:
        return 0.9
    if tag == ConvictionTag.MODERATE:
        return 0.6
    return 0.3


def _persist_extraction(
    extraction: ExtractionResult,
    ticker: str,
    date_et: str | None,
    timeframe: str | None,
    filename: str | None,
    quality_passed: bool,
) -> int:
    """Record an extraction event in the persistent store and return its row id."""
    levels_dict: dict[str, Any] | None = None
    if extraction.levels_payload is not None:
        try:
            levels_dict = dataclasses.asdict(extraction.levels_payload)
        except Exception:
            pass
    try:
        rec = ExtractionRecord(
            ticker=ticker,
            date_et=date_et,
            timeframe=timeframe,
            filename=filename,
            session=extraction.detected_session,
            num_lines=extraction.num_lines_detected,
            num_axis_points=extraction.num_axis_points,
            confidence=extraction.extraction_confidence,
            current_price=extraction.current_price,
            atr=extraction.detected_atr,
            labeled_levels=extraction.labeled_levels,
            levels_json=levels_dict,
            warning=extraction.warning,
            quality_passed=quality_passed,
        )
        return get_store().record_extraction(rec)
    except Exception as exc:
        logger.warning("Failed to persist extraction for ticker=%s: %s", ticker, exc)
        return 0


def _persist_analysis(
    analysis_resp: AnalyzeResponse,
    ticker: str,
    date_et: str | None,
    session: str | None,
    extraction_id: int,
) -> None:
    """Record an analysis result in the persistent store."""
    try:
        rec = AnalysisRecord(
            ticker=ticker,
            date_et=date_et,
            session=session,
            action_state=analysis_resp.action_state,
            confidence=analysis_resp.confidence,
            strongest_res=[lvl.price for lvl in analysis_resp.strongest_resistance],
            strongest_sup=[lvl.price for lvl in analysis_resp.strongest_support],
            policy_json=analysis_resp.policy,
            poster_text=analysis_resp.poster_text,
            extraction_id=extraction_id,
        )
        get_store().record_analysis(rec, extraction_id=extraction_id)
    except Exception as exc:
        logger.warning("Failed to persist analysis for ticker=%s: %s", ticker, exc)


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
                axis_min=extraction.axis_price_min,
                axis_max=extraction.axis_price_max,
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

    # Persist extraction and analysis to the living data store
    _t = ticker or "UNKNOWN"
    _d = date_et or dt.date.today().isoformat()
    quality_passed = analysis is not None
    extraction_id = _persist_extraction(
        extraction=extraction,
        ticker=_t,
        date_et=_d,
        timeframe=timeframe,
        filename=file.filename,
        quality_passed=quality_passed,
    )
    if analysis is not None:
        _persist_analysis(
            analysis_resp=analysis,
            ticker=_t,
            date_et=_d,
            session=extraction.detected_session,
            extraction_id=extraction_id,
        )

    return AnalyzeImageResponse(
        filename=file.filename or "",
        content_type=content_type,
        size_bytes=len(contents),
        message="upload received",
        extraction_confidence=extraction.extraction_confidence,
        extraction_warning=extraction.warning,
        analysis=analysis,
        debug_info=extraction.debug_info if debug else None,
        labeled_levels=extraction.labeled_levels if extraction.labeled_levels else None,
        detected_session=extraction.detected_session,
        rejected_outliers=(
            [{"price": round(p, 4), "reason": r} for p, r in extraction.rejected_outliers]
            if extraction.rejected_outliers else None
        ),
        interior_confluence_zones=(
            extraction.interior_confluence_zones if extraction.interior_confluence_zones else None
        ),
    )


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(payload: AnalyzeRequest) -> AnalyzeResponse:
    # --- Daily profit cap: $550 lockout -----------------------------------
    if payload.daily_pnl is not None and payload.daily_pnl >= DAILY_PNL_CAP:
        return AnalyzeResponse(
            ticker=payload.ticker,
            date_et=payload.date_et,
            strongest_resistance=[],
            weakest_resistance=[],
            strongest_support=[],
            weakest_support=[],
            action_state=ActionState.STOP_TRADING_DAY.value,
            confidence=1.0,
            poster_text=(
                f"{payload.ticker} DAILY CAP REACHED\n"
                f"Date: {payload.date_et}\n"
                f"Realized P&L: ${payload.daily_pnl:.2f} >= ${DAILY_PNL_CAP:.0f} daily cap.\n"
                "Action State: STOP_TRADING_DAY\n"
                "Confidence: HIGH\n"
                "Rationale: Daily profit target reached. No further trading today. Bank it."
            ),
            policy={
                "original_action_state": ActionState.STOP_TRADING_DAY.value,
                "enforced_action_state": ActionState.STOP_TRADING_DAY.value,
                "lockout_active": True,
                "daily_profit_cap_usd": DAILY_PNL_CAP,
                "lockout_reset_timezone": None,
                "lockout_reset_time": None,
                "rr_target": None,
                "confidence_value": 1.0,
                "min_confidence_for_action": None,
                "nearby_structure_threshold": None,
                "stand_down_reasons": [],
                "template_350": None,
            },
        )
    # ----------------------------------------------------------------------
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


@router.get("/history", response_model=HistoryResponse)
async def get_history(
    ticker: str = Query(..., description="Ticker symbol (e.g. NQ, ES)"),
    date_et: str | None = Query(default=None, description="Filter by date (YYYY-MM-DD)"),
    limit: int = Query(default=50, ge=1, le=500, description="Max number of records"),
) -> HistoryResponse:
    """
    Retrieve the upload and analysis history for a ticker.

    Returns the most recent *limit* extraction events along with any
    associated analysis results, ordered newest-first.  This powers the
    "living memory" of the engine — every upload is stored and recallable.
    """
    try:
        rows = get_store().get_history(ticker=ticker.upper(), limit=limit, date_et=date_et)
    except Exception as exc:
        logger.exception("History query failed for ticker=%s: %s", ticker, exc)
        raise HTTPException(status_code=500, detail="Failed to retrieve history.")

    entries: list[HistoryEntryResponse] = []
    for r in rows:
        try:
            ll = r.get("labeled_levels")
            if isinstance(ll, str):
                import json as _json
                ll = _json.loads(ll) if ll else {}
        except Exception:
            ll = {}
        entries.append(
            HistoryEntryResponse(
                id=r["id"],
                created_at=r["created_at"],
                ticker=r["ticker"],
                date_et=r.get("date_et"),
                session=r.get("session"),
                filename=r.get("filename"),
                confidence=r.get("confidence", 0.0),
                current_price=r.get("current_price"),
                quality_passed=bool(r.get("quality_passed", 0)),
                action_state=r.get("action_state"),
                analysis_confidence=r.get("analysis_confidence"),
                labeled_levels=ll or None,
                warning=r.get("warning"),
            )
        )

    return HistoryResponse(ticker=ticker.upper(), entries=entries, total=len(entries))


@router.get("/insights", response_model=InsightsResponse)
async def get_insights(
    ticker: str = Query(..., description="Ticker symbol (e.g. NQ, ES)"),
    lookback: int = Query(default=100, ge=10, le=1000, description="Number of past extractions to analyse"),
) -> InsightsResponse:
    """
    Generate self-improvement insights for a ticker.

    Analyses stored extraction history to surface:
    - Quality gate pass rate and average confidence
    - Recurring price zones (structural memory / confluence)
    - Fields that are frequently missing (gap analysis)
    - Actionable suggestions to improve extraction accuracy

    This endpoint makes the engine "think about itself" — identifying its own
    weak spots and telling you how to feed it better data.
    """
    try:
        report = generate_ticker_report(ticker=ticker.upper(), lookback=lookback)
    except Exception as exc:
        logger.exception("Insights generation failed for ticker=%s: %s", ticker, exc)
        raise HTTPException(status_code=500, detail="Failed to generate insights.")

    zones = [
        ConfluenceZoneResponse(
            price=z["price"],
            hit_count=z["hit_count"],
            fields=z["fields"],
            last_seen=z.get("last_seen"),
            confluence=z.get("confluence", False),
        )
        for z in (report.recurring_levels or [])
    ]

    return InsightsResponse(
        ticker=report.ticker,
        generated_at=report.generated_at,
        total_extractions=report.total_extractions,
        quality_gate_pass_rate=report.quality_gate_pass_rate,
        avg_confidence=report.avg_confidence,
        avg_labeled_levels=report.avg_labeled_levels,
        common_sessions=report.common_sessions,
        recurring_levels=zones,
        gap_analysis=report.gap_analysis,
        suggested_improvements=report.suggested_improvements,
    )
