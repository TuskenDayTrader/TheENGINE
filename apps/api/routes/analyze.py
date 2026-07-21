"""
POST /analyze route.

Keeps the endpoint thin; all business logic lives in packages/core/.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from packages.core.models import AnalysisPayload, AnalysisResult
from packages.core.scoring import score
from packages.core.output import build_poster

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/analyze",
    response_model=AnalysisResult,
    summary="Run 2+2+2+2 confluence analysis",
    description=(
        "Accepts a canonical AnalysisPayload and returns the scored "
        "2+2+2+2 level map, action state, and a branded poster text block."
    ),
    responses={
        422: {"description": "Validation error – invalid or missing fields"},
        500: {"description": "Internal server error"},
    },
)
async def analyze(payload: AnalysisPayload) -> AnalysisResult:
    """
    Run the scoring engine and format the response.

    - **422** is returned automatically by FastAPI when Pydantic validation fails.
    - Any unexpected error during scoring returns a plain **500**.
    """
    try:
        scored = score(payload)
    except Exception:
        logger.exception(
            "Scoring failed for ticker=%s date=%s", payload.ticker, payload.date_et
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "An internal error occurred during analysis. Please try again."},
        )

    result = AnalysisResult(
        ticker=payload.ticker,
        date_et=payload.date_et,
        timeframe=payload.timeframe.value,
        current_price=payload.current_price,
        **scored,
        poster_text="",  # placeholder until build_poster runs
    )

    # Attach poster text block (runs after full result is assembled)
    result = result.model_copy(update={"poster_text": build_poster(result)})

    return result

