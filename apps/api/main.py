"""
TheENGINE – FastAPI application entry point.

Run locally:
    uvicorn apps.api.main:app --reload

Or from the repo root:
    python -m uvicorn apps.api.main:app --reload
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from apps.api.routes.analyze import router as analyze_router

logger = logging.getLogger(__name__)

app = FastAPI(
    title="TheENGINE",
    version="0.3.0",
    description=(
        "2+2+2+2 confluence scoring engine for NQ/ES futures. "
        "POST /analyze → strongest/weakest S/R levels + action state + poster text block."
    ),
    contact={"name": "TuskenDayTrader"},
    license_info={"name": "MIT"},
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(analyze_router, tags=["Analysis"])

# ---------------------------------------------------------------------------
# Global error handler – keeps 5xx responses safe (no internal stack traces)
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected internal error occurred. Please try again later."},
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/health", tags=["Ops"])
async def health() -> dict:
    return {"status": "ok", "service": "TheENGINE"}
