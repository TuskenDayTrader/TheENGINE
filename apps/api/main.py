from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from apps.api.routes.analyze import router as analyze_router

logger = logging.getLogger(__name__)

app = FastAPI(
    title="TheENGINE",
    version="0.3.0",
    description="2+2+2+2 confluence scoring engine API.",
)

app.include_router(analyze_router, tags=["Analysis"])


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected internal error occurred. Please try again later."},
    )


@app.get("/health", tags=["Ops"])
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "TheENGINE"}
