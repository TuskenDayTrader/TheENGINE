from __future__ import annotations

import logging
import pathlib

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from apps.api.routes.analyze import router as analyze_router
from apps.api.routes.ui import router as ui_router

logger = logging.getLogger(__name__)

app = FastAPI(
    title="TheENGINE",
    version="0.3.0",
    description="2+2+2+2 confluence scoring engine API.",
)

_STATIC_DIR = pathlib.Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

app.include_router(analyze_router, tags=["Analysis"])
app.include_router(ui_router)


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
