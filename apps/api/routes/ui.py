from __future__ import annotations

import pathlib

from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from packages.core.ui_theme import get_theme

router = APIRouter()

_TEMPLATES_DIR = pathlib.Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


@router.get("/app", response_class=HTMLResponse, tags=["UI"])
async def app_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "app.html", {})


@router.get("/api/theme", tags=["UI"])
async def theme_settings() -> dict[str, str]:
    return get_theme()
