from __future__ import annotations

import pathlib

from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

router = APIRouter()

_TEMPLATES_DIR = pathlib.Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


@router.get("/app", response_class=HTMLResponse, tags=["UI"])
async def app_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "app.html")
