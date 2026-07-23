from __future__ import annotations

import logging
import os
import pathlib
import re

logger = logging.getLogger(__name__)

DEFAULT_THEME_NAME = "hombre_red"
DEFAULT_THEME = {
    "primary_dark": "#0a0a0a",
    "primary_mid": "#8b0000",
    "primary_bright": "#ff0000",
    "accent": "#ff6b6b",
}
DEFAULT_THEME_FILE = pathlib.Path(__file__).resolve().parents[2] / "config" / "ui_theme.yaml"
_THEME_LINE_PATTERN = re.compile(r'^\s+([a-z_]+):\s*["\']?(#[0-9a-fA-F]{6})["\']?\s*$')


def _theme_file_for(theme_name: str) -> pathlib.Path:
    if theme_name == DEFAULT_THEME_NAME:
        return DEFAULT_THEME_FILE
    return DEFAULT_THEME_FILE.with_name(f"ui_theme.{theme_name}.yaml")


def _load_theme_from_file(theme_file: pathlib.Path) -> dict[str, str]:
    theme = DEFAULT_THEME.copy()

    if not theme_file.exists():
        return theme

    in_theme_block = False
    for raw_line in theme_file.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped == "theme:":
            in_theme_block = True
            continue
        if in_theme_block and not raw_line.startswith("  "):
            break

        if not in_theme_block:
            continue

        match = _THEME_LINE_PATTERN.match(raw_line)
        if match:
            key, value = match.groups()
            theme[key] = value

    return theme


def get_theme() -> dict[str, str]:
    requested_theme = os.getenv("UI_THEME", DEFAULT_THEME_NAME).strip() or DEFAULT_THEME_NAME
    theme_file = _theme_file_for(requested_theme)
    if requested_theme != DEFAULT_THEME_NAME and not theme_file.exists():
        logger.warning("Unsupported UI_THEME '%s'; falling back to %s", requested_theme, DEFAULT_THEME_NAME)
        theme_file = _theme_file_for(DEFAULT_THEME_NAME)
    return _load_theme_from_file(theme_file)
