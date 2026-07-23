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
# Matches indented YAML color properties with optional quotes, supporting
# 3-, 6-, and 8-digit hex color values.
_THEME_COLOR_LINE_PATTERN = re.compile(r'^\s+([a-z_]+):\s*["\']?(#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8}))["\']?\s*$')


def _theme_file_for(theme_name: str) -> pathlib.Path:
    if theme_name == DEFAULT_THEME_NAME:
        return DEFAULT_THEME_FILE
    return DEFAULT_THEME_FILE.with_name(f"ui_theme.{theme_name}.yaml")


def _load_theme_from_file(theme_file: pathlib.Path) -> dict[str, str]:
    theme = DEFAULT_THEME.copy()

    if not theme_file.exists():
        return theme

    in_theme_block = False
    with theme_file.open(encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped == "theme:":
                in_theme_block = True
                continue
            if in_theme_block and _is_unindented_key(raw_line, stripped):
                break

            if not in_theme_block:
                continue

            match = _THEME_COLOR_LINE_PATTERN.match(raw_line)
            if match:
                key, value = match.groups()
                theme[key] = value

    return theme


def _is_unindented_key(raw_line: str, stripped_line: str) -> bool:
    return bool(stripped_line.endswith(":") and raw_line and raw_line[0] not in (" ", "\t"))


def get_theme() -> dict[str, str]:
    raw_theme_name = os.getenv("UI_THEME")
    if raw_theme_name is None:
        requested_theme = DEFAULT_THEME_NAME
    else:
        requested_theme = raw_theme_name.strip()
        if not requested_theme:
            logger.warning("Empty UI_THEME value detected; falling back to %s", DEFAULT_THEME_NAME)
            requested_theme = DEFAULT_THEME_NAME

    theme_file = _theme_file_for(requested_theme)
    if requested_theme != DEFAULT_THEME_NAME and not theme_file.exists():
        logger.warning("Unsupported UI_THEME '%s'; falling back to %s", requested_theme, DEFAULT_THEME_NAME)
        theme_file = _theme_file_for(DEFAULT_THEME_NAME)
    return _load_theme_from_file(theme_file)
