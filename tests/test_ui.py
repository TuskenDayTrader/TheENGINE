from __future__ import annotations

from fastapi.testclient import TestClient

from apps.api.main import app
from packages.core import ui_theme

client = TestClient(app)


def test_app_page_returns_200():
    resp = client.get("/app")
    assert resp.status_code == 200


def test_app_page_content_type_is_html():
    resp = client.get("/app")
    assert "text/html" in resp.headers["content-type"]


def test_app_page_contains_form_elements():
    resp = client.get("/app")
    html = resp.text

    # Screenshot upload input
    assert 'type="file"' in html
    assert 'accept="image/png,image/jpeg,image/jpg,image/webp"' in html

    # Ticker text input
    assert 'id="ticker"' in html

    # Timeframe select with expected options
    assert '<select' in html
    assert 'value="30m"' in html
    assert 'value="1h"' in html

    # Lookback days input
    assert 'id="lookback_days"' in html

    # Optional date input
    assert 'id="date_et"' in html
    assert 'type="date"' in html

    # Analyze button
    assert 'type="submit"' in html
    assert "Analyze" in html


def test_theme_api_returns_hombre_palette():
    resp = client.get("/api/theme")

    assert resp.status_code == 200
    assert resp.json() == {
        "primary_dark": "#0a0a0a",
        "primary_mid": "#8b0000",
        "primary_bright": "#ff0000",
        "accent": "#ff6b6b",
    }


def test_get_theme_accepts_alternate_theme_via_env(monkeypatch, tmp_path):
    default_file = tmp_path / "ui_theme.yaml"
    default_file.write_text(
        'theme:\n'
        '  primary_dark: "#0a0a0a"\n'
        '  primary_mid: "#8b0000"\n'
        '  primary_bright: "#ff0000"\n'
        '  accent: "#ff6b6b"\n',
        encoding="utf-8",
    )
    alternate_file = tmp_path / "ui_theme.nightfall.yaml"
    alternate_file.write_text(
        'theme:\n'
        '  primary_dark: "#111111"\n'
        '  primary_mid: "#550000"\n'
        '  primary_bright: "#ff3333"\n'
        '  accent: "#ffaa55"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(ui_theme, "DEFAULT_THEME_FILE", default_file)

    assert ui_theme.get_theme()["primary_dark"] == "#0a0a0a"
    monkeypatch.setenv("UI_THEME", "nightfall")
    assert ui_theme.get_theme() == {
        "primary_dark": "#111111",
        "primary_mid": "#550000",
        "primary_bright": "#ff3333",
        "accent": "#ffaa55",
    }
