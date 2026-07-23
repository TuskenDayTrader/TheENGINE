from __future__ import annotations

from fastapi.testclient import TestClient

from apps.api.main import app

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


def test_theme_api_accepts_hombre_env_override(monkeypatch):
    monkeypatch.setenv("UI_THEME", "hombre_red")

    resp = client.get("/api/theme")

    assert resp.status_code == 200
    assert resp.json()["primary_mid"] == "#8b0000"
