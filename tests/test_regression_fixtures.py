from __future__ import annotations

import json
import pathlib

import pytest
from fastapi.testclient import TestClient

from apps.api.main import app

client = TestClient(app)
SCREENSHOTS_ROOT = pathlib.Path(__file__).parent / "fixtures" / "screenshots"
MANIFEST_PATH = SCREENSHOTS_ROOT / "manifest.json"


def _manifest_entries() -> list[dict]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return manifest["fixtures"]


@pytest.mark.parametrize("ticker", ["YM", "NQ", "ES", "RTY"])
def test_yieldpoints(ticker: str):
    entries = [entry for entry in _manifest_entries() if entry["ticker"] == ticker]
    assert entries, f"Expected at least one fixture for ticker {ticker}"

    for entry in entries:
        fixture_path = SCREENSHOTS_ROOT / entry["path"]
        assert fixture_path.exists(), f"Missing fixture file: {fixture_path}"


@pytest.mark.parametrize("entry", _manifest_entries(), ids=lambda e: e["filename"])
def test_regression_fixture_uploads_decode(entry: dict):
    fixture_path = SCREENSHOTS_ROOT / entry["path"]
    with fixture_path.open("rb") as f:
        payload = f.read()

    resp = client.post(
        "/analyze-image",
        files={"file": (fixture_path.name, payload, "image/png")},
        data={"ticker": entry["ticker"], "date_et": entry["timestamp"][:10], "timeframe": "30m"},
    )

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["filename"] == fixture_path.name
    assert "extraction_confidence" in data
