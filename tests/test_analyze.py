from __future__ import annotations

import json
import pathlib

import pytest
from fastapi.testclient import TestClient

from apps.api.main import app

client = TestClient(app)
FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures"


def _load_fixture(name: str) -> dict:
    with open(FIXTURES_DIR / name) as f:
        data = json.load(f)
    data.pop("_comment", None)
    data.pop("_expected", None)
    return data


def _assert_contract(data: dict) -> None:
    for bucket in (
        "strongest_resistance",
        "weakest_resistance",
        "strongest_support",
        "weakest_support",
    ):
        assert bucket in data
        assert len(data[bucket]) == 2
        for entry in data[bucket]:
            for field in ("label", "price", "score", "distance_atr", "rationale"):
                assert field in entry


def test_analyze_nq_fixture_success():
    payload = _load_fixture("nq_sample.json")
    resp = client.post("/analyze", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    _assert_contract(data)
    assert data["action_state"] in ("ACTIVE_LONG", "ACTIVE_SHORT", "STAND_DOWN")
    assert 0.0 <= data["confidence"] <= 1.0
    assert payload["ticker"] in data["poster_text"]
    assert "SESSION CONFLUENCE MAP" in data["poster_text"]


def test_analyze_es_fixture_success():
    payload = _load_fixture("es_sample.json")
    resp = client.post("/analyze", json=payload)
    assert resp.status_code == 200, resp.text
    _assert_contract(resp.json())


def test_missing_required_field_returns_422():
    payload = _load_fixture("nq_sample.json")
    payload.pop("ticker")
    resp = client.post("/analyze", json=payload)
    assert resp.status_code == 422


def test_invalid_atr_zero_returns_422():
    payload = _load_fixture("nq_sample.json")
    payload["levels"]["atr14"] = 0
    resp = client.post("/analyze", json=payload)
    assert resp.status_code == 422


def test_high_less_than_low_returns_422():
    payload = _load_fixture("nq_sample.json")
    payload["levels"]["pdh"] = payload["levels"]["pdl"] - 1
    resp = client.post("/analyze", json=payload)
    assert resp.status_code == 422


def test_invalid_date_format_returns_422():
    payload = _load_fixture("nq_sample.json")
    payload["date_et"] = "21-07-2026"
    resp = client.post("/analyze", json=payload)
    assert resp.status_code == 422


def test_invalid_timeframe_returns_422():
    payload = _load_fixture("nq_sample.json")
    payload["timeframe"] = "2h"
    resp = client.post("/analyze", json=payload)
    assert resp.status_code == 422


def test_health_check():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_analyze_image_valid_upload_returns_200():
    # The fake PNG bytes fail to decode; the endpoint returns the minimal
    # four-field payload (extraction fields are excluded when image_decoded=False).
    resp = client.post(
        "/analyze-image",
        files={"file": ("chart.png", b"\x89PNG\r\n\x1a\nfakepng", "image/png")},
        data={
            "ticker": "NQ",
            "timeframe": "30m",
            "lookback_days": "5",
            "date_et": "2026-07-21",
        },
    )

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data == {
        "filename": "chart.png",
        "content_type": "image/png",
        "size_bytes": 15,
        "message": "upload received",
    }


def test_analyze_image_missing_file_returns_422():
    resp = client.post("/analyze-image", data={"ticker": "NQ"})
    assert resp.status_code == 422


def test_analyze_image_invalid_content_type_returns_400():
    resp = client.post(
        "/analyze-image",
        files={"file": ("notes.txt", b"not-an-image", "text/plain")},
    )
    assert resp.status_code == 400


def test_analyze_image_openapi_uses_multipart_form_data():
    schema = app.openapi()
    request_body = schema["paths"]["/analyze-image"]["post"]["requestBody"]
    assert "multipart/form-data" in request_body["content"]


def test_analyze_image_blank_image_decoded_returns_extraction_fields():
    """A valid but blank image should decode, include extraction_confidence, and set a warning."""
    import cv2
    import numpy as np

    blank = np.zeros((200, 400, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", blank)
    assert ok

    resp = client.post(
        "/analyze-image",
        files={"file": ("blank.png", bytes(buf), "image/png")},
        data={"ticker": "NQ"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["message"] == "upload received"
    # Blank image decodes but has nothing useful; extraction fields must be present
    assert "extraction_confidence" in data
    assert data["extraction_confidence"] < 0.5
    assert "extraction_warning" in data
    assert data.get("analysis") is None


def test_analyze_image_image_with_green_line_sets_line_count():
    """An image with a detectable green line should set num lines via extraction."""
    import cv2
    import numpy as np

    img = np.zeros((400, 800, 3), dtype=np.uint8)
    img[200, 80:700] = [129, 153, 8]  # TradingView green BGR
    ok, buf = cv2.imencode(".png", img)
    assert ok

    resp = client.post(
        "/analyze-image",
        files={"file": ("chart.png", bytes(buf), "image/png")},
        data={"ticker": "NQ"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["message"] == "upload received"
    assert "extraction_confidence" in data
    # With a line detected, confidence > 0
    assert data["extraction_confidence"] > 0.0


# ---------------------------------------------------------------------------
# End-to-end: /app submission path producing an analysis payload
# ---------------------------------------------------------------------------


def test_analyze_image_e2e_produces_analysis_when_extraction_succeeds(monkeypatch):
    """
    E2E: when the extractor returns a usable LevelsPayload, the /analyze-image
    endpoint runs the full scoring pipeline and returns an analysis payload
    matching the AnalyzeResponse schema.
    """
    import packages.core.image_extractor as img_ext
    import apps.api.routes.analyze as analyze_route
    from packages.core.models import LevelsPayload

    mock_result = img_ext.ExtractionResult(
        image_decoded=True,
        levels_payload=LevelsPayload(
            pdh=21000.0,
            pdl=19000.0,
            prior_settle=20000.0,
            atr14=150.0,
        ),
        current_price=20000.0,
        num_lines_detected=2,
        num_axis_points=5,
        extraction_confidence=0.75,
        warning=None,
    )

    # Patch the name as imported into the route module
    monkeypatch.setattr(analyze_route, "extract_from_image", lambda *a, **kw: mock_result)

    resp = client.post(
        "/analyze-image",
        files={"file": ("chart.png", b"\x89PNG\r\n\x1a\nfakepng", "image/png")},
        data={
            "ticker": "NQU2026",
            "timeframe": "30m",
            "lookback_days": "5",
            "date_et": "2026-07-21",
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()

    # Basic response fields
    assert data["message"] == "upload received"
    assert data["extraction_confidence"] == pytest.approx(0.75)
    assert data.get("extraction_warning") is None

    # Full analysis payload must be present
    analysis = data.get("analysis")
    assert analysis is not None, "Expected 'analysis' in response"

    # Validate AnalyzeResponse contract
    for bucket in ("strongest_resistance", "weakest_resistance", "strongest_support", "weakest_support"):
        assert bucket in analysis
        assert len(analysis[bucket]) == 2
        for entry in analysis[bucket]:
            assert "label" in entry
            assert "price" in entry
            assert "score" in entry

    assert analysis["action_state"] in ("ACTIVE_LONG", "ACTIVE_SHORT", "STAND_DOWN")
    assert 0.0 <= analysis["confidence"] <= 1.0
    assert "SESSION CONFLUENCE MAP" in analysis["poster_text"]
    assert "NQU2026" in analysis["poster_text"]


def test_analyze_image_e2e_low_confidence_shows_warning_but_may_include_analysis(monkeypatch):
    """
    When extraction confidence is below the threshold, the response includes
    extraction_warning; analysis is present only if a payload was produced.
    """
    import packages.core.image_extractor as img_ext
    import apps.api.routes.analyze as analyze_route
    from packages.core.models import LevelsPayload

    mock_result = img_ext.ExtractionResult(
        image_decoded=True,
        levels_payload=LevelsPayload(
            pdh=21000.0,
            pdl=19000.0,
            prior_settle=20000.0,
            atr14=150.0,
        ),
        current_price=20000.0,
        num_lines_detected=1,
        num_axis_points=1,
        extraction_confidence=0.25,
        warning="Low confidence extraction: only 1 price label(s) found on the axis (need ≥ 3 for reliable scale).",
    )

    monkeypatch.setattr(analyze_route, "extract_from_image", lambda *a, **kw: mock_result)

    resp = client.post(
        "/analyze-image",
        files={"file": ("chart.png", b"\x89PNG\r\n\x1a\nfakepng", "image/png")},
        data={"ticker": "NQ"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "extraction_warning" in data
    assert "Low confidence" in data["extraction_warning"]
    assert data["extraction_confidence"] == pytest.approx(0.25)


def test_analyze_image_e2e_no_levels_no_analysis(monkeypatch):
    """
    When extraction produces no levels, the analysis field is absent from
    the response (None values excluded via response_model_exclude_none=True).
    """
    import packages.core.image_extractor as img_ext
    import apps.api.routes.analyze as analyze_route

    mock_result = img_ext.ExtractionResult(
        image_decoded=True,
        levels_payload=None,
        current_price=None,
        num_lines_detected=0,
        num_axis_points=0,
        extraction_confidence=0.0,
        warning="Low confidence extraction: no coloured horizontal lines detected in the chart area.",
    )

    monkeypatch.setattr(analyze_route, "extract_from_image", lambda *a, **kw: mock_result)

    resp = client.post(
        "/analyze-image",
        files={"file": ("chart.png", b"\x89PNG\r\n\x1a\nfakepng", "image/png")},
        data={"ticker": "NQ"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "analysis" not in data  # excluded because it's None
    assert "extraction_warning" in data


def test_analyze_image_debug_query_param_returns_debug_info(monkeypatch):
    """
    When debug=true is passed as a query param, the response must include
    debug_info populated by the extractor.
    """
    import packages.core.image_extractor as img_ext
    import apps.api.routes.analyze as analyze_route

    mock_result = img_ext.ExtractionResult(
        image_decoded=True,
        levels_payload=None,
        current_price=None,
        num_lines_detected=2,
        num_axis_points=0,
        extraction_confidence=0.25,
        warning="Low confidence extraction: extraction confidence is low.",
        debug_info={
            "image_size": {"width": 800, "height": 600},
            "chart_roi": {"top": 48, "bottom": 540, "left": 0, "right": 680},
            "green_mask_pixels": 1200,
            "red_mask_pixels": 800,
            "contour_segments_raw": 4,
            "hough_segments_raw": 3,
            "segments_before_dedup": 4,
            "segments_after_dedup": 2,
        },
    )

    monkeypatch.setattr(analyze_route, "extract_from_image", lambda *a, **kw: mock_result)

    resp = client.post(
        "/analyze-image?debug=true",
        files={"file": ("chart.png", b"\x89PNG\r\n\x1a\nfakepng", "image/png")},
        data={"ticker": "NQ"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "debug_info" in data
    di = data["debug_info"]
    assert di["image_size"] == {"width": 800, "height": 600}
    assert di["green_mask_pixels"] == 1200
    assert di["segments_after_dedup"] == 2


def test_analyze_image_no_debug_param_excludes_debug_info(monkeypatch):
    """
    Without debug=true, debug_info must be absent from the response even if
    the extractor returned it.
    """
    import packages.core.image_extractor as img_ext
    import apps.api.routes.analyze as analyze_route

    mock_result = img_ext.ExtractionResult(
        image_decoded=True,
        num_lines_detected=0,
        extraction_confidence=0.0,
        warning="Low confidence extraction: extraction confidence is low.",
        debug_info={"image_size": {"width": 800, "height": 600}},
    )

    monkeypatch.setattr(analyze_route, "extract_from_image", lambda *a, **kw: mock_result)

    resp = client.post(
        "/analyze-image",
        files={"file": ("chart.png", b"\x89PNG\r\n\x1a\nfakepng", "image/png")},
        data={"ticker": "NQ"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "debug_info" not in data
