from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.manage_fixtures import add_fixture, parse_fixture_filename


@pytest.mark.parametrize(
    ("filename", "ticker", "minute", "second"),
    [
        ("YM1!_2026-07-22_21-44-14.png", "YM", "44", "14"),
        ("RTY_2026-07-22_21-43-53.png", "RTY", "43", "53"),
    ],
)
def test_parse_fixture_filename_extracts_ticker_and_time(
    filename: str, ticker: str, minute: str, second: str
):
    parsed = parse_fixture_filename(filename)

    assert parsed.ticker == ticker
    assert parsed.year == "2026"
    assert parsed.month == "07"
    assert parsed.day == "22"
    assert parsed.hour == "21"
    assert parsed.minute == minute
    assert parsed.second == second


def test_add_fixture_creates_nested_path_and_manifest(tmp_path: Path):
    source = tmp_path / "NQ1!_2026-07-22_21-43-17.png"
    source.write_bytes(b"fake-png-bytes")

    fixtures_root = tmp_path / "screenshots"
    entry = add_fixture(source, fixtures_root, expected_action="STAND_DOWN", notes="test")

    assert entry["path"] == "NQ/2026/07/22/21/43/17_NQ1!_2026-07-22_21-43-17.png"
    assert (fixtures_root / entry["path"]).exists()

    manifest_path = fixtures_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["summary"]["total_fixtures"] == 1
    assert manifest["summary"]["tickers"] == ["NQ"]


def test_add_fixture_dedupes_by_hash_within_ticker(tmp_path: Path):
    fixtures_root = tmp_path / "screenshots"

    first = tmp_path / "ES1!_2026-07-22_21-43-33.png"
    second = tmp_path / "ES1!_2026-07-22_21-44-33.png"
    first.write_bytes(b"same-content")
    second.write_bytes(b"same-content")

    add_fixture(first, fixtures_root)
    add_fixture(second, fixtures_root)

    manifest = json.loads((fixtures_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["summary"]["total_fixtures"] == 1
