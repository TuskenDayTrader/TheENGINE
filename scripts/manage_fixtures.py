from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

DEFAULT_FIXTURES_ROOT = Path("tests/fixtures/screenshots")
MAX_FILE_SIZE_BYTES = 2 * 1024 * 1024
TICKER_FOLDER_WARN_BYTES = 100 * 1024 * 1024
ARCHIVE_AGE_DAYS = 60


@dataclass(frozen=True)
class ParsedFixtureName:
    ticker: str
    symbol: str
    year: str
    month: str
    day: str
    hour: str
    minute: str
    second: str

    @property
    def timestamp(self) -> dt.datetime:
        return dt.datetime(
            int(self.year),
            int(self.month),
            int(self.day),
            int(self.hour),
            int(self.minute),
            int(self.second),
            tzinfo=dt.timezone.utc,
        )


def _split_stem(stem: str) -> tuple[str, str, str]:
    parts = stem.split("_")
    if len(parts) != 3:
        raise ValueError(
            "Filename must match SYMBOL_YYYY-MM-DD_HH-MM-SS.png (example: YM1!_2026-07-22_21-44-14.png)"
        )
    return parts[0], parts[1], parts[2]


def parse_fixture_filename(filename: str) -> ParsedFixtureName:
    stem = Path(filename).stem
    symbol, date_part, time_part = _split_stem(stem)

    symbol_upper = symbol.upper()
    ticker = ""
    for char in symbol_upper:
        if char.isalpha():
            ticker += char
        else:
            break

    if not ticker:
        raise ValueError(f"Unable to parse ticker from filename: {filename}")

    try:
        year, month, day = date_part.split("-")
        hour, minute, second = time_part.split("-")
        dt.datetime.strptime(f"{date_part} {time_part}", "%Y-%m-%d %H-%M-%S")
    except ValueError as exc:
        raise ValueError(f"Invalid date/time in filename: {filename}") from exc

    return ParsedFixtureName(
        ticker=ticker,
        symbol=symbol_upper,
        year=year,
        month=month,
        day=day,
        hour=hour,
        minute=minute,
        second=second,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_path(fixtures_root: Path) -> Path:
    return fixtures_root / "manifest.json"


def _load_manifest(fixtures_root: Path) -> dict:
    manifest_path = _manifest_path(fixtures_root)
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {"fixtures": [], "summary": {}}


def _save_manifest(fixtures_root: Path, manifest: dict) -> None:
    manifest_path = _manifest_path(fixtures_root)
    fixtures_root.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")


def _ticker_folder_size_bytes(fixtures_root: Path, ticker: str) -> int:
    ticker_path = fixtures_root / ticker
    total = 0
    if not ticker_path.exists():
        return 0
    for file_path in ticker_path.rglob("*"):
        if file_path.is_file():
            total += file_path.stat().st_size
    return total


def _archive_old_fixtures(fixtures_root: Path, manifest: dict, now: dt.datetime | None = None) -> None:
    now = now or dt.datetime.now(dt.timezone.utc)
    archive_cutoff = now - dt.timedelta(days=ARCHIVE_AGE_DAYS)
    archived_root = fixtures_root / "archived"

    retained: list[dict] = []
    changed = False

    for entry in manifest.get("fixtures", []):
        timestamp = dt.datetime.fromisoformat(entry["timestamp"].replace("Z", "+00:00"))
        entry_path = fixtures_root / entry["path"]
        if timestamp >= archive_cutoff:
            retained.append(entry)
            continue

        changed = True
        if entry_path.exists():
            archive_dest = archived_root / entry["path"]
            archive_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(entry_path), str(archive_dest))

    if changed:
        manifest["fixtures"] = retained


def _refresh_summary(manifest: dict) -> None:
    fixtures = manifest.get("fixtures", [])
    tickers = sorted({item["ticker"] for item in fixtures})

    if fixtures:
        timestamps = [dt.datetime.fromisoformat(item["timestamp"].replace("Z", "+00:00")) for item in fixtures]
        first = min(timestamps).date().isoformat()
        last = max(timestamps).date().isoformat()
        date_range = f"{first} to {last}"
        last_updated = max(timestamps).strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        date_range = None
        last_updated = None

    manifest["summary"] = {
        "total_fixtures": len(fixtures),
        "tickers": tickers,
        "date_range": date_range,
        "last_updated": last_updated,
    }


def add_fixture(
    source_file: Path,
    fixtures_root: Path,
    expected_action: str | None = None,
    notes: str | None = None,
) -> dict:
    if not source_file.exists():
        raise FileNotFoundError(source_file)

    file_size = source_file.stat().st_size
    if file_size > MAX_FILE_SIZE_BYTES:
        raise ValueError(f"File exceeds 2MB limit: {source_file.name}")

    parsed = parse_fixture_filename(source_file.name)
    manifest = _load_manifest(fixtures_root)

    fixture_hash = _sha256(source_file)

    for existing in manifest.get("fixtures", []):
        if existing.get("ticker") == parsed.ticker and existing.get("hash") == fixture_hash:
            return existing

    destination_name = f"{parsed.second}_{source_file.name}"
    rel_dir = Path(parsed.ticker) / parsed.year / parsed.month / parsed.day / parsed.hour / parsed.minute
    rel_path = rel_dir / destination_name
    destination_path = fixtures_root / rel_path
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_file, destination_path)

    kb = math.ceil(file_size / 1024)
    timestamp = parsed.timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")

    entry = {
        "ticker": parsed.ticker,
        "filename": destination_name,
        "path": rel_path.as_posix(),
        "timestamp": timestamp,
        "file_size_kb": kb,
        "hash": fixture_hash,
    }
    if expected_action:
        entry["expected_action"] = expected_action
    if notes:
        entry["notes"] = notes

    manifest.setdefault("fixtures", []).append(entry)
    manifest["fixtures"] = sorted(
        manifest["fixtures"],
        key=lambda item: (item["timestamp"], item["ticker"], item["filename"]),
    )

    _archive_old_fixtures(fixtures_root, manifest)
    _refresh_summary(manifest)
    _save_manifest(fixtures_root, manifest)

    ticker_size = _ticker_folder_size_bytes(fixtures_root, parsed.ticker)
    if ticker_size > TICKER_FOLDER_WARN_BYTES:
        print(
            f"WARNING: ticker folder '{parsed.ticker}' exceeds 100MB "
            f"({ticker_size / (1024 * 1024):.2f} MB)"
        )

    return entry


def _filtered_entries(fixtures_root: Path, ticker: str | None = None) -> list[dict]:
    manifest = _load_manifest(fixtures_root)
    entries = manifest.get("fixtures", [])
    if ticker:
        ticker_upper = ticker.upper()
        entries = [entry for entry in entries if entry.get("ticker") == ticker_upper]
    return entries


def cmd_add(args: argparse.Namespace) -> int:
    entry = add_fixture(
        source_file=Path(args.file).expanduser().resolve(),
        fixtures_root=Path(args.fixtures_root),
        expected_action=args.expected_action,
        notes=args.notes,
    )
    print(json.dumps(entry, indent=2))
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    entries = _filtered_entries(Path(args.fixtures_root), args.ticker)
    print(json.dumps(entries, indent=2))
    return 0


def cmd_timeline(args: argparse.Namespace) -> int:
    entries = _filtered_entries(Path(args.fixtures_root), args.ticker)

    if args.year:
        entries = [entry for entry in entries if entry["timestamp"].startswith(str(args.year))]
    if args.month:
        month = f"-{int(args.month):02d}-"
        entries = [entry for entry in entries if month in entry["timestamp"]]

    print(json.dumps(entries, indent=2))
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    date_str = dt.datetime.strptime(args.date, "%Y-%m-%d").date().isoformat()

    entries = _filtered_entries(Path(args.fixtures_root))
    entries = [entry for entry in entries if entry["timestamp"].startswith(date_str)]
    print(json.dumps(entries, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage screenshot fixtures")
    parser.add_argument(
        "--fixtures-root",
        default=str(DEFAULT_FIXTURES_ROOT),
        help="Root screenshot fixtures directory",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser("add", help="Add a fixture screenshot")
    add_parser.add_argument("--file", required=True, help="Path to screenshot file")
    add_parser.add_argument("--expected-action", default=None)
    add_parser.add_argument("--notes", default=None)
    add_parser.set_defaults(func=cmd_add)

    list_parser = subparsers.add_parser("list", help="List fixture entries")
    list_parser.add_argument("--ticker", default=None)
    list_parser.set_defaults(func=cmd_list)

    timeline_parser = subparsers.add_parser("timeline", help="Show ticker timeline")
    timeline_parser.add_argument("--ticker", required=True)
    timeline_parser.add_argument("--year", default=None)
    timeline_parser.add_argument("--month", default=None)
    timeline_parser.set_defaults(func=cmd_timeline)

    search_parser = subparsers.add_parser("search", help="Search fixtures by date")
    search_parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    search_parser.set_defaults(func=cmd_search)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
