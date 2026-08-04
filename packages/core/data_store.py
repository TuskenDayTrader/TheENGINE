"""
Persistent data store for TheENGINE.

Every chart upload, extraction result, and analysis output is recorded here
so the system can learn from history, recall prior readings, and identify
recurring patterns and gaps in its own accuracy.

Storage backend: SQLite (single-file, zero-dependency, runs locally and in CI).
The database file path defaults to ``data/engine_store.db`` relative to the
project root; override via the ``ENGINE_STORE_PATH`` environment variable.

Public API
----------
- :func:`get_store`         – return the singleton ``EngineStore`` instance.
- :class:`EngineStore`      – main store class.
  - :meth:`record_extraction`  – persist an extraction event.
  - :meth:`record_analysis`    – persist an analysis result.
  - :meth:`get_history`        – retrieve recent extractions for a ticker.
  - :meth:`get_insights`       – produce self-improvement signals.
  - :meth:`get_level_memory`   – recall historically important price levels.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Database path
# ---------------------------------------------------------------------------

_DEFAULT_DB_DIR = Path(__file__).parent.parent.parent / "data"
_DEFAULT_DB_PATH = _DEFAULT_DB_DIR / "engine_store.db"


def _resolve_db_path() -> Path:
    env_path = os.environ.get("ENGINE_STORE_PATH")
    if env_path:
        return Path(env_path)
    return _DEFAULT_DB_PATH


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS extractions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT    NOT NULL,
    ticker          TEXT    NOT NULL,
    date_et         TEXT,
    timeframe       TEXT,
    filename        TEXT,
    session         TEXT,
    -- OCR metrics
    num_lines       INTEGER DEFAULT 0,
    num_axis_points INTEGER DEFAULT 0,
    confidence      REAL    DEFAULT 0.0,
    -- Extracted current price
    current_price   REAL,
    -- Detected ATR
    atr             REAL,
    -- Labeled levels as JSON {field: price}
    labeled_levels  TEXT    DEFAULT '{}',
    -- Full LevelsPayload as JSON (NULL when extraction failed quality gates)
    levels_json     TEXT,
    -- Warning message
    warning         TEXT,
    -- Quality gate passed?
    quality_passed  INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS analyses (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    extraction_id   INTEGER REFERENCES extractions(id),
    created_at      TEXT    NOT NULL,
    ticker          TEXT    NOT NULL,
    date_et         TEXT,
    session         TEXT,
    action_state    TEXT,
    confidence      REAL,
    -- Strongest resistance / support prices (JSON arrays)
    strongest_res   TEXT,
    strongest_sup   TEXT,
    -- Full policy decision JSON
    policy_json     TEXT,
    -- Poster text for recall
    poster_text     TEXT
);

CREATE TABLE IF NOT EXISTS level_observations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT    NOT NULL,
    ticker          TEXT    NOT NULL,
    field_name      TEXT    NOT NULL,
    price           REAL    NOT NULL,
    session         TEXT,
    date_et         TEXT,
    -- How many times this price±tolerance has appeared
    hit_count       INTEGER DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_extractions_ticker_date
    ON extractions (ticker, date_et);
CREATE INDEX IF NOT EXISTS idx_analyses_ticker_date
    ON analyses (ticker, date_et);
CREATE INDEX IF NOT EXISTS idx_level_obs_ticker_field
    ON level_observations (ticker, field_name);
"""

# Price tolerance for level_observations deduplication (NQ ~5 pts, ES ~2 pts)
_LEVEL_DEDUP_TOLERANCE: float = 10.0


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ExtractionRecord:
    ticker: str
    date_et: Optional[str]
    timeframe: Optional[str]
    filename: Optional[str]
    session: Optional[str]
    num_lines: int
    num_axis_points: int
    confidence: float
    current_price: Optional[float]
    atr: Optional[float]
    labeled_levels: Dict[str, float]
    levels_json: Optional[Dict[str, Any]]
    warning: Optional[str]
    quality_passed: bool
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class AnalysisRecord:
    ticker: str
    date_et: Optional[str]
    session: Optional[str]
    action_state: str
    confidence: float
    strongest_res: List[float]
    strongest_sup: List[float]
    policy_json: Optional[Dict[str, Any]]
    poster_text: Optional[str]
    extraction_id: Optional[int] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class InsightReport:
    ticker: str
    generated_at: str
    total_extractions: int
    quality_gate_pass_rate: float
    avg_confidence: float
    avg_labeled_levels: float
    common_sessions: List[str]
    recurring_levels: List[Dict[str, Any]]
    gap_analysis: List[str]
    suggested_improvements: List[str]


# ---------------------------------------------------------------------------
# Store implementation
# ---------------------------------------------------------------------------

class EngineStore:
    """Thread-safe SQLite-backed persistent store for TheENGINE."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._initialise()

    def _initialise(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA_SQL)
        logger.info("EngineStore initialised at %s", self._db_path)

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self._db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------------

    def record_extraction(self, record: ExtractionRecord) -> int:
        """Persist an extraction event and return the new row id."""
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute(
                    """
                    INSERT INTO extractions
                        (created_at, ticker, date_et, timeframe, filename,
                         session, num_lines, num_axis_points, confidence,
                         current_price, atr, labeled_levels, levels_json,
                         warning, quality_passed)
                    VALUES
                        (:created_at, :ticker, :date_et, :timeframe, :filename,
                         :session, :num_lines, :num_axis_points, :confidence,
                         :current_price, :atr, :labeled_levels, :levels_json,
                         :warning, :quality_passed)
                    """,
                    {
                        "created_at": record.created_at,
                        "ticker": record.ticker,
                        "date_et": record.date_et,
                        "timeframe": record.timeframe,
                        "filename": record.filename,
                        "session": record.session,
                        "num_lines": record.num_lines,
                        "num_axis_points": record.num_axis_points,
                        "confidence": record.confidence,
                        "current_price": record.current_price,
                        "atr": record.atr,
                        "labeled_levels": json.dumps(record.labeled_levels),
                        "levels_json": json.dumps(record.levels_json) if record.levels_json else None,
                        "warning": record.warning,
                        "quality_passed": int(record.quality_passed),
                    },
                )
                extraction_id = cur.lastrowid or 0

            # Record level observations for confluence memory
            self._update_level_observations(
                conn_fn=self._connect,
                ticker=record.ticker,
                labeled_levels=record.labeled_levels,
                session=record.session,
                date_et=record.date_et,
            )
            return extraction_id

    def _update_level_observations(
        self,
        conn_fn: Any,
        ticker: str,
        labeled_levels: Dict[str, float],
        session: Optional[str],
        date_et: Optional[str],
    ) -> None:
        """Upsert level observations for confluence tracking."""
        if not labeled_levels:
            return
        now = datetime.now(timezone.utc).isoformat()
        with conn_fn() as conn:
            for field_name, price in labeled_levels.items():
                if field_name in ("atr14",) or price is None:
                    continue
                # Check if a nearby price already exists for this field
                row = conn.execute(
                    """
                    SELECT id, hit_count FROM level_observations
                    WHERE ticker = ? AND field_name = ?
                      AND ABS(price - ?) <= ?
                    ORDER BY ABS(price - ?) ASC
                    LIMIT 1
                    """,
                    (ticker, field_name, price, _LEVEL_DEDUP_TOLERANCE, price),
                ).fetchone()
                if row:
                    conn.execute(
                        "UPDATE level_observations SET hit_count = hit_count + 1, "
                        "created_at = ? WHERE id = ?",
                        (now, row["id"]),
                    )
                else:
                    conn.execute(
                        """
                        INSERT INTO level_observations
                            (created_at, ticker, field_name, price, session, date_et)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (now, ticker, field_name, price, session, date_et),
                    )

    def record_analysis(
        self, record: AnalysisRecord, extraction_id: Optional[int] = None
    ) -> int:
        """Persist an analysis result and return the new row id."""
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute(
                    """
                    INSERT INTO analyses
                        (extraction_id, created_at, ticker, date_et, session,
                         action_state, confidence, strongest_res, strongest_sup,
                         policy_json, poster_text)
                    VALUES
                        (:extraction_id, :created_at, :ticker, :date_et, :session,
                         :action_state, :confidence, :strongest_res, :strongest_sup,
                         :policy_json, :poster_text)
                    """,
                    {
                        "extraction_id": extraction_id or record.extraction_id,
                        "created_at": record.created_at,
                        "ticker": record.ticker,
                        "date_et": record.date_et,
                        "session": record.session,
                        "action_state": record.action_state,
                        "confidence": record.confidence,
                        "strongest_res": json.dumps(record.strongest_res),
                        "strongest_sup": json.dumps(record.strongest_sup),
                        "policy_json": json.dumps(record.policy_json) if record.policy_json else None,
                        "poster_text": record.poster_text,
                    },
                )
                return cur.lastrowid or 0

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    def get_history(
        self,
        ticker: str,
        limit: int = 50,
        date_et: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return recent extraction + analysis history for *ticker*."""
        with self._connect() as conn:
            params: list = [ticker]
            date_clause = ""
            if date_et:
                date_clause = "AND e.date_et = ?"
                params.append(date_et)
            params.append(limit)
            rows = conn.execute(
                f"""
                SELECT
                    e.id, e.created_at, e.ticker, e.date_et, e.session,
                    e.timeframe, e.filename,
                    e.num_lines, e.num_axis_points, e.confidence,
                    e.current_price, e.atr, e.labeled_levels,
                    e.warning, e.quality_passed,
                    a.action_state, a.confidence AS analysis_confidence,
                    a.strongest_res, a.strongest_sup, a.poster_text
                FROM extractions e
                LEFT JOIN analyses a ON a.extraction_id = e.id
                WHERE e.ticker = ? {date_clause}
                ORDER BY e.id DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [dict(r) for r in rows]

    def get_level_memory(
        self,
        ticker: str,
        field_name: Optional[str] = None,
        min_hits: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        Return historically significant price levels for *ticker*.

        Levels that have appeared across multiple sessions (hit_count ≥
        *min_hits*) indicate strong structural zones.
        """
        with self._connect() as conn:
            params: list = [ticker, min_hits]
            field_clause = ""
            if field_name:
                field_clause = "AND field_name = ?"
                params.insert(1, field_name)
            rows = conn.execute(
                f"""
                SELECT field_name, price, hit_count, MAX(created_at) AS last_seen,
                       GROUP_CONCAT(DISTINCT date_et) AS dates
                FROM level_observations
                WHERE ticker = ? {field_clause} AND hit_count >= ?
                GROUP BY field_name, ROUND(price / {_LEVEL_DEDUP_TOLERANCE}) * {_LEVEL_DEDUP_TOLERANCE}
                ORDER BY hit_count DESC, field_name
                """,
                params,
            ).fetchall()
        return [dict(r) for r in rows]

    def get_insights(self, ticker: str, lookback_rows: int = 100) -> InsightReport:
        """
        Analyse stored history to produce self-improvement signals.

        Returns an :class:`InsightReport` containing:
        - quality gate pass rate
        - average confidence
        - average number of labeled levels per extraction
        - recurring high-confidence price zones
        - gap analysis (fields that are frequently missing)
        - suggested improvements
        """
        now = datetime.now(timezone.utc).isoformat()

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT quality_passed, confidence, labeled_levels,
                       session, num_lines, num_axis_points, warning
                FROM extractions
                WHERE ticker = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (ticker, lookback_rows),
            ).fetchall()

        if not rows:
            return InsightReport(
                ticker=ticker,
                generated_at=now,
                total_extractions=0,
                quality_gate_pass_rate=0.0,
                avg_confidence=0.0,
                avg_labeled_levels=0.0,
                common_sessions=[],
                recurring_levels=[],
                gap_analysis=["No historical data yet — upload more charts."],
                suggested_improvements=["Continue uploading charts to build history."],
            )

        total = len(rows)
        passed = sum(1 for r in rows if r["quality_passed"])
        avg_conf = sum(r["confidence"] for r in rows) / total

        # Count labeled levels per extraction
        label_counts: List[int] = []
        all_fields: Dict[str, int] = {}
        sessions: Dict[str, int] = {}
        for r in rows:
            try:
                ll = json.loads(r["labeled_levels"] or "{}")
            except (json.JSONDecodeError, TypeError):
                ll = {}
            label_counts.append(len(ll))
            for f in ll:
                all_fields[f] = all_fields.get(f, 0) + 1
            sess = r["session"] or "unknown"
            sessions[sess] = sessions.get(sess, 0) + 1

        avg_labels = sum(label_counts) / total if label_counts else 0.0

        # Top sessions
        common_sessions = sorted(sessions, key=lambda s: sessions[s], reverse=True)[:5]

        # Recurring levels (from level_observations)
        recurring = self.get_level_memory(ticker, min_hits=2)

        # Gap analysis: fields that appear in < 30 % of extractions
        expected_fields = [
            "pdh", "pdl", "asia_high", "asia_low",
            "london_high", "london_low", "ny_high", "ny_low",
        ]
        gap_analysis: List[str] = []
        for f in expected_fields:
            freq = all_fields.get(f, 0) / total if total > 0 else 0.0
            if freq < 0.30:
                gap_analysis.append(
                    f"Field '{f}' extracted in only {freq*100:.0f}% of uploads "
                    f"— label may be off-screen or OCR needs tuning"
                )

        # Suggested improvements
        suggestions: List[str] = []
        pass_rate = passed / total if total > 0 else 0.0
        if pass_rate < 0.70:
            suggestions.append(
                f"Quality gate pass rate is {pass_rate*100:.0f}% — "
                "consider adding more horizontal lines to the chart or "
                "zooming in to make labels more legible."
            )
        if avg_conf < 0.60:
            suggestions.append(
                f"Average extraction confidence is {avg_conf*100:.0f}% — "
                "ensure price-axis labels are clearly visible on the right margin."
            )
        if avg_labels < 2:
            suggestions.append(
                "Fewer than 2 labels per chart on average — "
                "verify that the TradingView indicator drawing labels "
                "(Asia High, NY Low, etc.) are visible on the right side of the chart."
            )
        if not suggestions:
            suggestions.append(
                "Extraction quality looks good. "
                "Keep uploading all four sessions per day for maximum confluence."
            )

        return InsightReport(
            ticker=ticker,
            generated_at=now,
            total_extractions=total,
            quality_gate_pass_rate=round(pass_rate, 4),
            avg_confidence=round(avg_conf, 4),
            avg_labeled_levels=round(avg_labels, 2),
            common_sessions=common_sessions,
            recurring_levels=recurring[:20],
            gap_analysis=gap_analysis,
            suggested_improvements=suggestions,
        )


# ---------------------------------------------------------------------------
# Singleton access
# ---------------------------------------------------------------------------

_store: Optional[EngineStore] = None
_store_lock = threading.Lock()


def get_store() -> EngineStore:
    """Return the process-singleton :class:`EngineStore` instance."""
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = EngineStore(_resolve_db_path())
    return _store
