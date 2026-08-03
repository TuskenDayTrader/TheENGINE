"""
Insight engine for TheENGINE.

Analyses stored historical data to produce self-improvement signals,
identify structural confluence zones, and surface recurring price memory
that strengthens forecasting accuracy over time.

This module builds on top of :mod:`packages.core.data_store` and is
designed to grow smarter with every chart upload.

Public API
----------
- :func:`generate_ticker_report`  – full insight report for one ticker.
- :func:`find_confluence_zones`   – price clusters recurring across sessions.
- :func:`identify_gaps`           – fields consistently missing from extractions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from .data_store import EngineStore, InsightReport, get_store

logger = logging.getLogger(__name__)

# Tolerance (price points) within which separate observations are treated
# as the same structural zone.
_CONFLUENCE_TOLERANCE: float = 15.0

# Minimum number of session-observations for a price zone to be considered
# a "structural memory" level.
_MIN_ZONE_HITS: int = 2


def find_confluence_zones(
    ticker: str,
    store: Optional[EngineStore] = None,
    min_hits: int = _MIN_ZONE_HITS,
) -> List[Dict[str, Any]]:
    """
    Identify price zones that have appeared as named levels across multiple
    chart uploads for *ticker*.

    Each zone record contains:

    ``price``
        Representative price of the zone (median of observations).
    ``hit_count``
        Total number of times any level in this zone has been observed.
    ``fields``
        List of ``LevelsPayload`` field names that have been seen at this zone
        (e.g. ``["ny_low", "london_low"]`` when they co-locate → confluence).
    ``last_seen``
        ISO timestamp of the most recent observation.
    ``confluence``
        ``True`` when two or more *different* fields appear at this zone —
        meaning multiple session levels share this price (highest conviction).

    Parameters
    ----------
    ticker:
        Ticker symbol (e.g. "NQ", "ES").
    store:
        Optional store instance; defaults to the process singleton.
    min_hits:
        Minimum observation count to include a zone.
    """
    s = store or get_store()
    raw = s.get_level_memory(ticker, min_hits=min_hits)

    # Group by price zone
    zones: List[Dict[str, Any]] = []
    for rec in raw:
        price = rec["price"]
        field = rec["field_name"]
        # Try to merge into an existing zone
        merged = False
        for zone in zones:
            if abs(zone["price"] - price) <= _CONFLUENCE_TOLERANCE:
                zone["hit_count"] += rec["hit_count"]
                if field not in zone["fields"]:
                    zone["fields"].append(field)
                merged = True
                break
        if not merged:
            zones.append(
                {
                    "price": round(price, 4),
                    "hit_count": rec["hit_count"],
                    "fields": [field],
                    "last_seen": rec["last_seen"],
                    "confluence": False,
                }
            )

    # Mark confluence zones and sort by hit count
    for zone in zones:
        zone["confluence"] = len(zone["fields"]) >= 2
    zones.sort(key=lambda z: (-z["hit_count"], z["price"]))
    return zones


def identify_gaps(
    ticker: str,
    store: Optional[EngineStore] = None,
    lookback: int = 50,
) -> List[str]:
    """
    Return a list of human-readable gap descriptions: ``LevelsPayload``
    fields that are consistently missing from recent extractions for
    *ticker*, indicating either chart layout issues or OCR limitations.
    """
    report = generate_ticker_report(ticker, store=store, lookback=lookback)
    return report.gap_analysis


def generate_ticker_report(
    ticker: str,
    store: Optional[EngineStore] = None,
    lookback: int = 100,
) -> InsightReport:
    """
    Generate a full :class:`~packages.core.data_store.InsightReport` for
    *ticker*, enriched with confluence zone analysis.

    Parameters
    ----------
    ticker:
        Ticker symbol.
    store:
        Optional store instance; defaults to the process singleton.
    lookback:
        Number of most-recent extraction rows to analyse.
    """
    s = store or get_store()
    report = s.get_insights(ticker, lookback_rows=lookback)

    # Enrich recurring_levels with confluence flag from find_confluence_zones
    zones = find_confluence_zones(ticker, store=s)
    if zones:
        report.recurring_levels = zones  # type: ignore[assignment]

    return report
