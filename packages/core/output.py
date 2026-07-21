"""
Poster text block formatter.

Generates the branded "SESSION CONFLUENCE MAP" ASCII output
used for dashboard graphics and social-media-ready cards.

Format
------
┌─────────────────────────────────────┐
│  SESSION CONFLUENCE MAP             │
│  NQU2026 · 30m · 2026-07-21        │
├─────────────────────────────────────┤
│  ▲ STRONGEST RESISTANCE             │
│    [price]  Label  (score)          │
│    [price]  Label  (score)          │
│  ▲ WEAKEST RESISTANCE               │
│    [price]  Label  (score)          │
│    [price]  Label  (score)          │
├─────────────────────────────────────┤
│  ⬤  CURRENT PRICE: [price]          │
├─────────────────────────────────────┤
│  ▼ STRONGEST SUPPORT                │
│    [price]  Label  (score)          │
│    [price]  Label  (score)          │
│  ▼ WEAKEST SUPPORT                  │
│    [price]  Label  (score)          │
│    [price]  Label  (score)          │
├─────────────────────────────────────┤
│  ACTION STATE : ACTIVE_LONG ✅      │
│  CONFIDENCE   : 0.72                │
├─────────────────────────────────────┤
│  TheENGINE · Not financial advice   │
└─────────────────────────────────────┘
"""

from __future__ import annotations

from typing import List

from .models import ActionState, AnalysisResult, LevelDecision

_WIDTH = 45
_STATE_ICON = {
    ActionState.ACTIVE_LONG: "✅",
    ActionState.ACTIVE_SHORT: "🔴",
    ActionState.STAND_DOWN: "⏸️",
}


def _border(char: str = "─") -> str:
    return "├" + char * (_WIDTH - 2) + "┤"


def _top() -> str:
    return "┌" + "─" * (_WIDTH - 2) + "┐"


def _bottom() -> str:
    return "└" + "─" * (_WIDTH - 2) + "┘"


def _row(content: str) -> str:
    padded = content.ljust(_WIDTH - 4)
    return f"│  {padded}  │"


def _level_rows(levels: List[LevelDecision], indent: str = "  ") -> List[str]:
    rows = []
    for lv in levels:
        text = f"{indent}{lv.price:>12,.2f}  {lv.label:<18}  ({lv.score:.3f})"
        rows.append(_row(text))
    return rows


def build_poster(result: AnalysisResult) -> str:
    """Return the full branded text block as a single string."""

    icon = _STATE_ICON.get(result.action_state, "")
    meta = f"{result.ticker} · {result.timeframe} · {result.date_et}"

    lines: List[str] = [
        _top(),
        _row("SESSION CONFLUENCE MAP"),
        _row(meta),
        _border(),
        _row("▲ STRONGEST RESISTANCE"),
        *_level_rows(result.strongest_resistance),
        _row("▲ WEAKEST RESISTANCE"),
        *_level_rows(result.weakest_resistance),
        _border(),
        _row(f"⬤  CURRENT PRICE: {result.current_price:,.2f}"),
        _border(),
        _row("▼ STRONGEST SUPPORT"),
        *_level_rows(result.strongest_support),
        _row("▼ WEAKEST SUPPORT"),
        *_level_rows(result.weakest_support),
        _border(),
        _row(f"ACTION STATE : {result.action_state.value} {icon}"),
        _row(f"CONFIDENCE   : {result.confidence:.2f}"),
        _border(),
        _row("TheENGINE · Not financial advice"),
        _bottom(),
    ]

    return "\n".join(lines)
