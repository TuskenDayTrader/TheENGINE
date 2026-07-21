from __future__ import annotations

from packages.core.models import AnalysisResult, LevelDecision


def _line(title: str, levels: list[LevelDecision]) -> str:
    items = ", ".join(f"{lvl.price:.2f} ({lvl.conviction.value})" for lvl in levels)
    return f"{title}: {items}"


def build_poster(result: AnalysisResult) -> str:
    return "\n".join(
        [
            f"{result.ticker} SESSION CONFLUENCE MAP",
            f"Date: {result.date_et}",
            _line("Strongest Resistance", result.strongest_resistance),
            _line("Weakest Resistance", result.weakest_resistance),
            _line("Strongest Support", result.strongest_support),
            _line("Weakest Support", result.weakest_support),
            f"Action State: {result.action_state.value}",
            f"Confidence: {result.confidence.value}",
            f"Rationale: {result.rationale}",
        ]
    )
