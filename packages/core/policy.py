from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Optional

import yaml

from .models import ActionState, AnalysisResult, ConvictionTag


@dataclass(frozen=True)
class SymbolTickModel:
    tick_size: float
    ticks_per_point: float
    dollars_per_tick: float


@dataclass(frozen=True)
class PolicyConfig:
    daily_profit_cap_usd: float
    lockout_reset_timezone: str
    lockout_reset_time: str
    default_rr: float
    min_confidence_for_action: float
    allow_only_nearby_structure: bool
    proximity_model_type: str
    proximity_atr_multiple: float
    proximity_max_ticks_by_symbol: dict[str, int]
    template_value: int
    template_unit: str
    symbol_tick_model: dict[str, SymbolTickModel]
    minimum_distinct_levels: int = 1
    instrument_price_floor: dict[str, float] = field(default_factory=dict)


@dataclass
class ExtractionQualityResult:
    """Result of extraction quality gate checks.

    ``passed`` is ``True`` when all gates pass and analysis can proceed.
    ``rejection_reasons`` is a list of human-readable explanations when any
    gate fails; it is empty when ``passed`` is ``True``.
    """

    passed: bool
    rejection_reasons: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class PolicyTemplateView:
    symbol: str
    value: int
    unit: str
    tick_size: float | None
    ticks_per_point: float | None
    dollars_per_tick: float | None
    template_ticks: int | None
    template_price_distance: float | None
    estimated_risk_usd: float | None
    estimated_reward_usd: float | None


@dataclass(frozen=True)
class PolicyDecision:
    original_action_state: ActionState
    enforced_action_state: ActionState
    lockout_active: bool
    confidence_value: float
    confidence_threshold: float
    nearby_structure_threshold: float | None
    stand_down_reasons: list[str]
    rr_target: float
    daily_profit_cap_usd: float
    lockout_reset_timezone: str
    lockout_reset_time: str
    template: PolicyTemplateView


def _to_confidence_value(tag: ConvictionTag) -> float:
    if tag == ConvictionTag.HIGH:
        return 0.9
    if tag == ConvictionTag.MODERATE:
        return 0.6
    return 0.3


def _policy_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "trading_policy.yaml"


def _resolve_symbol_key(ticker: str, symbol_map: dict[str, Any]) -> str:
    ticker_upper = (ticker or "").upper()
    if not ticker_upper:
        return "UNKNOWN"
    matches = sorted((k for k in symbol_map.keys() if ticker_upper.startswith(k)), key=len, reverse=True)
    if matches:
        return matches[0]
    return ticker_upper


@lru_cache(maxsize=1)
def load_policy_config() -> PolicyConfig:
    with _policy_path().open() as f:
        raw = yaml.safe_load(f) or {}

    prox = raw.get("proximity_model", {})
    fixed = raw.get("fixed_350_template", {})
    symbol_models_raw = fixed.get("symbol_tick_model", {})
    symbol_models: dict[str, SymbolTickModel] = {}
    for symbol, model in symbol_models_raw.items():
        symbol_models[str(symbol).upper()] = SymbolTickModel(
            tick_size=float(model["tick_size"]),
            ticks_per_point=float(model["ticks_per_point"]),
            dollars_per_tick=float(model["dollars_per_tick"]),
        )

    template_unit = str(fixed.get("unit", "ticks")).lower()
    if template_unit not in {"ticks", "points"}:
        raise ValueError(
            f"fixed_350_template.unit must be either 'ticks' or 'points', got '{template_unit}'"
        )

    quality_raw = raw.get("extraction_quality", {})
    instrument_price_floor: dict[str, float] = {
        str(k).upper(): float(v)
        for k, v in (quality_raw.get("instrument_price_floor", {}) or {}).items()
    }

    return PolicyConfig(
        daily_profit_cap_usd=float(raw["daily_profit_cap_usd"]),
        lockout_reset_timezone=str(raw["lockout_reset_timezone"]),
        lockout_reset_time=str(raw["lockout_reset_time"]),
        default_rr=float(raw["default_rr"]),
        min_confidence_for_action=float(raw["min_confidence_for_action"]),
        allow_only_nearby_structure=bool(raw["allow_only_nearby_structure"]),
        proximity_model_type=str(prox.get("type", "min_of_atr_and_ticks")),
        proximity_atr_multiple=float(prox.get("atr_multiple", 0.25)),
        proximity_max_ticks_by_symbol={
            str(k).upper(): int(v) for k, v in (prox.get("max_ticks_by_symbol", {}) or {}).items()
        },
        template_value=int(fixed.get("value", 350)),
        template_unit=template_unit,
        symbol_tick_model=symbol_models,
        minimum_distinct_levels=int(quality_raw.get("minimum_distinct_levels", 1)),
        instrument_price_floor=instrument_price_floor,
    )


def _compute_template_view(config: PolicyConfig, contract_ticker: str) -> PolicyTemplateView:
    symbol = _resolve_symbol_key(ticker=contract_ticker, symbol_map=config.symbol_tick_model)
    model = config.symbol_tick_model.get(symbol)
    if model is None:
        return PolicyTemplateView(
            symbol=symbol,
            value=config.template_value,
            unit=config.template_unit,
            tick_size=None,
            ticks_per_point=None,
            dollars_per_tick=None,
            template_ticks=None,
            template_price_distance=None,
            estimated_risk_usd=None,
            estimated_reward_usd=None,
        )

    template_ticks = config.template_value if config.template_unit == "ticks" else int(
        round(config.template_value * model.ticks_per_point)
    )
    template_price_distance = round(template_ticks * model.tick_size, 6)
    estimated_risk_usd = round(template_ticks * model.dollars_per_tick, 2)
    estimated_reward_usd = round(estimated_risk_usd * config.default_rr, 2)
    return PolicyTemplateView(
        symbol=symbol,
        value=config.template_value,
        unit=config.template_unit,
        tick_size=model.tick_size,
        ticks_per_point=model.ticks_per_point,
        dollars_per_tick=model.dollars_per_tick,
        template_ticks=template_ticks,
        template_price_distance=template_price_distance,
        estimated_risk_usd=estimated_risk_usd,
        estimated_reward_usd=estimated_reward_usd,
    )


def _nearby_structure_threshold(
    config: PolicyConfig, contract_ticker: str, atr14: float | None
) -> float | None:
    symbol = _resolve_symbol_key(
        ticker=contract_ticker, symbol_map=config.proximity_max_ticks_by_symbol
    )
    max_ticks = config.proximity_max_ticks_by_symbol.get(symbol)
    tick_model = config.symbol_tick_model.get(symbol)

    atr_threshold = atr14 * config.proximity_atr_multiple if atr14 and atr14 > 0 else None
    has_tick_cap = max_ticks is not None and tick_model is not None
    tick_threshold = (max_ticks * tick_model.tick_size) if has_tick_cap else None

    thresholds = [x for x in (atr_threshold, tick_threshold) if x is not None and x > 0]
    if not thresholds:
        return None
    return min(thresholds) if config.proximity_model_type == "min_of_atr_and_ticks" else thresholds[0]


def enforce_scalper_policy(
    result: AnalysisResult,
    current_price: float,
    contract_ticker: str,
    realized_pnl_usd: float | None = None,
    atr14: float | None = None,
) -> PolicyDecision:
    config = load_policy_config()
    confidence_value = _to_confidence_value(result.confidence)
    reasons: list[str] = []
    enforced = result.action_state
    lockout = False

    if realized_pnl_usd is not None and realized_pnl_usd >= config.daily_profit_cap_usd:
        lockout = True
        enforced = ActionState.STAND_DOWN
        reasons.append(
            f"Daily lockout active: realized PnL ${realized_pnl_usd:.2f} reached cap ${config.daily_profit_cap_usd:.2f}."
        )

    if confidence_value < config.min_confidence_for_action:
        enforced = ActionState.STAND_DOWN
        reasons.append(
            f"Confidence {confidence_value:.2f} below minimum {config.min_confidence_for_action:.2f}."
        )

    nearby_threshold = _nearby_structure_threshold(
        config=config, contract_ticker=contract_ticker, atr14=atr14
    )

    if (
        config.allow_only_nearby_structure
        and result.action_state != ActionState.STAND_DOWN
        and nearby_threshold is not None
    ):
        if (
            result.action_state == ActionState.ACTIVE_LONG
            and result.strongest_support
            and len(result.strongest_support) > 0
        ):
            dist = abs(result.strongest_support[0].price - current_price)
            if dist > nearby_threshold:
                enforced = ActionState.STAND_DOWN
                reasons.append(
                    f"Support edge unclear: nearest support distance {dist:.2f} exceeds nearby threshold {nearby_threshold:.2f}."
                )
        elif (
            result.action_state == ActionState.ACTIVE_SHORT
            and result.strongest_resistance
            and len(result.strongest_resistance) > 0
        ):
            dist = abs(result.strongest_resistance[0].price - current_price)
            if dist > nearby_threshold:
                enforced = ActionState.STAND_DOWN
                reasons.append(
                    f"Resistance edge unclear: nearest resistance distance {dist:.2f} exceeds nearby threshold {nearby_threshold:.2f}."
                )

    if enforced == ActionState.STAND_DOWN and not reasons:
        reasons.append("Edge/structure unclear under strict scalper policy.")

    template = _compute_template_view(config=config, contract_ticker=contract_ticker)
    return PolicyDecision(
        original_action_state=result.action_state,
        enforced_action_state=enforced,
        lockout_active=lockout,
        confidence_value=confidence_value,
        confidence_threshold=config.min_confidence_for_action,
        nearby_structure_threshold=nearby_threshold,
        stand_down_reasons=reasons,
        rr_target=config.default_rr,
        daily_profit_cap_usd=config.daily_profit_cap_usd,
        lockout_reset_timezone=config.lockout_reset_timezone,
        lockout_reset_time=config.lockout_reset_time,
        template=template,
    )


def check_extraction_quality_gates(
    ticker: str,
    level_prices: List[float],
    current_price: Optional[float] = None,
    axis_min: Optional[float] = None,
    axis_max: Optional[float] = None,
) -> ExtractionQualityResult:
    """Apply instrument-aware sanity checks to image-extracted price levels.

    Returns an :class:`ExtractionQualityResult` whose ``passed`` flag is
    ``True`` only when all of the following gates pass:

    1. **Instrument price floor** – every extracted level must be >= the
       configured minimum for the matched symbol (e.g. YM levels below
       10 000 are impossible and signal an OCR scale error).
    2. **Axis-bounds enforcement** – when OCR axis min/max are available,
       every level must lie within the axis range (+-10 % extrapolation
       tolerance).
    3. **Minimum distinct levels** – the number of supplied levels must
       meet ``minimum_distinct_levels`` from config.

    Parameters
    ----------
    ticker:
        Instrument ticker; used to look up symbol-specific thresholds.
    level_prices:
        Flat list of all extracted price levels (support and resistance
        combined) mapped from the chart image.
    current_price:
        Estimated current price used for a plausibility cross-check
        (optional).
    axis_min:
        Lowest price label successfully OCR'd from the chart axis
        (optional).
    axis_max:
        Highest price label successfully OCR'd from the chart axis
        (optional).
    """
    config = load_policy_config()
    reasons: List[str] = []
    symbol = _resolve_symbol_key(ticker=ticker, symbol_map=config.instrument_price_floor)
    floor = config.instrument_price_floor.get(symbol)

    # Gate 1: instrument price floor – reject impossible scale values
    if floor is not None and floor > 0:
        below_floor = [p for p in level_prices if p < floor]
        if below_floor:
            sample = sorted(below_floor)[:3]
            reasons.append(
                f"Impossible scale: {len(below_floor)} extracted level(s) for {symbol} "
                f"below instrument price floor {floor:.0f} "
                f"(examples: {[round(p, 4) for p in sample]}). "
                "Likely OCR/scale mapping error. Rejecting extraction."
            )

    # Gate 2: axis-bounds enforcement
    if axis_min is not None and axis_max is not None and axis_min < axis_max:
        tolerance = (axis_max - axis_min) * 0.10
        lo = axis_min - tolerance
        hi = axis_max + tolerance
        out_of_bounds = [p for p in level_prices if not (lo <= p <= hi)]
        if out_of_bounds:
            sample = sorted(out_of_bounds)[:3]
            reasons.append(
                f"Out-of-axis-bounds: {len(out_of_bounds)} level(s) outside "
                f"axis range [{axis_min:.2f}, {axis_max:.2f}] +-10 % "
                f"(examples: {[round(p, 4) for p in sample]}). "
                "Rejecting extraction."
            )

    # Gate 3: minimum distinct levels
    distinct = len(set(level_prices))
    if distinct < config.minimum_distinct_levels:
        reasons.append(
            f"Insufficient visual evidence: only {distinct} distinct price level(s) "
            f"extracted (minimum required: {config.minimum_distinct_levels}). "
            "Stand down until edge is clear."
        )

    return ExtractionQualityResult(passed=len(reasons) == 0, rejection_reasons=reasons)
