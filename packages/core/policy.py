"""
Scalper policy engine for TheENGINE.

Enforces strict trading discipline on every analysis decision:
  1. Daily PnL cap ($550 by default) — hard lock once reached.
  2. Confidence floor (0.70) — stand down on low-confidence extractions.
  3. Nearby-structure gate — stand down when price is too far from a level.
  4. 1:1 RR template — all ACTIVE signals use the 350-tick fixed template.

Public API
----------
- ``enforce_scalper_policy(...)``  – main gate function.
- ``check_extraction_quality_gates(...)`` – validate extracted price levels.
- ``PolicyDecision``  – structured result dataclass.
- ``QualityGateResult`` – result of extraction quality check.

Action states produced
----------------------
- ``ACTIVE_LONG_1TO1``         – long, 1:1 RR, 350-tick template applied.
- ``ACTIVE_SHORT_1TO1``        – short, 1:1 RR, 350-tick template applied.
- ``STAND_DOWN``               – no-trade: failed a policy gate.
- ``STOP_TRADING_DAY``         – hard lock: daily PnL cap reached.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    import yaml as _yaml

    _YAML_AVAILABLE = True
except ImportError:  # pragma: no cover
    _YAML_AVAILABLE = False
    _yaml = None  # type: ignore[assignment]

_CONFIG_PATH = pathlib.Path(__file__).parents[2] / "config" / "trading_policy.yaml"


# ---------------------------------------------------------------------------
# Configuration loader (lazy singleton)
# ---------------------------------------------------------------------------


def _load_config(path: pathlib.Path = _CONFIG_PATH) -> dict:
    """Load YAML policy config.  Returns an empty dict on any failure."""
    if not _YAML_AVAILABLE:
        return {}
    try:
        with open(path) as fh:
            return _yaml.safe_load(fh) or {}
    except Exception:
        return {}


_CACHED_CONFIG: dict = {}


def _cfg() -> dict:
    global _CACHED_CONFIG
    if not _CACHED_CONFIG:
        _CACHED_CONFIG = _load_config()
    return _CACHED_CONFIG


def reload_config(path: pathlib.Path = _CONFIG_PATH) -> None:
    """Force a config reload (useful in tests)."""
    global _CACHED_CONFIG
    _CACHED_CONFIG = _load_config(path)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class Template350:
    """Instrument-specific trade template derived from the 350-tick intent."""

    symbol: str
    template_price_distance: float
    estimated_risk_usd: float


@dataclass
class PolicyDecision:
    """Result of ``enforce_scalper_policy``."""

    #: Enforced action state (may differ from the raw analysis state).
    enforced_action_state: str

    #: True when the daily PnL cap has been reached.
    lockout_active: bool

    #: Human-readable reason(s) for STAND_DOWN / STOP_TRADING_DAY.
    stand_down_reasons: List[str] = field(default_factory=list)

    #: Risk-reward ratio applied to ACTIVE signals.
    rr_target: float = 1.0

    #: 350-tick template for the requested symbol (always populated).
    template_350: Optional[Template350] = None


@dataclass
class QualityGateResult:
    """Result of ``check_extraction_quality_gates``."""

    passed: bool
    rejection_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_template(config: dict, symbol: str) -> Optional[Template350]:
    """Build a ``Template350`` for *symbol* from the loaded config."""
    tpl_cfg = config.get("fixed_distance_template", {})
    by_sym = tpl_cfg.get("values_by_symbol", {})
    sym_key = (symbol or "").upper()
    entry = by_sym.get(sym_key)
    if not entry:
        return None
    points = float(entry.get("points", 0))
    ticks = int(entry.get("ticks", 0))
    dpt = float(entry.get("dollars_per_tick", 0))
    risk_usd = round(ticks * dpt, 2)
    return Template350(
        symbol=sym_key,
        template_price_distance=points,
        estimated_risk_usd=risk_usd,
    )


def _proximity_limit(config: dict, symbol: str, atr14: Optional[float]) -> float:
    """Return the maximum distance (price units) a level may be from price."""
    prox = config.get("proximity_model", {})
    if atr14 and atr14 > 0:
        mult = float(prox.get("atr_multiple", 1.0))
        return atr14 * mult

    # Fallback: ticks → points (uses per-symbol tick_size from template config).
    # When the symbol is unknown, ticks default to 160 and tick_size to 1.0
    # (1 point per tick), giving a 160-point proximity limit as a safe default.
    max_ticks = prox.get("max_ticks_by_symbol", {})
    sym_key = (symbol or "").upper()
    ticks = max_ticks.get(sym_key, 160)

    # Derive tick_size from template config; default 1.0 (= 1 pt per tick)
    # when the symbol is not in the template table.
    tpl = config.get("fixed_distance_template", {}).get("values_by_symbol", {})
    entry = tpl.get(sym_key, {})
    tick_size = float(entry.get("tick_size", 1.0))
    return ticks * tick_size


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def enforce_scalper_policy(
    ticker: str,
    action_state: str,
    confidence: float,
    nearest_resistance: Optional[float] = None,
    nearest_support: Optional[float] = None,
    current_price: Optional[float] = None,
    atr14: Optional[float] = None,
    realized_pnl_today: float = 0.0,
) -> PolicyDecision:
    """
    Apply scalper policy gates in order and return a ``PolicyDecision``.

    Parameters
    ----------
    ticker:
        Instrument symbol (NQ, ES, YM, RTY …).
    action_state:
        Raw analysis action state: ``ACTIVE_LONG``, ``ACTIVE_SHORT``, or
        ``STAND_DOWN``.
    confidence:
        Extraction/analysis confidence in [0.0, 1.0].
    nearest_resistance:
        Price of the closest resistance level above current price.
    nearest_support:
        Price of the closest support level below current price.
    current_price:
        Estimated current market price.
    atr14:
        14-period ATR for proximity normalisation.
    realized_pnl_today:
        Realised PnL so far today in USD (used for daily cap gate).

    Returns
    -------
    A :class:`PolicyDecision` with the enforced state and rationale.
    """
    config = _cfg()
    daily_cap = float(config.get("daily_profit_cap_usd", 550))
    min_confidence = float(config.get("min_confidence_for_action", 0.70))
    default_rr = float(config.get("default_rr", 1.0))
    allow_only_nearby = bool(config.get("allow_only_nearby_structure", True))

    template = _get_template(config, ticker)
    stop_dist = template.template_price_distance if template else None
    target_dist = stop_dist  # 1:1 RR

    # ── Gate 1: Daily PnL cap ─────────────────────────────────────────────
    if realized_pnl_today >= daily_cap:
        return PolicyDecision(
            enforced_action_state="STOP_TRADING_DAY",
            lockout_active=True,
            stand_down_reasons=[
                f"Daily profit cap of ${daily_cap:.0f} reached. "
                "No new trades until midnight ET reset."
            ],
            rr_target=default_rr,
            template_350=template,
        )

    reasons: List[str] = []

    # ── Gate 2: Confidence ────────────────────────────────────────────────
    if confidence < min_confidence:
        reasons.append(
            f"Confidence {confidence:.2f} below minimum {min_confidence:.2f}."
        )

    # ── Gate 3: Nearby structure (active signals only) ────────────────────
    if action_state in ("ACTIVE_LONG", "ACTIVE_SHORT") and allow_only_nearby:
        if current_price is not None:
            limit = _proximity_limit(config, ticker, atr14)
            if action_state == "ACTIVE_LONG":
                ref_level = nearest_resistance
                label = "resistance"
            else:
                ref_level = nearest_support
                label = "support"

            if ref_level is None:
                reasons.append(
                    f"No qualified {label} level found near current price."
                )
            else:
                dist = abs(ref_level - current_price)
                if dist > limit:
                    reasons.append(
                        f"Nearest {label} {dist:.2f} pts from price exceeds "
                        f"proximity limit {limit:.2f} pts. Stand down."
                    )

    if reasons:
        return PolicyDecision(
            enforced_action_state="STAND_DOWN",
            lockout_active=False,
            stand_down_reasons=reasons,
            rr_target=default_rr,
            template_350=template,
        )

    # ── All gates passed → enforce 1:1 template ───────────────────────────
    if action_state == "ACTIVE_LONG":
        enforced = "ACTIVE_LONG_1TO1"
    elif action_state == "ACTIVE_SHORT":
        enforced = "ACTIVE_SHORT_1TO1"
    else:
        enforced = "STAND_DOWN"

    return PolicyDecision(
        enforced_action_state=enforced,
        lockout_active=False,
        stand_down_reasons=[],
        rr_target=default_rr,
        template_350=template,
    )


def check_extraction_quality_gates(
    ticker: str,
    levels: List[float],
    axis_min: Optional[float] = None,
    axis_max: Optional[float] = None,
) -> QualityGateResult:
    """
    Validate extracted price levels against quality gates.

    Gates applied (in order):
    1. **Instrument price floor** – levels below the configured floor for
       the symbol are physically impossible (e.g. YM = 2.41).
    2. **Axis-bounds** – levels outside the OCR axis range ±10 % tolerance
       are rejected as extrapolation artefacts.
    3. **Minimum distinct levels** – too few levels to form a useful decision.

    Parameters
    ----------
    ticker:
        Instrument symbol used to look up the price floor.
    levels:
        List of extracted price values (resistance + support combined).
    axis_min, axis_max:
        Lowest and highest OCR-parsed price-axis values.  When provided,
        levels outside [axis_min × 0.90, axis_max × 1.10] are rejected.

    Returns
    -------
    A :class:`QualityGateResult`.
    """
    config = _cfg()
    eq = config.get("extraction_quality", {})
    min_distinct = int(eq.get("minimum_distinct_levels", 1))
    floors: Dict[str, float] = eq.get("instrument_price_floor", {})

    sym_key = (ticker or "UNKNOWN").upper()
    floor = floors.get(sym_key)

    # Gate 1: Impossible scale
    if floor is not None:
        bad = [p for p in levels if p < floor]
        if bad:
            examples = ", ".join(f"{p:.2f}" for p in bad[:3])
            return QualityGateResult(
                passed=False,
                rejection_reason=(
                    f"Impossible scale: extracted level(s) {examples} below "
                    f"instrument floor {floor:.0f} for {sym_key}."
                ),
            )

    # Gate 2: Axis-bounds
    if axis_min is not None and axis_max is not None and levels:
        lo = axis_min * 0.90
        hi = axis_max * 1.10
        out = [p for p in levels if p < lo or p > hi]
        if out:
            examples = ", ".join(f"{p:.2f}" for p in out[:3])
            return QualityGateResult(
                passed=False,
                rejection_reason=(
                    f"Levels {examples} outside axis bounds "
                    f"[{lo:.2f}, {hi:.2f}]."
                ),
            )

    # Gate 3: Minimum distinct levels
    distinct = len(set(round(p, 2) for p in levels))
    if distinct < min_distinct:
        return QualityGateResult(
            passed=False,
            rejection_reason=(
                f"Only {distinct} distinct level(s) extracted; "
                f"minimum required is {min_distinct}."
            ),
        )

    return QualityGateResult(passed=True)
