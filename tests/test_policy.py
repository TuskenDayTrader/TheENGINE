from __future__ import annotations

import pytest

from packages.core.policy import PolicyConfig, SymbolTickModel, _compute_template_view


def test_template_reward_uses_rr_multiplier():
    # Use a non-default RR to prove reward uses RR math, not a copied risk value.
    config = PolicyConfig(
        daily_profit_cap_usd=550.0,
        lockout_reset_timezone="America/New_York",
        lockout_reset_time="00:00",
        default_rr=2.0,
        min_confidence_for_action=0.70,
        allow_only_nearby_structure=True,
        proximity_model_type="min_of_atr_and_ticks",
        proximity_atr_multiple=0.25,
        proximity_max_ticks_by_symbol={"NQ": 40},
        template_value=350,
        template_unit="ticks",
        symbol_tick_model={
            "NQ": SymbolTickModel(tick_size=0.25, ticks_per_point=4.0, dollars_per_tick=5.0)
        },
    )
    template = _compute_template_view(config=config, contract_ticker="NQU2026")
    assert template.estimated_risk_usd == pytest.approx(1750.0)
    assert template.estimated_reward_usd == pytest.approx(template.estimated_risk_usd * 2.0)
