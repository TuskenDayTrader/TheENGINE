# AnalysisPayload — Input Contract

The `AnalysisPayload` is the canonical input structure consumed by every
scoring module in TheENGINE. Every field listed below is **required**.

---

## Top-level fields

| Field | Type | Constraints | Description |
|---|---|---|---|
| `date_et` | string | `YYYY-MM-DD` | Analysis date in ET timezone |
| `ticker` | string | non-blank | Instrument symbol (e.g. `NQU2026`, `ESU2026`) |
| `timeframe` | string | non-blank | Chart timeframe (e.g. `5m`, `30m`, `1h`) |
| `lookback_days` | integer | `>= 1` | Calendar days of history to include |
| `current_price` | number | `> 0` | Market price at time of analysis |
| `levels` | object | see below | All required price-level anchors |

---

## `levels` object

All sub-fields are **required** numbers.  
Every H/L pair must satisfy `high >= low` or validation will fail.

| Field | Type | Constraint | Description |
|---|---|---|---|
| `pdh` | number | `pdh >= pdl` | Prior day high |
| `pdl` | number | — | Prior day low |
| `prior_settle` | number | — | Prior session settlement price |
| `rth_open` | number | — | Regular trading hours open |
| `globex_high` | number | `>= globex_low` | Globex (overnight) session high |
| `globex_low` | number | — | Globex (overnight) session low |
| `asia_high` | number | `>= asia_low` | Asia session high |
| `asia_low` | number | — | Asia session low |
| `london_high` | number | `>= london_low` | London session high |
| `london_low` | number | — | London session low |
| `ny_high` | number | `>= ny_low` | New York session high |
| `ny_low` | number | — | New York session low |
| `asia_ib_high` | number | `>= asia_ib_low` | Asia initial balance high |
| `asia_ib_low` | number | — | Asia initial balance low |
| `london_ib_high` | number | `>= london_ib_low` | London initial balance high |
| `london_ib_low` | number | — | London initial balance low |
| `ny_ib_high` | number | `>= ny_ib_low` | New York initial balance high |
| `ny_ib_low` | number | — | New York initial balance low |
| `atr14` | number | `> 0` | 14-period ATR |

---

## Valid example

```json
{
  "date_et": "2026-07-21",
  "ticker": "NQU2026",
  "timeframe": "30m",
  "lookback_days": 14,
  "current_price": 29142.25,
  "levels": {
    "pdh": 29198.50,
    "pdl": 28701.25,
    "prior_settle": 29117.25,
    "rth_open": 29125.00,
    "globex_high": 29195.25,
    "globex_low": 28701.00,
    "asia_high": 28804.00,
    "asia_low": 28701.00,
    "london_high": 29195.25,
    "london_low": 28803.00,
    "ny_high": 29191.25,
    "ny_low": 28751.25,
    "asia_ib_high": 28780.00,
    "asia_ib_low": 28720.00,
    "london_ib_high": 29100.00,
    "london_ib_low": 28950.00,
    "ny_ib_high": 29191.00,
    "ny_ib_low": 28906.00,
    "atr14": 57.75
  }
}
```

This payload is also committed at [`examples/nq_sample_valid.json`](../../examples/nq_sample_valid.json).

---

## Invalid example

```json
{
  "date_et": "2026-07-21",
  "timeframe": "30m",
  "lookback_days": 14,
  "current_price": -999,
  "levels": {
    "pdh": "not-a-number",
    "pdl": 28701.25,
    "prior_settle": 29117.25,
    "rth_open": 29125.00,
    "globex_high": 29195.25,
    "globex_low": 28701.00,
    "asia_high": 28804.00,
    "asia_low": 28701.00,
    "london_high": 29195.25,
    "london_low": 28803.00,
    "ny_high": 29191.25,
    "ny_low": 28751.25,
    "asia_ib_high": 28780.00,
    "asia_ib_low": 28720.00,
    "london_ib_high": 29100.00,
    "london_ib_low": 28950.00,
    "ny_ib_high": 29191.00,
    "ny_ib_low": 28906.00
  }
}
```

### Expected errors for invalid example

| Error path | Description |
|---|---|
| `ticker` | Field required |
| `current_price` | Input should be greater than 0 |
| `levels.pdh` | Input should be a valid number, unable to parse string as a number |
| `levels.atr14` | Field required |

---

## Python usage

```python
from packages.core.validators import validate_payload, PayloadValidationError

try:
    payload = validate_payload(raw_dict)
    print(payload.ticker)
except PayloadValidationError as e:
    # e.field_errors is a list of "field.path: description" strings
    for error in e.field_errors:
        print(error)
```
