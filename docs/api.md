# POST /analyze – API Reference

## Overview

`POST /analyze` is the core endpoint of TheENGINE.  
It accepts a canonical `AnalysisPayload` and returns a scored `AnalysisResult`
with 2+2+2+2 confluence level buckets, an action state, and a branded poster
text block ready for dashboard graphics.

---

## Base URL

```
http://localhost:8000
```

Start the server:

```bash
pip install -r requirements.txt
uvicorn apps.api.main:app --reload
```

Interactive docs: <http://localhost:8000/docs>

---

## Endpoint

### `POST /analyze`

**Request headers**

| Header | Value |
|---|---|
| `Content-Type` | `application/json` |

**Request body** – `AnalysisPayload`

| Field | Type | Required | Notes |
|---|---|---|---|
| `date_et` | string | ✅ | `YYYY-MM-DD` in ET timezone |
| `ticker` | string | ✅ | e.g. `NQU2026`, `ESU2026` |
| `timeframe` | string | ✅ | One of `5m` `15m` `30m` `60m` |
| `lookback_days` | integer | ✅ | 1–365 |
| `current_price` | float | ✅ | Must be > 0 |
| `levels` | object | ✅ | See Levels block below |

**Levels block**

| Field | Description |
|---|---|
| `pdh` / `pdl` | Prior-day high / low |
| `prior_settle` | Prior session settlement |
| `rth_open` | Regular-trading-hours open |
| `globex_high` / `globex_low` | Overnight Globex session |
| `asia_high` / `asia_low` | Asia session extremes |
| `london_high` / `london_low` | London session extremes |
| `ny_high` / `ny_low` | NY session extremes |
| `asia_ib_high` / `asia_ib_low` | Asia initial-balance |
| `london_ib_high` / `london_ib_low` | London initial-balance |
| `ny_ib_high` / `ny_ib_low` | NY initial-balance |
| `atr14` | 14-period ATR (must be > 0) |

All price fields must be positive. Each `high` must exceed its paired `low`.

---

**Response body** – `AnalysisResult`

| Field | Type | Description |
|---|---|---|
| `ticker` | string | Echo of input ticker |
| `date_et` | string | Echo of analysis date |
| `timeframe` | string | Echo of timeframe |
| `current_price` | float | Echo of current price |
| `strongest_resistance` | array[2] | Top-2 closest resistance levels |
| `weakest_resistance` | array[2] | Next-2 resistance levels |
| `strongest_support` | array[2] | Top-2 closest support levels |
| `weakest_support` | array[2] | Next-2 support levels |
| `action_state` | string | `ACTIVE_LONG` \| `ACTIVE_SHORT` \| `STAND_DOWN` |
| `confidence` | float | 0.0 – 1.0 |
| `rationale` | string | Human-readable explanation |
| `poster_text` | string | Branded SESSION CONFLUENCE MAP text block |

Each level decision object contains:

| Field | Description |
|---|---|
| `label` | Level name (e.g. `PDH`, `Globex High`) |
| `price` | Exact price |
| `score` | Composite confluence score |
| `distance_atr` | Distance from current price in ATR units |
| `rationale` | Short scoring rationale |

---

**Error responses**

| Status | Cause |
|---|---|
| `422 Unprocessable Entity` | Missing/invalid field – body contains `detail` array with field-level errors |
| `500 Internal Server Error` | Unexpected scoring failure – safe generic message returned |

---

## curl examples

### NQ (E-mini Nasdaq 100)

```bash
curl -s -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d @examples/nq_sample.json | python -m json.tool
```

### ES (E-mini S&P 500)

```bash
curl -s -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d @examples/es_sample.json | python -m json.tool
```

### Minimal inline example

```bash
curl -s -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "date_et": "2026-07-21",
    "ticker": "NQU2026",
    "timeframe": "30m",
    "lookback_days": 14,
    "current_price": 21850.0,
    "levels": {
      "pdh": 22100.0, "pdl": 21600.0, "prior_settle": 21700.0,
      "rth_open": 21750.0,
      "globex_high": 21900.0, "globex_low": 21580.0,
      "asia_high": 21880.0, "asia_low": 21640.0,
      "london_high": 21920.0, "london_low": 21700.0,
      "ny_high": 21875.0, "ny_low": 21780.0,
      "asia_ib_high": 21855.0, "asia_ib_low": 21720.0,
      "london_ib_high": 21910.0, "london_ib_low": 21740.0,
      "ny_ib_high": 21870.0, "ny_ib_low": 21800.0,
      "atr14": 220.0
    }
  }' | python -m json.tool
```

### Validation error example (missing ticker)

```bash
curl -s -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"date_et": "2026-07-21", "timeframe": "30m", "lookback_days": 14,
       "current_price": 21850.0, "levels": {}}' | python -m json.tool
```

Expected: HTTP 422 with field-level `detail` array.

---

## Health check

```bash
curl http://localhost:8000/health
# {"status":"ok","service":"TheENGINE"}
```

---

## Running tests

```bash
pip install -r requirements.txt
pytest tests/test_analyze.py -v
```
