# AnalysisResult — Output Contract

The `AnalysisResult` is the canonical output produced by the scoring engine.
It carries exactly 2 items in each of the four level buckets (2+2+2+2),
a directional action state, an aggregate confidence score, and a rationale list.

---

## Top-level fields

| Field | Type | Constraints | Description |
|---|---|---|---|
| `strongest_resistance` | array of `LevelItem` | exactly 2 items | Levels most likely to reject price |
| `weakest_resistance` | array of `LevelItem` | exactly 2 items | Resistance most likely to break up through |
| `strongest_support` | array of `LevelItem` | exactly 2 items | Levels most likely to bounce price |
| `weakest_support` | array of `LevelItem` | exactly 2 items | Support most likely to fail down through |
| `action_state` | enum string | see below | Session directional bias |
| `confidence` | number | `0.0 – 1.0` | Aggregate engine confidence |
| `rationale` | array of strings | `>= 1` item | Ordered reasoning statements |

---

## `action_state` enum

| Value | Meaning |
|---|---|
| `ACTIVE_LONG` | Bias is long; engine sees acceptance above key level |
| `ACTIVE_SHORT` | Bias is short; engine sees rejection / failure below key level |
| `STAND_DOWN` | No clear directional edge; avoid initiating new positions |

---

## `LevelItem` schema

Each entry in every bucket must include all four fields below.

| Field | Type | Description |
|---|---|---|
| `level` | string | Price level or range (e.g. `"29195-29205"` or `"29195.25"`) |
| `source` | string | Origin label (e.g. `"PDH"`, `"Globex High"`, `"NY IB High"`) |
| `conviction` | string | Confidence tag (e.g. `"HIGH"`, `"MEDIUM"`, `"LOW"`) |
| `trigger_note` | string | Brief trigger condition or invalidation note |

---

## Valid example

```json
{
  "strongest_resistance": [
    {
      "level": "29195-29205",
      "source": "PDH",
      "conviction": "HIGH",
      "trigger_note": "Rejection expected on first test; acceptance above triggers long."
    },
    {
      "level": "29320-29350",
      "source": "Globex High",
      "conviction": "HIGH",
      "trigger_note": "Extended range target only if 29205 breaks and holds."
    }
  ],
  "weakest_resistance": [
    {
      "level": "29115-29125",
      "source": "NY IB High",
      "conviction": "LOW",
      "trigger_note": "Likely to break on any early-session momentum."
    },
    {
      "level": "29165-29175",
      "source": "London High",
      "conviction": "LOW",
      "trigger_note": "Single-touch level, no prior confluence."
    }
  ],
  "strongest_support": [
    {
      "level": "29000-29020",
      "source": "Prior Settle",
      "conviction": "HIGH",
      "trigger_note": "High-volume node; failure triggers cascade to PDL."
    },
    {
      "level": "28885-28905",
      "source": "PDL",
      "conviction": "HIGH",
      "trigger_note": "Multi-session low; expect significant bounce attempt."
    }
  ],
  "weakest_support": [
    {
      "level": "28955-28970",
      "source": "NY Low",
      "conviction": "LOW",
      "trigger_note": "Isolated overnight low, no session confluence."
    },
    {
      "level": "28760-28780",
      "source": "Asia Low",
      "conviction": "LOW",
      "trigger_note": "Thin area, likely to slice through quickly."
    }
  ],
  "action_state": "STAND_DOWN",
  "confidence": 0.62,
  "rationale": [
    "Price is between PDH and prior settle — no clear directional acceptance.",
    "Globex range is unusually wide; avoid chasing.",
    "STAND DOWN unless acceptance above 29,195 or loss of 29,000 with follow-through."
  ]
}
```

---

## Invalid example

The following result is invalid because `strongest_resistance` has only
1 item instead of the required 2, and `action_state` is not a recognized
enum value.

```json
{
  "strongest_resistance": [
    {
      "level": "29195-29205",
      "source": "PDH",
      "conviction": "HIGH",
      "trigger_note": "Only one item — bucket requires exactly two."
    }
  ],
  "weakest_resistance": [...],
  "strongest_support": [...],
  "weakest_support": [...],
  "action_state": "BUY_THE_DIP",
  "confidence": 0.62,
  "rationale": ["Some text"]
}
```

### Expected errors for invalid example

| Error path | Description |
|---|---|
| `strongest_resistance` | List should have at least 2 items after validation, not 1 |
| `action_state` | Input should be 'ACTIVE_LONG', 'ACTIVE_SHORT' or 'STAND_DOWN' |

---

## Python usage

```python
from packages.core.validators import validate_result, ResultValidationError

try:
    result = validate_result(engine_output_dict)
    print(result.action_state)
    for item in result.strongest_resistance:
        print(item.level, item.conviction)
except ResultValidationError as e:
    # e.field_errors is a list of "field.path: description" strings
    for error in e.field_errors:
        print(error)
```
