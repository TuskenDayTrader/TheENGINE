# TheENGINE
A compilation of my work Tusken Day Trader
April and May Days
Prompt Iterations
models and versions

---

## /analyze API – Quickstart

TheENGINE provides a FastAPI endpoint that runs the **2+2+2+2 confluence scoring engine** on any NQ/ES futures session and returns the strongest/weakest support & resistance levels, an action state, and a branded poster text block.

### Install & run

```bash
pip install -r requirements.txt
uvicorn apps.api.main:app --reload
```

Interactive docs: <http://localhost:8000/docs>

### curl – NQ sample

```bash
curl -s -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d @examples/nq_sample.json | python -m json.tool
```

### curl – ES sample

```bash
curl -s -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d @examples/es_sample.json | python -m json.tool
```

### Run tests

```bash
pytest tests/test_analyze.py -v
```

### Full API reference

See [`docs/api.md`](docs/api.md) for complete request/response schema, error codes, and more examples.

---

## Project structure

```text
TheENGINE/
  apps/api/
    main.py              # FastAPI app entry point
    routes/analyze.py    # POST /analyze route
  packages/core/
    models.py            # AnalysisPayload, AnalysisResult, LevelDecision
    scoring.py           # 2+2+2+2 scoring engine + ActionState logic
    output.py            # Poster text block formatter
  examples/
    nq_sample.json       # NQ payload example
    es_sample.json       # ES payload example
  tests/
    test_analyze.py      # Endpoint and validation tests
  docs/
    api.md               # Full API reference
  requirements.txt
```
