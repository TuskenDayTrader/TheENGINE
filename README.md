# TheENGINE

A compilation of my work Tusken Day Trader.

## UI theme hooks

Bare-bones UI skinning is configured through `config/ui_theme.yaml`.

```yaml
theme:
  primary_dark: "#0a0a0a"
  primary_mid: "#8b0000"
  primary_bright: "#ff0000"
  accent: "#ff6b6b"
```

Optional environment override:

```bash
# Explicitly keep using the default built-in palette
UI_THEME=hombre_red python -m uvicorn apps.api.main:app --reload --port 8000
```

The repo currently ships one built-in profile (`hombre_red`) in `config/ui_theme.yaml`. If you add more profiles later, name them `config/ui_theme.<name>.yaml` and select them with `UI_THEME=<name>`.

Read the current palette from the API:

```bash
curl http://127.0.0.1:8000/api/theme
```

To customize the UI later, edit `config/ui_theme.yaml` and `frontend/templates/app_hombre.html`.

## Fixture screenshot organization

Screenshot fixtures are organized by ticker and timestamp under `tests/fixtures/screenshots/`:

```text
tests/fixtures/screenshots/
├── YM/2026/07/22/21/44/14_YM1!_2026-07-22_21-44-14.png
├── NQ/2026/07/22/21/43/17_NQ1!_2026-07-22_21-43-17.png
├── ES/2026/07/22/21/43/33_ES1!_2026-07-22_21-43-33.png
├── RTY/2026/07/22/21/43/53_RTY_2026-07-22_21-43-53.png
└── manifest.json
```

### Fixture CLI

Use `scripts/manage_fixtures.py` to add and query fixtures:

```bash
python scripts/manage_fixtures.py add --file ~/Downloads/YM1!_2026-07-22_21-44-14.png
python scripts/manage_fixtures.py list --ticker YM
python scripts/manage_fixtures.py timeline --ticker YM --year 2026 --month 07
python scripts/manage_fixtures.py search --date 2026-07-22
```

`add` enforces housekeeping:
- max 2MB per file
- dedupe by SHA-256 hash within a ticker
- warning when a ticker folder exceeds 100MB
- archive fixtures older than 60 days to `tests/fixtures/screenshots/archived/`

## Run tests

```bash
python -m pytest tests/ -v
```
