# TheENGINE

A compilation of my work Tusken Day Trader.

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
