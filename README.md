# TheENGINE

TheENGINE is the working repository for trading-system research, archived prompt iterations, and the bootstrap scaffolding needed to start the future dashboard and API quickly.

## Quickstart

```bash
npm install
npm run lint
npm run typecheck
npm test
```

Run the local bootstrap surfaces in separate terminals:

```bash
npm run dev:api
npm run dev:app
```

- API: `http://127.0.0.1:3001/health`
- Dashboard: `http://127.0.0.1:3000`
- Examples: `examples/sample-analysis-payload.json` and `examples/sample-analysis-response.json`

More detail lives in [`docs/developer-setup.md`](docs/developer-setup.md).

## Repository layout

- `apps/api/` — minimal local API bootstrap for rapid iteration.
- `apps/dashboard/` — local dashboard shell for upload workflow experiments.
- `docs/` — contributor setup notes.
- `examples/` — sample request/response artifacts.
- `ORB-Strategy/`, `April_2026/`, `agent_collections/` — existing research and archive material.
