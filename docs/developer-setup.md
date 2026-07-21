# Developer setup

## Prerequisites

- Node.js 20 or newer
- npm 10 or newer

## Install dependencies

```bash
npm install
```

## Validate the scaffold

```bash
npm run lint
npm run typecheck
npm test
```

## Run the local API

```bash
npm run dev:api
```

The bootstrap API serves:

- `GET /health`
- `GET /examples/sample-analysis-payload`
- `GET /examples/sample-analysis-response`

## Run the local dashboard

```bash
npm run dev:app
```

Then open `http://127.0.0.1:3000` in your browser.
