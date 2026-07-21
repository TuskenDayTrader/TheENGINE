// @ts-check

const test = require('node:test');
const assert = require('node:assert/strict');

const { createApiServer } = require('../apps/api/server');

/**
 * @returns {Promise<{ server: import('node:http').Server, baseUrl: string }>}
 */
async function startServer() {
  const server = createApiServer();

  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', () => resolve(undefined));
  });

  const address = server.address();
  if (!address || typeof address === 'string') {
    throw new Error('Server did not expose a numeric port.');
  }

  return {
    server,
    baseUrl: `http://127.0.0.1:${address.port}`
  };
}

test('health endpoint responds with bootstrap status', async () => {
  const { server, baseUrl } = await startServer();

  try {
    const response = await fetch(`${baseUrl}/health`);
    assert.equal(response.status, 200);
    assert.deepEqual(await response.json(), {
      status: 'ok',
      service: 'theengine-api-bootstrap'
    });
  } finally {
    await new Promise((resolve, reject) => {
      server.close((error) => {
        if (error) {
          reject(error);
          return;
        }

        resolve(undefined);
      });
    });
  }
});

test('sample payload route serves the example artifact', async () => {
  const { server, baseUrl } = await startServer();

  try {
    const response = await fetch(`${baseUrl}/examples/sample-analysis-payload`);
    assert.equal(response.status, 200);

    const payload = await response.json();
    assert.equal(payload.ticker, 'NQU2026');
    assert.equal(payload.timeframe, '30m');
  } finally {
    await new Promise((resolve, reject) => {
      server.close((error) => {
        if (error) {
          reject(error);
          return;
        }

        resolve(undefined);
      });
    });
  }
});
