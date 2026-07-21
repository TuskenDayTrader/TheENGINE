// @ts-check

const test = require('node:test');
const assert = require('node:assert/strict');

const { createDashboardServer } = require('../apps/dashboard/server');

test('dashboard server renders the bootstrap page', async () => {
  const server = createDashboardServer();

  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', () => resolve(undefined));
  });

  const address = server.address();
  if (!address || typeof address === 'string') {
    throw new Error('Dashboard server did not expose a numeric port.');
  }

  try {
    const response = await fetch(`http://127.0.0.1:${address.port}`);
    assert.equal(response.status, 200);

    const html = await response.text();
    assert.match(html, /TheENGINE developer bootstrap dashboard/);
    assert.match(html, /npm run dev:api/);
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
