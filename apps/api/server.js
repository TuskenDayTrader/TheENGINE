// @ts-check

const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');

const repoRoot = path.resolve(__dirname, '..', '..');
const defaultPort = Number.parseInt(process.env.PORT ?? '3001', 10);

/**
 * @param {string} relativePath
 * @returns {string}
 */
function readExample(relativePath) {
  return fs.readFileSync(path.join(repoRoot, 'examples', relativePath), 'utf8');
}

/**
 * @param {http.ServerResponse} response
 * @param {number} statusCode
 * @param {string} body
 * @param {Record<string, string>} [headers]
 */
function send(response, statusCode, body, headers = {}) {
  response.writeHead(statusCode, {
    'content-type': 'application/json; charset=utf-8',
    ...headers
  });
  response.end(body);
}

/**
 * @returns {http.Server}
 */
function createApiServer() {
  return http.createServer((request, response) => {
    if (!request.url) {
      send(response, 400, JSON.stringify({ error: 'Missing request URL.' }));
      return;
    }

    const url = new URL(request.url, 'http://127.0.0.1');

    if (url.pathname === '/health') {
      send(response, 200, JSON.stringify({ status: 'ok', service: 'theengine-api-bootstrap' }));
      return;
    }

    if (url.pathname === '/examples/sample-analysis-payload') {
      send(response, 200, readExample('sample-analysis-payload.json'));
      return;
    }

    if (url.pathname === '/examples/sample-analysis-response') {
      send(response, 200, readExample('sample-analysis-response.json'));
      return;
    }

    send(response, 404, JSON.stringify({ error: 'Not found.' }));
  });
}

if (require.main === module) {
  const server = createApiServer();
  server.listen(defaultPort, () => {
    console.log(`TheENGINE API bootstrap running at http://127.0.0.1:${defaultPort}`);
  });
}

module.exports = {
  createApiServer,
  defaultPort
};
