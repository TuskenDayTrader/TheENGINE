// @ts-check

const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');

const dashboardPort = Number.parseInt(process.env.PORT ?? '3000', 10);
const indexPath = path.join(__dirname, 'index.html');

/**
 * @returns {http.Server}
 */
function createDashboardServer() {
  return http.createServer((_request, response) => {
    response.writeHead(200, { 'content-type': 'text/html; charset=utf-8' });
    response.end(fs.readFileSync(indexPath, 'utf8'));
  });
}

if (require.main === module) {
  const server = createDashboardServer();
  server.listen(dashboardPort, () => {
    console.log(`TheENGINE dashboard bootstrap running at http://127.0.0.1:${dashboardPort}`);
  });
}

module.exports = {
  createDashboardServer,
  dashboardPort
};
