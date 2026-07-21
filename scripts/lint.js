// @ts-check

const fs = require('node:fs');
const path = require('node:path');

const repoRoot = path.resolve(__dirname, '..');
const requiredPaths = [
  'README.md',
  'docs/developer-setup.md',
  'examples/README.md',
  'examples/sample-analysis-payload.json',
  'examples/sample-analysis-response.json',
  'apps/api/server.js',
  'apps/dashboard/index.html',
  'apps/dashboard/server.js',
  '.github/workflows/ci.yml'
];

/**
 * @param {string} directory
 * @returns {string[]}
 */
function collectJsonFiles(directory) {
  /** @type {string[]} */
  const jsonFiles = [];

  for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
    if (entry.name === '.git' || entry.name === 'node_modules') {
      continue;
    }

    const entryPath = path.join(directory, entry.name);

    if (entry.isDirectory()) {
      jsonFiles.push(...collectJsonFiles(entryPath));
      continue;
    }

    if (entry.isFile() && entry.name.endsWith('.json')) {
      jsonFiles.push(entryPath);
    }
  }

  return jsonFiles;
}

/**
 * @param {string} filePath
 * @returns {string}
 */
function readFile(filePath) {
  return fs.readFileSync(path.join(repoRoot, filePath), 'utf8');
}

/** @type {string[]} */
const errors = [];

for (const requiredPath of requiredPaths) {
  if (!fs.existsSync(path.join(repoRoot, requiredPath))) {
    errors.push(`Missing required path: ${requiredPath}`);
  }
}

for (const jsonFile of collectJsonFiles(repoRoot)) {
  const relativePath = path.relative(repoRoot, jsonFile);

  try {
    JSON.parse(fs.readFileSync(jsonFile, 'utf8'));
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown JSON parsing error';
    errors.push(`Invalid JSON in ${relativePath}: ${message}`);
  }
}

const readme = readFile('README.md');
if (!readme.includes('## Quickstart')) {
  errors.push('README.md must include a Quickstart section.');
}

if (!readme.includes('npm run dev:api') || !readme.includes('npm run dev:app')) {
  errors.push('README.md must document local API and app commands.');
}

const developerSetup = readFile('docs/developer-setup.md');
for (const command of ['npm install', 'npm run lint', 'npm run typecheck', 'npm test']) {
  if (!developerSetup.includes(command)) {
    errors.push(`docs/developer-setup.md must mention \`${command}\`.`);
  }
}

if (errors.length > 0) {
  for (const error of errors) {
    console.error(`- ${error}`);
  }

  process.exitCode = 1;
} else {
  console.log('Repository lint checks passed.');
}
