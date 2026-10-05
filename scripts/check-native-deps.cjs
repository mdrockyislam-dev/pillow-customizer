const fs = require('node:fs');
const path = require('node:path');
const { createRequire } = require('node:module');
const root = path.resolve(__dirname, '..');
const appRequire = createRequire(path.join(root, 'package.json'));
const bgRequire = createRequire(appRequire.resolve('@imgly/background-removal-node'));
for (const dependency of ['sharp', 'onnxruntime-node']) {
  const appEntry = fs.realpathSync(appRequire.resolve(dependency));
  const bgEntry = fs.realpathSync(bgRequire.resolve(dependency));
  if (appEntry !== bgEntry) throw new Error(`Conflicting native copies of ${dependency}. Use the supplied package-lock.json.`);
  console.log(`${dependency}: server and IMG.LY share ${appEntry}`);
}
console.log('Native dependency check passed.');
