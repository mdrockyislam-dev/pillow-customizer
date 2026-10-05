// Verify resolution without loading several native copies into one process.
const fs = require('node:fs');
const path = require('node:path');
const { createRequire } = require('node:module');

const root = path.resolve(__dirname, '..');
const rootRequire = createRequire(path.join(root, 'package.json'));
const pkg = rootRequire('./package.json');
const contexts = [
  ['server', rootRequire],
  ['transformers', createRequire(rootRequire.resolve('@huggingface/transformers'))],
  ['background-removal', createRequire(rootRequire.resolve('@imgly/background-removal-node'))]
];

function packageInfo(entry, name) {
  let directory = path.dirname(entry);
  while (true) {
    const manifest = path.join(directory, 'package.json');
    if (fs.existsSync(manifest)) {
      const data = JSON.parse(fs.readFileSync(manifest, 'utf8'));
      if (data.name === name) return { version: data.version, entry };
    }
    const parent = path.dirname(directory);
    if (parent === directory) throw new Error(`Cannot find manifest for ${name}`);
    directory = parent;
  }
}

for (const [name, expected] of [
  ['sharp', pkg.dependencies.sharp],
  ['onnxruntime-node', pkg.overrides['onnxruntime-node']]
]) {
  const resolved = contexts.map(([context, localRequire]) => {
    const info = packageInfo(fs.realpathSync(localRequire.resolve(name)), name);
    console.log(`${context}: ${name}@${info.version} -> ${info.entry}`);
    return info;
  });
  if (new Set(resolved.map(info => info.entry)).size !== 1 ||
      resolved.some(info => info.version !== expected)) {
    throw new Error(`Conflicting ${name} installations. Install this package-lock.json with npm ci.`);
  }
}
console.log('Native dependencies are shared: sharp and ONNX Runtime have one installation each.');
