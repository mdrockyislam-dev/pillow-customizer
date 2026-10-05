import fs from 'node:fs/promises';
import { createReadStream, createWriteStream } from 'node:fs';
import crypto from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';

const modelDir = process.env.SUBJECT_MODEL_DIR ||
  fileURLToPath(new URL('../models/subject/', import.meta.url));
const destination = path.join(modelDir, 'model.onnx');
const expected = (await fs.readFile(path.join(modelDir, 'model.sha256'), 'utf8')).trim();
const url = 'https://huggingface.co/Xenova/detr-resnet-50-panoptic/resolve/main/onnx/model.onnx';

async function sha256(filename) {
  const hash = crypto.createHash('sha256');
  for await (const chunk of createReadStream(filename)) hash.update(chunk);
  return hash.digest('hex');
}

await fs.mkdir(modelDir, { recursive: true });
let cached = false;
try { cached = (await sha256(destination)) === expected; }
catch (error) { if (error.code !== 'ENOENT') throw error; }
if (cached) {
  console.log('Subject model already cached and checksum verified.');
} else {
  const temporary = `${destination}.download`;
  let completed = false;
  for (let attempt = 1; attempt <= 3; attempt++) {
    try {
      console.log(`Downloading subject model during build (attempt ${attempt})...`);
      const response = await fetch(url, { signal: AbortSignal.timeout(180_000) });
      if (!response.ok || !response.body) throw new Error(`Model download HTTP ${response.status}`);
      await pipeline(Readable.fromWeb(response.body), createWriteStream(temporary));
      if ((await sha256(temporary)) !== expected) throw new Error('Subject model checksum differs from the tested model.');
      await fs.rename(temporary, destination);
      completed = true;
      console.log('Subject model downloaded and checksum verified.');
      break;
    } catch (error) {
      await fs.rm(temporary, { force: true });
      if (attempt === 3) throw error;
      console.warn(`Subject model download failed: ${error.message}`);
    }
  }
  if (!completed) throw new Error('Subject model cache was not created.');
}
