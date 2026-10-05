import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import sharp from 'sharp';
import * as ort from 'onnxruntime-node';

// One native ONNX Runtime shared with IMG.LY; no second Sharp/ML runtime.
export const SUBJECT_MODEL = 'Xenova/detr-resnet-50-panoptic';
export const SUBJECT_MODEL_DIR = process.env.SUBJECT_MODEL_DIR ||
  fileURLToPath(new URL('../models/subject/', import.meta.url));
const INPUT_SIDE = Math.max(384, Math.min(768, Number(process.env.SUBJECT_INPUT_SIDE || 512)));
const SCORE_THRESHOLD = Math.max(0.1, Math.min(0.95, Number(process.env.DETECTION_THRESHOLD || 0.55)));
const MASK_THRESHOLD = 0.28;
const PETS = new Set(['dog', 'cat', 'bird', 'horse', 'cow', 'sheep', 'bear', 'elephant', 'zebra', 'giraffe']);
let modelPromise = null;
let ready = false;

export function subjectDetectorReady() { return ready; }

export async function warmSubjectDetector() {
  if (!modelPromise) {
    modelPromise = (async () => {
      const config = JSON.parse(await fs.readFile(path.join(SUBJECT_MODEL_DIR, 'config.json'), 'utf8'));
      const processor = JSON.parse(await fs.readFile(path.join(SUBJECT_MODEL_DIR, 'preprocessor_config.json'), 'utf8'));
      const session = await ort.InferenceSession.create(path.join(SUBJECT_MODEL_DIR, 'model.onnx'), {
        executionProviders: ['cpu'], graphOptimizationLevel: 'all',
        executionMode: 'sequential', intraOpNumThreads: 1, interOpNumThreads: 1,
        enableCpuMemArena: false, enableMemPattern: false,
        logSeverityLevel: 3
      });
      ready = true;
      console.log('PP3D main-subject detector ready (DETR panoptic fp32).');
      return { config, processor, session };
    })().catch(error => { modelPromise = null; ready = false; throw error; });
  }
  return modelPromise;
}

export async function detectMainSubject(buffer, id, logStage) {
  logStage(id, 'subject:start');
  const { config, processor, session } = await warmSubjectDetector();
  const { data, info } = await sharp(buffer).removeAlpha().toColourspace('srgb')
    .resize({ width: INPUT_SIDE, height: INPUT_SIDE, fit: 'inside', withoutEnlargement: true, kernel: 'cubic' })
    .raw().toBuffer({ resolveWithObject: true });
  const pixels = info.width * info.height;
  const rgb = new Float32Array(pixels * 3);
  for (let i = 0; i < pixels; i++) {
    for (let c = 0; c < 3; c++) {
      const value = data[i * info.channels + c] * processor.rescale_factor;
      rgb[c * pixels + i] = (value - processor.image_mean[c]) / processor.image_std[c];
    }
  }
  const feeds = { pixel_values: new ort.Tensor('float32', rgb, [1, 3, info.height, info.width]) };
  if (session.inputNames.includes('pixel_mask')) {
    // One unpadded image: all pixels are valid. Matches the model's JS export.
    feeds.pixel_mask = new ort.Tensor('int64', new BigInt64Array(64 * 64).fill(1n), [1, 64, 64]);
  }
  const output = await session.run(feeds);
  const classes = output.logits.dims[2];
  const queries = output.logits.dims[1];
  const [, , maskHeight, maskWidth] = output.pred_masks.dims;
  const maskPixels = maskWidth * maskHeight;
  let best = null;

  for (let query = 0; query < queries; query++) {
    const offset = query * classes;
    let maximum = -Infinity;
    let classId = -1;
    for (let c = 0; c < classes; c++) {
      if (output.logits.data[offset + c] > maximum) {
        maximum = output.logits.data[offset + c]; classId = c;
      }
    }
    // COCO thing classes occupy IDs 1..90. Exclude scene/stuff classes.
    if (classId < 1 || classId > 90) continue;
    const label = String(config.id2label[classId] || '').toLowerCase();
    if (!label || label === 'n/a' || label.startsWith('label_')) continue;
    let denominator = 0;
    for (let c = 0; c < classes; c++) denominator += Math.exp(output.logits.data[offset + c] - maximum);
    const confidence = 1 / denominator;
    if (confidence < SCORE_THRESHOLD) continue;

    const binary = new Uint8Array(maskPixels);
    let area = 0, xTotal = 0, yTotal = 0;
    const maskOffset = query * maskPixels;
    const logitThreshold = Math.log(MASK_THRESHOLD / (1 - MASK_THRESHOLD));
    for (let i = 0; i < maskPixels; i++) {
      if (output.pred_masks.data[maskOffset + i] >= logitThreshold) {
        binary[i] = 255; area++; xTotal += i % maskWidth; yTotal += Math.floor(i / maskWidth);
      }
    }
    const fraction = area / maskPixels;
    if (fraction < 0.004 || fraction > 0.985) continue;
    const centerDistance = Math.hypot(xTotal / area / maskWidth - 0.5, yTotal / area / maskHeight - 0.5);
    const priority = PETS.has(label) ? 3 : label === 'person' ? 2 : 1;
    const score = priority * 10 + Math.min(fraction / 0.5, 1) * 0.5 + confidence * 0.3 +
      Math.max(0, 1 - centerDistance / 0.707) * 0.2;
    if (!best || score > best.rank) {
      best = { label, confidence, rank: score, query, mask: binary, maskWidth, maskHeight };
    }
  }

  if (!best) {
    logStage(id, 'subject:unrecognized', 'using existing foreground processing');
    return null;
  }
  const component = keepLargestComponent(best.mask, maskWidth, maskHeight);
  // Resize this instance's probabilities before refining its edge. Converting
  // every query to a full-size mask would waste memory and include scene pixels.
  const selectedOffset = best.query * maskPixels;
  best.mask = Uint8Array.from(component, (value, i) => value ?
    Math.round(255 / (1 + Math.exp(-output.pred_masks.data[selectedOffset + i]))) : 0);
  logStage(id, 'subject:done', `label=${best.label} confidence=${best.confidence.toFixed(4)}`);
  return best;
}

// Filter disconnected specks from this one instance's mask, not from the
// photograph. A box or "largest foreground blob" cannot exclude an attached prop.
export function keepLargestComponent(mask, width, height) {
  const visited = new Uint8Array(mask.length);
  const queue = new Uint32Array(mask.length);
  let largest = new Uint32Array(0);
  for (let start = 0; start < mask.length; start++) {
    if (!mask[start] || visited[start]) continue;
    let head = 0, tail = 1;
    queue[0] = start; visited[start] = 1;
    while (head < tail) {
      const index = queue[head++];
      const x = index % width, y = Math.floor(index / width);
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const nx = x + dx, ny = y + dy;
          if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
          const neighbor = ny * width + nx;
          if (mask[neighbor] && !visited[neighbor]) {
            visited[neighbor] = 1; queue[tail++] = neighbor;
          }
        }
      }
    }
    if (tail > largest.length) largest = queue.slice(0, tail);
  }
  const clean = new Uint8Array(mask.length);
  for (const index of largest) clean[index] = 255;
  return clean;
}

export function growMask(mask, width, height, radius) {
  const horizontal = new Uint8Array(mask.length);
  const expanded = new Uint8Array(mask.length);
  for (let y = 0; y < height; y++) {
    const row = y * width;
    let count = 0;
    for (let x = 0; x <= Math.min(radius, width - 1); x++) if (mask[row + x]) count++;
    for (let x = 0; x < width; x++) {
      horizontal[row + x] = count ? 255 : 0;
      if (x - radius >= 0 && mask[row + x - radius]) count--;
      if (x + radius + 1 < width && mask[row + x + radius + 1]) count++;
    }
  }
  for (let x = 0; x < width; x++) {
    let count = 0;
    for (let y = 0; y <= Math.min(radius, height - 1); y++) if (horizontal[y * width + x]) count++;
    for (let y = 0; y < height; y++) {
      expanded[y * width + x] = count ? 255 : 0;
      if (y - radius >= 0 && horizontal[(y - radius) * width + x]) count--;
      if (y + radius + 1 < height && horizontal[(y + radius + 1) * width + x]) count++;
    }
  }
  return expanded;
}

export async function isolateMainSubject(buffer, subject, id, logStage) {
  if (!subject) return buffer;
  logStage(id, 'subject-mask:start', `label=${subject.label}`);
  const { data, info } = await sharp(buffer).toColourspace('srgb').ensureAlpha().raw().toBuffer({ resolveWithObject: true });
  const resized = await sharp(Buffer.from(subject.mask), {
    raw: { width: subject.maskWidth, height: subject.maskHeight, channels: 1 }
  }).resize(info.width, info.height, { kernel: 'cubic' }).toColourspace('b-w').raw().toBuffer();
  const binary = Uint8Array.from(resized, value => value >= 115 ? 255 : 0);
  // A small margin preserves fine fur/ears; IMG.LY still supplies the detailed edge.
  const radius = Math.max(1, Math.min(4, Math.round(Math.max(info.width, info.height) * 0.002)));
  const expanded = growMask(binary, info.width, info.height, radius);
  const gate = await sharp(Buffer.from(expanded), {
    raw: { width: info.width, height: info.height, channels: 1 }
  }).blur(0.7).toColourspace('b-w').raw().toBuffer();
  let foregroundPixels = 0;
  for (let i = 0; i < info.width * info.height; i++) {
    const alphaIndex = i * info.channels + 3;
    data[alphaIndex] = Math.round(data[alphaIndex] * gate[i] / 255);
    if (data[alphaIndex] < 8) {
      // Clear hidden background colours so WebGL texture filtering cannot
      // pull them into the pillow's edge. Visible subject colours stay intact.
      data[alphaIndex] = 0;
      data[i * info.channels] = 0;
      data[i * info.channels + 1] = 0;
      data[i * info.channels + 2] = 0;
    }
    if (data[alphaIndex] > 28) foregroundPixels++;
  }
  if (foregroundPixels < 32) throw new Error('SUBJECT_NOT_FOUND');
  const result = await sharp(data, { raw: { width: info.width, height: info.height, channels: info.channels } })
    .png({ compressionLevel: 5, adaptiveFiltering: true }).toBuffer();
  logStage(id, 'subject-mask:done', `foregroundPixels=${foregroundPixels}`);
  return result;
}
