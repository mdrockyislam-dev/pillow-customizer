import crypto from 'node:crypto';
import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';
import multer from 'multer';
import pLimit from 'p-limit';
import sharp from 'sharp';
import { pipeline, RawImage, env } from '@huggingface/transformers';
import { removeBackground } from '@imgly/background-removal-node';

const PORT = Number(process.env.PORT || 3000);
const MAX_UPLOAD_MB = Math.max(1, Number(process.env.MAX_UPLOAD_MB || 20));
const MAX_PROCESSING_SIDE = Math.max(512, Number(process.env.MAX_PROCESSING_SIDE || 1400));
const DETECTOR_SIDE = Math.max(384, Math.min(960, Number(process.env.DETECTOR_SIDE || 640)));
const MAX_CONCURRENT = Math.max(1, Number(process.env.MAX_CONCURRENT || 1));
const DETECTION_THRESHOLD = Number(process.env.DETECTION_THRESHOLD || 0.55);
const WARM_DETECTOR = String(process.env.WARM_DETECTOR || 'false').toLowerCase() === 'true';
const DETECTOR_DTYPE = String(process.env.DETECTOR_DTYPE || 'q8');
const BACKGROUND_MODEL = String(process.env.BACKGROUND_MODEL || 'medium');
const MODEL_CACHE_DIR = process.env.MODEL_CACHE_DIR || '/tmp/tazrox-model-cache';
const ALLOWED_ORIGINS = String(process.env.ALLOWED_ORIGINS || '*')
  .split(',')
  .map(v => v.trim())
  .filter(Boolean);

env.cacheDir = MODEL_CACHE_DIR;
env.allowLocalModels = true;

const app = express();
app.set('trust proxy', 1);
app.use(helmet({ crossOriginResourcePolicy: false }));
app.use(express.json({ limit: '256kb' }));

app.use(cors({
  origin(origin, callback) {
    if (!origin || ALLOWED_ORIGINS.includes('*') || ALLOWED_ORIGINS.includes(origin)) {
      callback(null, true);
      return;
    }
    callback(new Error('Origin is not allowed by CORS'));
  },
  methods: ['GET', 'POST', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Accept', 'X-Requested-With'],
  exposedHeaders: ['X-PP3D-Request-Id', 'X-PP3D-Subject', 'X-PP3D-Confidence']
}));

app.use(rateLimit({
  windowMs: 60_000,
  limit: Number(process.env.RATE_LIMIT_PER_MINUTE || 30),
  standardHeaders: 'draft-7',
  legacyHeaders: false
}));

const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: MAX_UPLOAD_MB * 1024 * 1024,
    files: 1
  },
  fileFilter(_req, file, cb) {
    const ok = new Set(['image/jpeg', 'image/png', 'image/webp']).has(file.mimetype);
    if (!ok) {
      cb(new Error('UNSUPPORTED_IMAGE_TYPE'));
      return;
    }
    cb(null, true);
  }
});

const runLimited = pLimit(MAX_CONCURRENT);
let detectorPromise = null;
let detectorReady = false;
let lastSuccessfulProcessingAt = null;

async function processingStep(id, stage, operation) {
  const started = Date.now();
  const log = (state, extra = {}) => console.log(JSON.stringify({
    service: 'PP3D', requestId: id, stage, state,
    rssMb: Math.round(process.memoryUsage().rss / 1024 / 1024),
    ...extra
  }));

  log('start');
  try {
    const result = await operation();
    log('done', { elapsedMs: Date.now() - started });
    return result;
  } catch (error) {
    log('failed', {
      elapsedMs: Date.now() - started,
      message: String(error?.message || error)
    });
    throw error;
  }
}

function requestId() {
  return crypto.randomBytes(8).toString('hex');
}

async function getDetector() {
  if (!detectorPromise) {
    console.log(`PP3D server: loading DETR ResNet-50 (${DETECTOR_DTYPE})...`);
    detectorPromise = pipeline(
      'object-detection',
      'Xenova/detr-resnet-50',
      { dtype: DETECTOR_DTYPE }
    ).then(detector => {
      detectorReady = true;
      console.log(`PP3D server: DETR ResNet-50 ready (${DETECTOR_DTYPE}).`);
      return detector;
    }).catch(error => {
      detectorPromise = null;
      detectorReady = false;
      throw error;
    });
  }
  return detectorPromise;
}

async function normalizeImage(inputBuffer) {
  return sharp(inputBuffer, { failOn: 'warning' })
    .rotate()
    .resize({
      width: MAX_PROCESSING_SIDE,
      height: MAX_PROCESSING_SIDE,
      fit: 'inside',
      withoutEnlargement: true,
      kernel: sharp.kernel.lanczos3
    })
    .png({ compressionLevel: 6, adaptiveFiltering: true })
    .toBuffer();
}

async function getDimensions(buffer) {
  const meta = await sharp(buffer).metadata();
  return {
    width: Math.max(1, Number(meta.width || 1)),
    height: Math.max(1, Number(meta.height || 1))
  };
}


function boxIou(a, b) {
  const left = Math.max(a.xmin, b.xmin);
  const top = Math.max(a.ymin, b.ymin);
  const right = Math.min(a.xmax, b.xmax);
  const bottom = Math.min(a.ymax, b.ymax);

  const iw = Math.max(0, right - left);
  const ih = Math.max(0, bottom - top);
  const intersection = iw * ih;

  const areaA = Math.max(1, (a.xmax - a.xmin) * (a.ymax - a.ymin));
  const areaB = Math.max(1, (b.xmax - b.xmin) * (b.ymax - b.ymin));

  return intersection / Math.max(1, areaA + areaB - intersection);
}

function chooseRelevantSubjects(detections, width, height) {
  if (!Array.isArray(detections) || !detections.length) return null;

  const preferredLabels = new Set([
    'person', 'dog', 'cat', 'horse', 'bird', 'sheep', 'cow', 'bear'
  ]);

  const candidates = detections
    .filter(item => preferredLabels.has(String(item?.label || '').toLowerCase()))
    .map(item => {
      const box = item?.box;
      if (!box) return null;

      const xmin = Math.max(0, Number(box.xmin) || 0);
      const ymin = Math.max(0, Number(box.ymin) || 0);
      const xmax = Math.min(width, Number(box.xmax) || width);
      const ymax = Math.min(height, Number(box.ymax) || height);

      if (xmax <= xmin || ymax <= ymin) return null;

      return {
        label: String(item.label || '').toLowerCase(),
        confidence: Number(item.score) || 0,
        box: { xmin, ymin, xmax, ymax }
      };
    })
    .filter(Boolean)
    .sort((a, b) => b.confidence - a.confidence);

  if (!candidates.length) return null;

  // Keep real multiple subjects (person + pet, two pets, group photo), but
  // suppress duplicate DETR boxes for the same physical subject.
  const selected = [];
  const MAX_SUBJECTS = 10;

  for (const candidate of candidates) {
    const duplicate = selected.some(existing =>
      existing.label === candidate.label &&
      boxIou(existing.box, candidate.box) > 0.68
    );

    if (!duplicate) selected.push(candidate);
    if (selected.length >= MAX_SUBJECTS) break;
  }

  if (!selected.length) return null;

  const union = selected.reduce((acc, item) => ({
    xmin: Math.min(acc.xmin, item.box.xmin),
    ymin: Math.min(acc.ymin, item.box.ymin),
    xmax: Math.max(acc.xmax, item.box.xmax),
    ymax: Math.max(acc.ymax, item.box.ymax)
  }), {
    xmin: width,
    ymin: height,
    xmax: 0,
    ymax: 0
  });

  return {
    label: selected.length === 1 ? selected[0].label : 'group',
    confidence: Math.max(...selected.map(item => item.confidence)),
    box: union,
    subjects: selected
  };
}

async function detectRelevantSubjects(buffer) {
  const detector = await getDetector();
  const { width: originalWidth, height: originalHeight } = await getDimensions(buffer);

  // DETR is the slowest stage on Railway CPU. Run detection on a smaller copy
  // only, then map all subject boxes back to the normalized full-size image.
  // This does NOT change the image used for background removal or final output.
  const detectorBuffer = await sharp(buffer)
    .resize({
      width: DETECTOR_SIDE,
      height: DETECTOR_SIDE,
      fit: 'inside',
      withoutEnlargement: true,
      kernel: sharp.kernel.lanczos3
    })
    .png({ compressionLevel: 3 })
    .toBuffer();

  const { width: detectorWidth, height: detectorHeight } = await getDimensions(detectorBuffer);
  const blob = new Blob([detectorBuffer], { type: 'image/png' });
  const rawImage = await RawImage.fromBlob(blob);

  try {
    const detections = await detector(rawImage, {
      threshold: DETECTION_THRESHOLD
    });

    const detected = chooseRelevantSubjects(
      detections,
      detectorWidth,
      detectorHeight
    );

    if (!detected) return null;

    const scaleX = originalWidth / Math.max(1, detectorWidth);
    const scaleY = originalHeight / Math.max(1, detectorHeight);

    const mapBox = box => ({
      xmin: Math.max(0, box.xmin * scaleX),
      ymin: Math.max(0, box.ymin * scaleY),
      xmax: Math.min(originalWidth, box.xmax * scaleX),
      ymax: Math.min(originalHeight, box.ymax * scaleY)
    });

    const mappedSubjects = (detected.subjects || []).map(subject => ({
      ...subject,
      box: mapBox(subject.box)
    }));

    return {
      ...detected,
      box: mapBox(detected.box),
      subjects: mappedSubjects
    };
  } finally {
    // Critical for Railway memory: free the DETR/ONNX session before IMG.LY
    // loads its own background-removal model.
    try {
      if (typeof detector?.dispose === 'function') {
        await detector.dispose();
      }
    } catch (error) {
      console.warn('PP3D detector dispose warning:', error?.message || error);
    }

    detectorPromise = null;
    detectorReady = false;

    if (typeof global.gc === 'function') {
      try { global.gc(); } catch {}
    }
  }
}

async function cropAroundSubjects(buffer, detection) {
  if (!detection?.box) {
    return {
      buffer,
      subjectBoxes: []
    };
  }

  const { width: W, height: H } = await getDimensions(buffer);
  const { xmin, ymin, xmax, ymax } = detection.box;

  const groupW = Math.max(1, xmax - xmin);
  const groupH = Math.max(1, ymax - ymin);

  // Tighter than the old 20% crop. This is intentionally conservative:
  // enough room for ears/tails/hair, but far less room for furniture/background.
  const padX = Math.max(10, groupW * 0.08);
  const padTop = Math.max(10, groupH * 0.08);
  const padBottom = Math.max(12, groupH * 0.10);

  const x1 = Math.max(0, Math.floor(xmin - padX));
  const y1 = Math.max(0, Math.floor(ymin - padTop));
  const x2 = Math.min(W, Math.ceil(xmax + padX));
  const y2 = Math.min(H, Math.ceil(ymax + padBottom));

  const cropWidth = Math.max(1, x2 - x1);
  const cropHeight = Math.max(1, y2 - y1);

  const cropped = (
    x1 <= 1 &&
    y1 <= 1 &&
    x2 >= W - 1 &&
    y2 >= H - 1
  )
    ? buffer
    : await sharp(buffer)
        .extract({
          left: x1,
          top: y1,
          width: cropWidth,
          height: cropHeight
        })
        .png({ compressionLevel: 6 })
        .toBuffer();

  const subjects = Array.isArray(detection.subjects) && detection.subjects.length
    ? detection.subjects
    : [detection];

  const subjectBoxes = subjects
    .map(subject => {
      const box = subject?.box;
      if (!box) return null;

      return {
        label: subject.label || 'subject',
        confidence: Number(subject.confidence) || 0,
        xmin: Math.max(0, box.xmin - x1),
        ymin: Math.max(0, box.ymin - y1),
        xmax: Math.min(cropWidth, box.xmax - x1),
        ymax: Math.min(cropHeight, box.ymax - y1)
      };
    })
    .filter(box => box && box.xmax > box.xmin && box.ymax > box.ymin);

  return {
    buffer: cropped,
    subjectBoxes
  };
}

function expandSubjectBox(box, width, height, ratio = 0.14) {
  const w = Math.max(1, box.xmax - box.xmin);
  const h = Math.max(1, box.ymax - box.ymin);

  const px = Math.max(8, w * ratio);
  const py = Math.max(8, h * ratio);

  return {
    xmin: Math.max(0, Math.floor(box.xmin - px)),
    ymin: Math.max(0, Math.floor(box.ymin - py)),
    xmax: Math.min(width, Math.ceil(box.xmax + px)),
    ymax: Math.min(height, Math.ceil(box.ymax + py))
  };
}

async function removePixelsOutsideDetectedSubjects(buffer, subjectBoxes) {
  if (!Array.isArray(subjectBoxes) || !subjectBoxes.length) return buffer;

  const { data, info } = await sharp(buffer)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const { width, height, channels } = info;

  const allowedBoxes = subjectBoxes.map(box =>
    expandSubjectBox(box, width, height, 0.14)
  );

  const output = Buffer.from(data);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let allowed = false;

      for (const box of allowedBoxes) {
        if (
          x >= box.xmin &&
          x < box.xmax &&
          y >= box.ymin &&
          y < box.ymax
        ) {
          allowed = true;
          break;
        }
      }

      if (!allowed) {
        output[(y * width + x) * channels + 3] = 0;
      }
    }
  }

  return sharp(output, {
    raw: { width, height, channels }
  })
    .png({ compressionLevel: 6, adaptiveFiltering: true })
    .toBuffer();
}

function rectIntersectionArea(a, b) {
  const left = Math.max(a.xmin, b.xmin);
  const top = Math.max(a.ymin, b.ymin);
  const right = Math.min(a.xmax, b.xmax);
  const bottom = Math.min(a.ymax, b.ymax);

  return Math.max(0, right - left) * Math.max(0, bottom - top);
}

async function removeDisconnectedJunk(buffer, subjectBoxes) {
  const { data, info } = await sharp(buffer)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const { width, height, channels } = info;
  const alphaThreshold = 28;
  const pixelCount = width * height;

  const visited = new Uint8Array(pixelCount);
  const componentId = new Int32Array(pixelCount);
  const components = [];
  let nextId = 0;

  const isForeground = index =>
    data[index * channels + 3] > alphaThreshold;

  const stack = new Int32Array(pixelCount);

  for (let start = 0; start < pixelCount; start++) {
    if (visited[start] || !isForeground(start)) continue;

    nextId++;
    let stackSize = 0;
    stack[stackSize++] = start;
    visited[start] = 1;

    let area = 0;
    let minX = width;
    let minY = height;
    let maxX = -1;
    let maxY = -1;

    while (stackSize > 0) {
      const current = stack[--stackSize];
      componentId[current] = nextId;
      area++;

      const x = current % width;
      const y = Math.floor(current / width);

      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;

      const x0 = Math.max(0, x - 1);
      const x1 = Math.min(width - 1, x + 1);
      const y0 = Math.max(0, y - 1);
      const y1 = Math.min(height - 1, y + 1);

      for (let ny = y0; ny <= y1; ny++) {
        for (let nx = x0; nx <= x1; nx++) {
          if (nx === x && ny === y) continue;

          const ni = ny * width + nx;
          if (!visited[ni] && isForeground(ni)) {
            visited[ni] = 1;
            stack[stackSize++] = ni;
          }
        }
      }
    }

    components.push({
      id: nextId,
      area,
      box: {
        xmin: minX,
        ymin: minY,
        xmax: maxX + 1,
        ymax: maxY + 1
      }
    });
  }

  if (!components.length) return buffer;

  const largestArea = Math.max(...components.map(c => c.area));
  const totalForeground = components.reduce((sum, c) => sum + c.area, 0);
  const exactSubjectBoxes = Array.isArray(subjectBoxes) ? subjectBoxes : [];

  const keepIds = new Set();

  for (const component of components) {
    const boxArea = Math.max(
      1,
      (component.box.xmax - component.box.xmin) *
      (component.box.ymax - component.box.ymin)
    );

    let overlapsSubject = exactSubjectBoxes.length === 0;

    for (const subjectBox of exactSubjectBoxes) {
      const intersection = rectIntersectionArea(component.box, subjectBox);

      if (
        intersection / boxArea >= 0.10 ||
        intersection / Math.max(
          1,
          (subjectBox.xmax - subjectBox.xmin) *
          (subjectBox.ymax - subjectBox.ymin)
        ) >= 0.02
      ) {
        overlapsSubject = true;
        break;
      }
    }

    // Keep all meaningful detected subjects, including group members and pets.
    // Tiny disconnected specks are removed.
    const meaningfulSize =
      component.area >= Math.max(24, largestArea * 0.004) ||
      component.area >= totalForeground * 0.006;

    if (overlapsSubject && meaningfulSize) {
      keepIds.add(component.id);
    }
  }

  // Safety fallback: never return an empty image.
  if (!keepIds.size) {
    const biggest = components.reduce((a, b) => a.area >= b.area ? a : b);
    keepIds.add(biggest.id);
  }

  const output = Buffer.from(data);

  for (let i = 0; i < pixelCount; i++) {
    if (!keepIds.has(componentId[i])) {
      output[i * channels + 3] = 0;
    }
  }

  return sharp(output, {
    raw: { width, height, channels }
  })
    .png({ compressionLevel: 6, adaptiveFiltering: true })
    .toBuffer();
}

async function hasUsefulTransparency(buffer) {
  const { data, info } = await sharp(buffer)
    .ensureAlpha()
    .resize({
      width: 200,
      height: 200,
      fit: 'inside',
      withoutEnlargement: true
    })
    .raw()
    .toBuffer({ resolveWithObject: true });

  let transparent = 0;
  const pixels = info.width * info.height;
  for (let i = 3; i < data.length; i += 4) {
    if (data[i] < 240) transparent++;
  }
  return pixels > 0 && transparent / pixels > 0.025;
}

async function removeBackgroundServer(buffer) {
  // @imgly/background-removal-node@1.4.5 validates model as
  // 'small' | 'medium' | 'large'. 'medium' maps to the fp16 ISNet model,
  // which is the closest server-side match to the desktop GPU path while
  // using much less memory than the full 'large' model.
  const input = new Blob([buffer], { type: 'image/png' });

  const output = await removeBackground(input, {
    debug: false,
    model: BACKGROUND_MODEL,
    proxyToWorker: false,
    output: {
      format: 'image/png',
      quality: 1,
      type: 'foreground'
    }
  });

  return Buffer.from(await output.arrayBuffer());
}

async function cropTransparentLikeShopify(buffer) {
  const { data, info } = await sharp(buffer)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const { width, height, channels } = info;
  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const alpha = data[(y * width + x) * channels + 3];
      if (alpha > 28) {
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
      }
    }
  }

  if (maxX < minX || maxY < minY) {
    throw new Error('SUBJECT_NOT_FOUND');
  }

  const w0 = maxX - minX + 1;
  const h0 = maxY - minY + 1;
  const pad = Math.max(2, Math.round(Math.max(w0, h0) * 0.012));

  minX = Math.max(0, minX - pad);
  minY = Math.max(0, minY - pad);
  maxX = Math.min(width - 1, maxX + pad);
  maxY = Math.min(height - 1, maxY + pad);

  return sharp(buffer)
    .extract({
      left: minX,
      top: minY,
      width: maxX - minX + 1,
      height: maxY - minY + 1
    })
    .png({ compressionLevel: 6, adaptiveFiltering: true })
    .toBuffer();
}

async function processPillow(inputBuffer, id) {
  const started = Date.now();
  const normalized = await processingStep(
    id,
    'normalize',
    () => normalizeImage(inputBuffer)
  );

  let detection = null;
  let processingImage = normalized;
  let subjectBoxes = [];

  try {
    detection = await processingStep(
      id,
      'detect',
      () => detectRelevantSubjects(normalized)
    );

    if (detection?.confidence >= DETECTION_THRESHOLD) {
      const cropped = await processingStep(
        id,
        'subject-crop',
        () => cropAroundSubjects(normalized, detection)
      );

      processingImage = cropped.buffer;
      subjectBoxes = cropped.subjectBoxes;
    }
  } catch (error) {
    // Detection failure should not destroy the pillow flow.
    console.warn(
      'PP3D server: subject detection skipped:',
      error?.message || error
    );

    detection = null;
    processingImage = normalized;
    subjectBoxes = [];
  }

  let foreground = processingImage;

  if (!(
    await processingStep(
      id,
      'alpha-check',
      () => hasUsefulTransparency(processingImage)
    )
  )) {
    foreground = await processingStep(
      id,
      'remove-background',
      () => removeBackgroundServer(processingImage)
    );
  }

  // Key quality step:
  // Keep ALL detected people/pets, but erase foreground pixels that belong to
  // unrelated furniture/background outside those subject boxes.
  if (subjectBoxes.length) {
    foreground = await processingStep(
      id,
      'subject-mask',
      () => removePixelsOutsideDetectedSubjects(foreground, subjectBoxes)
    );

    foreground = await processingStep(
      id,
      'junk-cleanup',
      () => removeDisconnectedJunk(foreground, subjectBoxes)
    );
  }

  const finalPng = await processingStep(
    id,
    'transparent-crop',
    () => cropTransparentLikeShopify(foreground)
  );

  lastSuccessfulProcessingAt = new Date().toISOString();

  return {
    buffer: finalPng,
    detection,
    elapsedMs: Date.now() - started
  };
}

app.get('/', (_req, res) => {
  res.json({
    ok: true,
    service: 'TAZROX Pillow Processing API',
    version: '1.0.4'
  });
});

app.get('/health', (_req, res) => {
  res.json({
    ok: true,
    detectorReady,
    lastSuccessfulProcessingAt,
    maxProcessingSide: MAX_PROCESSING_SIDE,
    maxConcurrent: MAX_CONCURRENT,
    detectorDtype: DETECTOR_DTYPE,
    backgroundModel: BACKGROUND_MODEL,
    detectorSide: DETECTOR_SIDE
  });
});

app.post('/api/process-pillow', upload.single('image'), async (req, res, next) => {
  const id = requestId();
  res.setHeader('X-PP3D-Request-Id', id);

  try {
    if (!req.file?.buffer) {
      res.status(400).json({
        ok: false,
        code: 'IMAGE_REQUIRED',
        message: 'Upload an image using the multipart field named "image".'
      });
      return;
    }

    const result = await runLimited(() => processPillow(req.file.buffer, id));

    if (result.detection?.label) {
      res.setHeader('X-PP3D-Subject', result.detection.label);
      res.setHeader('X-PP3D-Confidence', String(result.detection.confidence || 0));
    }

    res.setHeader('Content-Type', 'image/png');
    res.setHeader('Content-Disposition', 'inline; filename="pillow-cutout.png"');
    res.setHeader('Cache-Control', 'no-store');
    res.setHeader('X-PP3D-Processing-Ms', String(result.elapsedMs));
    res.status(200).send(result.buffer);
  } catch (error) {
    error.requestId = id;
    next(error);
  }
});

app.use((error, _req, res, _next) => {
  console.error('PP3D SERVER ERROR:', error);

  let status = 500;
  let code = 'SERVER_PROCESSING_FAILED';
  let message = 'The server could not process this photo. Please try again.';

  if (error?.code === 'LIMIT_FILE_SIZE') {
    status = 413;
    code = 'IMAGE_TOO_LARGE';
    message = `The image is larger than ${MAX_UPLOAD_MB}MB.`;
  } else if (String(error?.message || '').includes('UNSUPPORTED_IMAGE_TYPE')) {
    status = 415;
    code = 'UNSUPPORTED_IMAGE_TYPE';
    message = 'Please upload a JPG, PNG, or WebP image.';
  } else if (String(error?.message || '').includes('SUBJECT_NOT_FOUND')) {
    status = 422;
    code = 'SUBJECT_NOT_FOUND';
    message = 'We could not clearly detect the subject in this photo.';
  } else if (String(error?.message || '').includes('CORS')) {
    status = 403;
    code = 'ORIGIN_NOT_ALLOWED';
    message = 'This Shopify domain is not allowed to use the processing API.';
  }

  res.status(status).json({
    ok: false,
    code,
    message,
    requestId: error?.requestId || null
  });
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`TAZROX Pillow Processing API listening on port ${PORT}`);
  console.log(`Allowed origins: ${ALLOWED_ORIGINS.join(', ')}`);
  console.log(`Max upload: ${MAX_UPLOAD_MB}MB`);
  console.log(`Max processing side: ${MAX_PROCESSING_SIDE}px`);
  console.log(`Concurrency: ${MAX_CONCURRENT}`);
  console.log(`Detector dtype: ${DETECTOR_DTYPE}`);
  console.log(`Detector side: ${DETECTOR_SIDE}px`);
  console.log(`Background model: ${BACKGROUND_MODEL}`);

  if (WARM_DETECTOR) {
    getDetector().catch(error => {
      console.warn('PP3D detector warmup failed:', error?.message || error);
    });
  }
});
