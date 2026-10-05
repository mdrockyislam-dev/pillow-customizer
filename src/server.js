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
const MAX_CONCURRENT = Math.max(1, Number(process.env.MAX_CONCURRENT || 1));
const DETECTION_THRESHOLD = Number(process.env.DETECTION_THRESHOLD || 0.55);
const WARM_DETECTOR = String(process.env.WARM_DETECTOR || 'false').toLowerCase() === 'true';
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

function requestId() {
  return crypto.randomBytes(8).toString('hex');
}

async function getDetector() {
  if (!detectorPromise) {
    console.log('PP3D server: loading DETR ResNet-50...');
    detectorPromise = pipeline(
      'object-detection',
      'Xenova/detr-resnet-50'
    ).then(detector => {
      detectorReady = true;
      console.log('PP3D server: DETR ResNet-50 ready.');
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

function chooseMainSubject(detections, width, height) {
  if (!Array.isArray(detections) || !detections.length) return null;

  const preferredLabels = new Set([
    'person', 'dog', 'cat', 'horse', 'bird', 'sheep', 'cow', 'bear'
  ]);

  const imageArea = Math.max(1, width * height);
  const imageCX = width / 2;
  const imageCY = height / 2;
  const maxDistance = Math.max(1, Math.hypot(imageCX, imageCY));

  const candidates = detections.filter(item =>
    preferredLabels.has(String(item?.label || '').toLowerCase())
  );

  if (!candidates.length) return null;

  let best = null;
  let bestScore = -Infinity;

  for (const item of candidates) {
    const box = item?.box;
    if (!box) continue;

    const xmin = Math.max(0, Number(box.xmin) || 0);
    const ymin = Math.max(0, Number(box.ymin) || 0);
    const xmax = Math.min(width, Number(box.xmax) || width);
    const ymax = Math.min(height, Number(box.ymax) || height);
    if (xmax <= xmin || ymax <= ymin) continue;

    const w = xmax - xmin;
    const h = ymax - ymin;
    const area = w * h;
    const boxCX = xmin + w / 2;
    const boxCY = ymin + h / 2;
    const confidence = Number(item.score) || 0;
    const areaScore = Math.min(1, area / (imageArea * 0.45));
    const distance = Math.hypot(boxCX - imageCX, boxCY - imageCY);
    const centerScore = 1 - Math.min(1, distance / maxDistance);

    const score =
      confidence * 0.68 +
      areaScore * 0.20 +
      centerScore * 0.12;

    if (score > bestScore) {
      bestScore = score;
      best = {
        label: String(item.label || '').toLowerCase(),
        confidence,
        box: { xmin, ymin, xmax, ymax }
      };
    }
  }

  return best;
}

async function detectMainSubject(buffer) {
  const detector = await getDetector();
  const { width, height } = await getDimensions(buffer);
  const blob = new Blob([buffer], { type: 'image/png' });
  const rawImage = await RawImage.fromBlob(blob);
  const detections = await detector(rawImage, { threshold: DETECTION_THRESHOLD });
  return chooseMainSubject(detections, width, height);
}

async function cropAroundSubject(buffer, detection) {
  if (!detection?.box) return buffer;

  const { width: W, height: H } = await getDimensions(buffer);
  const { xmin, ymin, xmax, ymax } = detection.box;
  const subjectW = Math.max(1, xmax - xmin);
  const subjectH = Math.max(1, ymax - ymin);

  // These values intentionally match the current desktop Shopify code.
  const padX = subjectW * 0.20;
  const padTop = subjectH * 0.18;
  const padBottom = subjectH * 0.20;

  const x1 = Math.max(0, Math.floor(xmin - padX));
  const y1 = Math.max(0, Math.floor(ymin - padTop));
  const x2 = Math.min(W, Math.ceil(xmax + padX));
  const y2 = Math.min(H, Math.ceil(ymax + padBottom));

  if (x1 <= 1 && y1 <= 1 && x2 >= W - 1 && y2 >= H - 1) {
    return buffer;
  }

  return sharp(buffer)
    .extract({
      left: x1,
      top: y1,
      width: Math.max(1, x2 - x1),
      height: Math.max(1, y2 - y1)
    })
    .png({ compressionLevel: 6 })
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
  const output = await removeBackground(buffer, {
    debug: false,
    model: 'isnet',
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

async function processPillow(inputBuffer) {
  const started = Date.now();
  const normalized = await normalizeImage(inputBuffer);

  let detection = null;
  let processingImage = normalized;

  try {
    detection = await detectMainSubject(normalized);
    if (detection?.confidence >= DETECTION_THRESHOLD) {
      processingImage = await cropAroundSubject(normalized, detection);
    }
  } catch (error) {
    // Same spirit as the existing desktop code: detection failure must not
    // destroy the whole pillow flow. Background removal still gets a chance.
    console.warn('PP3D server: subject detection skipped:', error?.message || error);
    detection = null;
    processingImage = normalized;
  }

  let foreground = processingImage;
  if (!(await hasUsefulTransparency(processingImage))) {
    foreground = await removeBackgroundServer(processingImage);
  }

  const finalPng = await cropTransparentLikeShopify(foreground);

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
    version: '1.0.0'
  });
});

app.get('/health', (_req, res) => {
  res.json({
    ok: true,
    detectorReady,
    maxProcessingSide: MAX_PROCESSING_SIDE,
    maxConcurrent: MAX_CONCURRENT
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

    const result = await runLimited(() => processPillow(req.file.buffer));

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

  if (WARM_DETECTOR) {
    getDetector().catch(error => {
      console.warn('PP3D detector warmup failed:', error?.message || error);
    });
  }
});
