import crypto from 'node:crypto';
import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';
import multer from 'multer';
import pLimit from 'p-limit';
import sharp from 'sharp';
import { removeBackground } from '@imgly/background-removal-node';

const PORT = Number(process.env.PORT || 3000);
const MAX_UPLOAD_MB = Math.max(1, Number(process.env.MAX_UPLOAD_MB || 20));
const MAX_PROCESSING_SIDE = Math.min(
  1400,
  Math.max(768, Number(process.env.MAX_PROCESSING_SIDE || 1200))
);
const MAX_CONCURRENT = Math.max(1, Number(process.env.MAX_CONCURRENT || 1));

const BACKGROUND_MODEL_RAW = String(
  process.env.BACKGROUND_MODEL || 'medium'
).toLowerCase();

const BACKGROUND_MODEL = ['small', 'medium', 'large'].includes(BACKGROUND_MODEL_RAW)
  ? BACKGROUND_MODEL_RAW
  : 'medium';

const ALLOWED_ORIGINS = String(process.env.ALLOWED_ORIGINS || '*')
  .split(',')
  .map((value) => value.trim())
  .filter(Boolean);

// IMPORTANT FOR RAILWAY/LINUX:
// @imgly/background-removal-node@1.4.5 itself depends on Sharp 0.32.x.
// This project pins/overrides Sharp to the SAME version so only one libvips/
// Sharp native runtime is loaded in the process. Loading Sharp 0.33.x alongside
// IMG.LY's Sharp 0.32.x can cause native allocator crashes such as:
//   munmap_chunk(): invalid pointer
//   Aborted
sharp.cache(false);
sharp.concurrency(1);

const app = express();
app.set('trust proxy', 1);

app.use(cors({
  origin(origin, callback) {
    if (
      !origin ||
      ALLOWED_ORIGINS.includes('*') ||
      ALLOWED_ORIGINS.includes(origin)
    ) {
      callback(null, true);
      return;
    }

    callback(new Error('Origin is not allowed by CORS'));
  },
  methods: ['GET', 'POST', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Accept', 'X-Requested-With'],
  exposedHeaders: [
    'X-PP3D-Request-Id',
    'X-PP3D-Processing-Ms',
    'X-PP3D-Pipeline'
  ]
}));

app.options('*', cors());

app.use(
  helmet({
    crossOriginResourcePolicy: false
  })
);

app.use(express.json({ limit: '256kb' }));

app.use(
  rateLimit({
    windowMs: 60_000,
    limit: Number(process.env.RATE_LIMIT_PER_MINUTE || 30),
    standardHeaders: 'draft-7',
    legacyHeaders: false
  })
);

const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: MAX_UPLOAD_MB * 1024 * 1024,
    files: 1
  },
  fileFilter(_req, file, callback) {
    const accepted = new Set([
      'image/jpeg',
      'image/png',
      'image/webp'
    ]);

    if (!accepted.has(file.mimetype)) {
      callback(new Error('UNSUPPORTED_IMAGE_TYPE'));
      return;
    }

    callback(null, true);
  }
});

const runLimited = pLimit(MAX_CONCURRENT);

function createRequestId() {
  return crypto.randomBytes(8).toString('hex');
}

function logStage(id, stage, details = '') {
  const suffix = details ? ` ${details}` : '';
  console.log(`[PP3D ${id}] ${stage}${suffix}`);
}

async function normalizeImage(inputBuffer, id) {
  logStage(id, 'normalize:start', `bytes=${inputBuffer.length}`);

  const output = await sharp(inputBuffer, {
    failOn: 'warning',
    sequentialRead: true,
    limitInputPixels: 80_000_000
  })
    .rotate()
    .resize({
      width: MAX_PROCESSING_SIDE,
      height: MAX_PROCESSING_SIDE,
      fit: 'inside',
      withoutEnlargement: true,
      kernel: sharp.kernel.lanczos3
    })
    .png({
      compressionLevel: 5,
      adaptiveFiltering: true
    })
    .toBuffer();

  logStage(id, 'normalize:done', `bytes=${output.length}`);
  return output;
}

async function hasUsefulTransparency(buffer, id) {
  logStage(id, 'alpha-check:start');

  const { data, info } = await sharp(buffer, {
    sequentialRead: true
  })
    .ensureAlpha()
    .resize({
      width: 180,
      height: 180,
      fit: 'inside',
      withoutEnlargement: true
    })
    .raw()
    .toBuffer({ resolveWithObject: true });

  let transparent = 0;
  const pixels = info.width * info.height;

  for (let i = 3; i < data.length; i += 4) {
    if (data[i] < 240) {
      transparent++;
    }
  }

  const useful =
    pixels > 0 &&
    transparent / pixels > 0.025;

  logStage(id, 'alpha-check:done', `transparent=${useful}`);
  return useful;
}

async function removeBackgroundServer(buffer, id) {
  /*
    IMPORTANT:
    @imgly/background-removal-node@1.4.5 decodes input based on Blob.type.
    Passing a bare Node Buffer can reach imageDecode() with no MIME type and
    throw "Unsupported format:" even when the bytes are a valid PNG.

    normalizeImage() always returns PNG bytes, so wrap those bytes in a typed
    Blob before calling IMG.LY.
  */
  const inputBlob = new Blob(
    [new Uint8Array(buffer)],
    { type: 'image/png' }
  );

  logStage(
    id,
    'background:start',
    `model=${BACKGROUND_MODEL} input=Blob type=${inputBlob.type} bytes=${inputBlob.size}`
  );

  const output = await removeBackground(inputBlob, {
    debug: false,
    model: BACKGROUND_MODEL,
    proxyToWorker: false,
    output: {
      format: 'image/png',
      quality: 1,
      type: 'foreground'
    }
  });

  const outputBuffer = Buffer.from(await output.arrayBuffer());

  logStage(
    id,
    'background:done',
    `type=${output.type || 'image/png'} bytes=${outputBuffer.length}`
  );

  return outputBuffer;
}

async function cropTransparentLikeShopify(buffer, id) {
  logStage(id, 'crop:start');

  const { data, info } = await sharp(buffer, {
    sequentialRead: true
  })
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
      const alpha =
        data[(y * width + x) * channels + 3];

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

  const originalWidth = maxX - minX + 1;
  const originalHeight = maxY - minY + 1;

  const padding = Math.max(
    2,
    Math.round(
      Math.max(originalWidth, originalHeight) * 0.012
    )
  );

  minX = Math.max(0, minX - padding);
  minY = Math.max(0, minY - padding);
  maxX = Math.min(width - 1, maxX + padding);
  maxY = Math.min(height - 1, maxY + padding);

  const output = await sharp(buffer, {
    sequentialRead: true
  })
    .extract({
      left: minX,
      top: minY,
      width: maxX - minX + 1,
      height: maxY - minY + 1
    })
    .png({
      compressionLevel: 5,
      adaptiveFiltering: true
    })
    .toBuffer();

  logStage(
    id,
    'crop:done',
    `size=${maxX - minX + 1}x${maxY - minY + 1}`
  );

  return output;
}

async function processPillow(inputBuffer, id) {
  const startedAt = Date.now();

  const normalized =
    await normalizeImage(inputBuffer, id);

  let foreground = normalized;

  if (!(await hasUsefulTransparency(normalized, id))) {
    foreground =
      await removeBackgroundServer(normalized, id);
  }

  const finalPng =
    await cropTransparentLikeShopify(foreground, id);

  return {
    buffer: finalPng,
    elapsedMs: Date.now() - startedAt
  };
}

app.get('/', (_req, res) => {
  res.json({
    ok: true,
    service: 'TAZROX Pillow Processing API',
    version: '1.3.0',
    pipeline: 'imgly-typed-blob-fix'
  });
});

app.get('/health', (_req, res) => {
  res.json({
    ok: true,
    version: '1.3.0',
    maxProcessingSide: MAX_PROCESSING_SIDE,
    maxConcurrent: MAX_CONCURRENT,
    backgroundModel: BACKGROUND_MODEL,
    detector: 'disabled',
    samePipelineForDesktopAndMobile: true,
    pipeline: 'imgly-typed-blob-fix',
    sharpVersion: sharp.versions?.sharp || 'unknown',
    libvipsVersion: sharp.versions?.vips || 'unknown'
  });
});

app.post(
  '/api/process-pillow',
  upload.single('image'),
  async (req, res, next) => {
    const id = createRequestId();

    res.setHeader('X-PP3D-Request-Id', id);
    res.setHeader(
      'X-PP3D-Pipeline',
      'imgly-typed-blob-fix'
    );

    try {
      if (!req.file?.buffer) {
        res.status(400).json({
          ok: false,
          code: 'IMAGE_REQUIRED',
          message:
            'Upload an image using the multipart field named "image".'
        });
        return;
      }

      logStage(
        id,
        'request:start',
        `mime=${req.file.mimetype} bytes=${req.file.buffer.length}`
      );

      const result = await runLimited(() =>
        processPillow(req.file.buffer, id)
      );

      res.setHeader('Content-Type', 'image/png');
      res.setHeader(
        'Content-Disposition',
        'inline; filename="pillow-cutout.png"'
      );
      res.setHeader('Cache-Control', 'no-store');
      res.setHeader(
        'X-PP3D-Processing-Ms',
        String(result.elapsedMs)
      );

      logStage(
        id,
        'request:success',
        `ms=${result.elapsedMs}`
      );

      res.status(200).send(result.buffer);
    } catch (error) {
      error.requestId = id;
      next(error);
    }
  }
);

app.use((error, _req, res, _next) => {
  console.error(
    'PP3D SERVER ERROR:',
    {
      requestId: error?.requestId || null,
      name: error?.name || null,
      code: error?.code || null,
      message: error?.message || String(error),
      stack: error?.stack || null
    }
  );

  let status = 500;
  let code = 'SERVER_PROCESSING_FAILED';
  let message =
    'The server could not process this photo. Please try again.';

  if (error?.code === 'LIMIT_FILE_SIZE') {
    status = 413;
    code = 'IMAGE_TOO_LARGE';
    message =
      `The image is larger than ${MAX_UPLOAD_MB}MB.`;
  } else if (
    String(error?.message || '')
      .includes('UNSUPPORTED_IMAGE_TYPE')
  ) {
    status = 415;
    code = 'UNSUPPORTED_IMAGE_TYPE';
    message =
      'Please upload a JPG, PNG, or WebP image.';
  } else if (
    String(error?.message || '')
      .includes('SUBJECT_NOT_FOUND')
  ) {
    status = 422;
    code = 'SUBJECT_NOT_FOUND';
    message =
      'We could not find a usable foreground subject in this photo.';
  }

  res.status(status).json({
    ok: false,
    code,
    message,
    requestId: error?.requestId || null
  });
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(
    `TAZROX Pillow Processing API v1.3.0 listening on port ${PORT}`
  );
  console.log(`Allowed origins: ${ALLOWED_ORIGINS.join(', ')}`);
  console.log(`Max upload: ${MAX_UPLOAD_MB}MB`);
  console.log(`Max processing side: ${MAX_PROCESSING_SIDE}px`);
  console.log(`Concurrency: ${MAX_CONCURRENT}`);
  console.log(`Background model: ${BACKGROUND_MODEL}`);
  console.log('Detector: disabled');
  console.log(
    `Sharp: ${sharp.versions?.sharp || 'unknown'} | libvips: ${sharp.versions?.vips || 'unknown'}`
  );
});
