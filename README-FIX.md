# TAZROX Railway backend — typed Blob input fix (v1.3.0)

This build fixes the current server error:

`Error: Unsupported format:`

## Root cause

`@imgly/background-removal-node@1.4.5` decodes image input from its MIME type.
The previous build passed a bare Node `Buffer`. In this runtime the decoder saw
no usable image type and rejected it even though the normalized bytes were PNG.

## Fix

The normalized PNG bytes are now wrapped as:

```js
const inputBlob = new Blob(
  [new Uint8Array(buffer)],
  { type: 'image/png' }
);
```

and that typed Blob is passed to `removeBackground()`.

The Sharp 0.32.4 / libvips native compatibility fix from v1.2.0 is retained.

## Railway variables

ALLOWED_ORIGINS=*
MAX_UPLOAD_MB=20
MAX_PROCESSING_SIDE=1200
MAX_CONCURRENT=1
BACKGROUND_MODEL=medium
RATE_LIMIT_PER_MINUTE=30

## Verify

After deployment open `/health` and confirm:

- version: `1.3.0`
- pipeline: `imgly-typed-blob-fix`
- sharpVersion: `0.32.4`
- detector: `disabled`

Then upload one image from Shopify.

Expected logs:

- request:start
- normalize:start
- normalize:done
- alpha-check:start
- alpha-check:done
- background:start ... input=Blob type=image/png
- background:done
- crop:start
- crop:done
- request:success
