# TAZROX Railway backend — native crash fix

This build fixes the Railway/Linux crash:

`munmap_chunk(): invalid pointer` / `Aborted`

## What changed

- Pins **Sharp 0.32.4**, matching `@imgly/background-removal-node@1.4.5`.
- Uses npm `overrides` so the process does not load Sharp 0.33.x and 0.32.x together.
- Disables Sharp cache and limits Sharp concurrency.
- Passes the PNG Buffer directly to IMG.LY instead of wrapping it in a Blob first.
- Adds stage logs so any remaining crash can be pinpointed.
- Keeps **one identical server pipeline for desktop and mobile**.

## Railway variables

Use:

ALLOWED_ORIGINS=*
MAX_UPLOAD_MB=20
MAX_PROCESSING_SIDE=1200
MAX_CONCURRENT=1
BACKGROUND_MODEL=medium
RATE_LIMIT_PER_MINUTE=30

## After deploy

Open `/health`.

You should see:

- version: `1.2.0`
- pipeline: `imgly-single-sharp-native-fix`
- sharpVersion: `0.32.4`
- detector: `disabled`

Then test one image from Shopify.

If a crash still occurs, Railway logs will now show the last completed stage:
`normalize`, `alpha-check`, `background`, or `crop`.
