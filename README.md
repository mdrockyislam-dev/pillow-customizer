# TAZROX Fast + Memory Safe Backend v1.0.4

## Why the previous version was slow / returned 502

Railway logs showed:
- DETR detection took about 47.6 seconds
- RSS reached about 590-617 MB
- when IMG.LY background removal started, the process was `Killed`
- Railway therefore returned HTTP 502

## Fixes in v1.0.4

1. DETR still uses ResNet-50 q8, but inference runs on a 640px temporary image.
2. Detection boxes are mapped back to the full normalized image.
3. Full 1400px image is still used for crop/background removal/final PNG.
4. The DETR pipeline is explicitly disposed before IMG.LY starts.
5. Node runs with --expose-gc so released model memory can be reclaimed.
6. Existing group-safe multi-subject cleanup remains enabled.

## Railway variables

Keep your current variables and add:

DETECTOR_SIDE=640

Recommended:
ALLOWED_ORIGINS=*
MAX_UPLOAD_MB=20
MAX_PROCESSING_SIDE=1400
MAX_CONCURRENT=1
DETECTION_THRESHOLD=0.55
WARM_DETECTOR=false
RATE_LIMIT_PER_MINUTE=30
MODEL_CACHE_DIR=/tmp/tazrox-model-cache
DETECTOR_DTYPE=q8
BACKGROUND_MODEL=medium
DETECTOR_SIDE=640

## Deploy

Replace the backend repository files with this folder, then:

git add .
git commit -m "Speed up detector and free memory before background removal"
git push

No Shopify JS change is required.

After deploy, /health should include:
- version 1.0.4
- detectorSide 640

Expected logs:
normalize
detect
subject-crop
alpha-check
remove-background
subject-mask
junk-cleanup
transparent-crop
