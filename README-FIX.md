# TAZROX Railway stable backend v1.1.0

This build removes DETR from the Railway process to stop 502/OOM restarts on the trial container.
Desktop and mobile both call the same `/api/process-pillow` endpoint and receive the same processed PNG.

Recommended Railway variables for the trial container:

ALLOWED_ORIGINS=*
MAX_UPLOAD_MB=20
MAX_PROCESSING_SIDE=1200
MAX_CONCURRENT=1
BACKGROUND_MODEL=medium
RATE_LIMIT_PER_MINUTE=30

Old variables such as DETECTOR_DTYPE, DETECTION_THRESHOLD, WARM_DETECTOR and MODEL_CACHE_DIR are no longer used and may be deleted.
