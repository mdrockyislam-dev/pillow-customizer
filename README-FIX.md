TAZROX Railway backend v1.0.1

Fixes:
- @imgly/background-removal-node@1.4.5 accepts model=small|medium|large, not model=isnet.
- Uses BACKGROUND_MODEL=medium (fp16 ISNet mapping) for desktop-like quality with lower memory.
- DETR uses q8 to reduce Railway memory usage while keeping the same DETR ResNet-50 detector for desktop and mobile.
- Passes a typed PNG Blob to IMG.LY.
- proxyToWorker=false to avoid extra worker/process memory overhead on Railway.

After replacing these backend files and pushing to GitHub, add Railway variables:
DETECTOR_DTYPE=q8
BACKGROUND_MODEL=medium

Keep the existing variables already configured.
