# TAZROX main-subject fix — v1.4.0

তোমার v1.3.0 code-এ `detector: disabled` ছিল। IMG.LY background removal পুরো
ছবির foreground বের করছিল, তাই কুকুরের পেছনের সাদা জিনিসটাও থেকে যাচ্ছিল।

এই version-এ DETR panoptic model দিয়ে একটি প্রধান subject এবং তার pixel mask
বেছে নেওয়া হয়। সেই mask দিয়ে IMG.LY-এর cutout থেকে অন্য বস্তু বাদ যায়। শুধু
bounding-box crop বা পুরো foreground-এর সবচেয়ে বড় connected অংশ বেছে নিলে
subject-এর সঙ্গে লেগে থাকা prop বাদ দেওয়া যায় না।

## কী বদলেছে

- কুকুর/বিড়ালসহ pets অগ্রাধিকার পায়; তারপর person ও অন্য পরিচিত object। একই
  ধরনের একাধিক object থাকলে area, confidence ও কেন্দ্রের অবস্থান দিয়ে একটি
  instance বেছে নেওয়া হয়।
- কেবল সেই instance-এর probability mask resize করা হয়; সব mask full resolution
  করলে অপ্রয়োজনীয় memory লাগে।
- IMG.LY-এর বিস্তারিত alpha edge-এর সঙ্গে subject mask মেশানো হয়। ছোট margin
  ও feather দিয়ে edge রাখা হয়; আলাদা mask speck বাদ দেওয়া হয়।
- সম্পূর্ণ transparent pixel-এর hidden background RGB পরিষ্কার করা হয়।
- Detector weights Docker build-এর সময় download ও SHA-256 verify হয়। Customer
  upload-এর সময় model download করতে হয় না।
- Subject detector server boot-এ warm হয় এবং একই session reuse হয়।
- `X-PP3D-Subject`, `X-PP3D-Confidence` ও subject-stage logs যোগ হয়েছে।

Sharp **0.32.4**, IMG.LY **1.4.5**, background model **medium**, typed PNG Blob,
concurrency setting, image normalization ও final transparent crop আগের মতো
রয়েছে। একটি shared ONNX Runtime **1.17.3** ব্যবহার হয়। আলাদা Transformers
package বা দ্বিতীয় native Sharp runtime যোগ করা হয়নি।

Shopify frontend, price, cart, size, notes, 2D/3D rendering বা upload field
বদলানো হয়নি। Endpoint এখনও `POST /api/process-pillow`; multipart field `image`।

## Railway-তে লাগাবে যেভাবে

1. ZIP extract করে সব file existing backend repository-তে add/replace করো।
   `src`, `scripts` ও `models/subject` folder structure রাখতে হবে।
2. `package-lock.json`-সহ push করো এবং Railway-তে নতুন build deploy করো।
   Dockerfile `npm ci`, native dependency check এবং subject-model download চালায়।
3. আগের origin/upload settings রাখো। বর্তমান settings-এর সঙ্গে এগুলো মেলে:

   ```env
   MAX_PROCESSING_SIDE=1200
   MAX_CONCURRENT=1
   BACKGROUND_MODEL=medium
   SUBJECT_INPUT_SIDE=512
   DETECTION_THRESHOLD=0.55
   WARM_SUBJECT=true
   ```

   নতুন তিনটি subject setting code-এর default-ও তাই। PORT Railway নির্ধারণ করবে।
4. Build logs-এ `Subject model downloaded and checksum verified` দেখবে।
   Model weights প্রায় 172 MB; ZIP-এ weights রাখা হয়নি, build script আনে।
5. `/health`-এ দেখবে:

   ```json
   {
     "version": "1.4.0",
     "detector": "detr-panoptic-instance-mask",
     "pipeline": "imgly-main-subject-mask",
     "sharpVersion": "0.32.4"
   }
   ```

   Server warm হওয়া শেষ হলে `detectorReady: true` হবে। Startup log:
   `PP3D main-subject detector ready (DETR panoptic fp32).`
6. একই original dog photo desktop/mobile থেকে upload করো। Logs-এ
   `subject:done label=dog`, `subject-mask:done`, `request:success` এবং Network-এ
   HTTP 200 থাকবে।

`.dockerignore` ও `.gitignore` model weights repository/context থেকে বাদ রাখে;
download script সঠিক weights container image-এ রাখে। `models/subject/config.json`,
`preprocessor_config.json` ও `model.sha256` commit করতে হবে।

Local setup:

```bash
npm ci
npm run check:native
npm run cache:subject
npm start
```

## দেওয়া ছবিতে কী পরীক্ষা করেছি

- `photo_2026-09-23_09-31-29(5).jpg` real inference দিয়ে process করা হয়েছে।
- Dog confidence: প্রায় **0.9986**।
- Desktop ও iPhone request headers দিয়ে একই original JPEG পাঠানো হয়েছে।
- দুই response-এ **HTTP 200**, এবং byte-identical transparent PNG এসেছে।
- Final PNG size: **623 × 1000**।
- Latest local test: প্রথম request প্রায় **7.2 seconds**, পরেরটি **6.1 seconds**।
- IMAGE_REQUIRED ও UNSUPPORTED_IMAGE_TYPE error behaviour পরীক্ষা হয়েছে।
- Mask padding/component cleanup-এর edge cases পরীক্ষা হয়েছে।
- Model download-এর fresh-cache path ও checksum validation সফল হয়েছে।
- Image normalize, alpha check, IMG.LY removal এবং transparent crop helper-এর
  source supplied v1.3.0 file-এর সঙ্গে একই আছে।

`verification/dog-cutout.png` actual API response। `dog-preview.png` একই cutout
pink background-এ দেখায়; preview background API PNG-এর অংশ নয়।
`api-test-results.json`-এ response measurements ও SHA-256 আছে।

এই tests Linux/Node 24 host-এ হয়েছে। Production Dockerfile-এর Node 20 base
রাখা হয়েছে; live Railway ও actual phone browser এখানে deploy/test করা হয়নি।
সময় host-specific; Railway CPU ও workload অনুযায়ী বদলাবে। Logged memory
checkpoints প্রায় 1.2 GB পর্যন্ত উঠেছে; inference-এর মধ্যবর্তী peak বেশি হতে
পারে। নতুন detector-এর জন্য service-এ memory headroom রাখো; 2 GB দিয়ে শুরু
করে Railway memory graph দেখো এবং `MAX_CONCURRENT=1` রাখো।

## সীমা

Automatic selection model-এর পরিচিত classes ও visible subject-এর ওপর নির্ভর
করে। কোনো পরিচিত instance না পাওয়া গেলে আগের foreground processing থাকে,
যাতে অন্য ধরনের ছবি upload করার existing flow বন্ধ না হয়। খুব জটিল বা
অস্পষ্ট ছবিতে automatic mask নিখুঁত হওয়ার নিশ্চয়তা নেই।

Model source: https://huggingface.co/Xenova/detr-resnet-50-panoptic
Base model: https://huggingface.co/facebook/detr-resnet-50-panoptic
