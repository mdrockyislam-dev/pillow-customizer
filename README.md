# TAZROX group-safe unwanted-object cleanup v1.0.3

This build is based on the user's working 1.0.2 DETR + IMG.LY backend.

## Goal
Remove unwanted foreground objects while preserving:
- one pet
- multiple pets
- person + pet
- group photos

## What changed
1. DETR now keeps multiple relevant people/pets instead of only one best subject.
2. Duplicate DETR boxes are suppressed.
3. The crop uses the union of all relevant people/pets.
4. Crop padding is reduced from 20% to about 8-10% to prevent furniture/background entering the processing area.
5. After IMG.LY background removal, foreground pixels outside expanded detected-subject boxes are cleared.
6. Tiny disconnected junk is removed, while meaningful detected group members are preserved.
7. The Shopify endpoint and multipart field are unchanged.

## Railway variables
Keep your existing variables:

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

## Deploy
Replace the backend repository contents with this folder, then:

git add .
git commit -m "Add group safe unwanted object cleanup"
git push

Railway should redeploy automatically.

No Shopify JS change is required because:
- endpoint stays /api/process-pillow
- field stays image
- response stays image/png

After deploy, open / and confirm version 1.0.3.
Then upload:
1. the dog image with the unwanted object,
2. a person + dog image,
3. a multi-person/group photo.

Expected server log stages include:
detect
subject-crop
remove-background
subject-mask
junk-cleanup
transparent-crop
