# Lab Assignment: Landmark Recognition Using SIFT

**Name:** Muhammad Umer Rafiq (FA23-BAI-013)
**Subject:** Introduction to Computer Vision
**Submitted to:** Dr. Jamal Hussain Shah
**Date:** 15-Oct-2026

\---

## 1\. What I did

In this lab I made a landmark recognition program in Python with OpenCV. I keep some reference images of each landmark, take SIFT features from them, and when I give a new image the program compares its features with the stored ones and tells which landmark it is. If it is not sure, it says the image is rejected / unknown.

My steps were:

1. Make the database (6 landmarks, 8 reference images each, one folder per landmark).
2. Extract SIFT keypoints and descriptors from the reference images and save them with their labels.
3. Take a query image, convert to gray, get its SIFT keypoints and descriptors.
4. Match with `cv2.BFMatcher` (L2), `knnMatch` with k=2, then Lowe's ratio test, then count the good matches for each landmark.
5. Use RANSAC homography to find where the landmark is in the query and draw a polygon.
6. Test on 6 kinds of query images and write down the results.

I ran everything in Google Colab (T4 runtime). OpenCV version was 4.14.0. SIFT and BFMatcher in OpenCV run on the CPU, so the GPU was not really used.

## 2\. Dataset

Dataset from Kaggle: `balabaskar/wonders-of-the-world-image-classification`. After download it had 12 classes:

|Class folder|Images|
|-|-|
|burj\_khalifa|390|
|chichen\_itza|340|
|christ\_the\_reedemer|323|
|eiffel\_tower|391|
|great\_wall\_of\_china|392|
|machu\_pichu|393|
|pyramids\_of\_giza|372|
|roman\_colosseum|394|
|statue\_of\_liberty|238|
|stonehenge|204|
|taj\_mahal|158|
|venezuela\_angel\_falls|251|

My code picked these 6 landmarks for the database: **Taj Mahal, Great Wall of China, Statue of Liberty, Colosseum, Pyramids of Giza, Christ the Redeemer**. **Stonehenge** was kept out of the database on purpose, it is only used as the "landmark not in database" test.

I used 8 reference images per landmark (48 in total) and 4 separate query images per landmark that are never in the database. Images were resized so the longest side is at most 800 px.

Database folder after running:

|Landmark|Reference images|
|-|-|
|Christ the Redeemer|8|
|Colosseum|8|
|Great Wall of China|8|
|Pyramids of Giza|8|
|Statue of Liberty|8|
|Taj Mahal|8|

!\[database preview](outputs/task1\_database\_preview.png)

## 3\. Settings used

These are the values in my config cell:

```python
SEED          = 42
MAX\_DIM       = 800     # resize longest image side to this (big speedup, little accuracy loss)
N\_FEATURES    = 2000    # max SIFT keypoints per image
N\_REF         = 8       # reference images per landmark (database)
N\_QUERY       = 4       # held-out query images per landmark (never in the database)
N\_LANDMARKS   = 6       # number of landmarks in the database (>= 5 required)

RATIO         = 0.75    # Lowe's ratio threshold
RANSAC\_THRESH = 5.0     # reprojection threshold (pixels)
MIN\_GOOD      = 20      # min reliable matches (after ratio test) to accept
MIN\_INLIERS   = 12      # min RANSAC inliers to accept
MIN\_MARGIN    = 1.3     # best landmark inliers must be >= MIN\_MARGIN x runner-up inliers
MIN\_HOMOG\_PTS = 8       # min matches needed to even attempt a homography
```

## 4\. Code

### Task 2: SIFT feature extraction

```python
def make\_sift(): return cv2.SIFT\_create(nfeatures=N\_FEATURES)

def extract(img\_bgr):
    """grayscale -> SIFT keypoints + 128-d descriptors"""
    gray = cv2.cvtColor(img\_bgr, cv2.COLOR\_BGR2GRAY)
    kp, des = make\_sift().detectAndCompute(gray, None)
    return gray, kp, des

kp\_to\_t = lambda k: (k.pt, k.size, k.angle, k.response, k.octave, k.class\_id)
t\_to\_kp = lambda t: cv2.KeyPoint(x=t\[0]\[0], y=t\[0]\[1], size=t\[1], angle=t\[2], response=t\[3], octave=t\[4], class\_id=t\[5])

def process\_ref(args):
    label, fp = args
    img = cv2.imread(fp); \_, kp, des = extract(img)
    return dict(label=label, file=fp, shape=img.shape\[:2], kp=\[kp\_to\_t(k) for k in kp], des=des)
```

Extracting features of all 48 reference images took about 6.2 seconds (I used threads). Average keypoints per image:

|Landmark|Images|Avg keypoints|
|-|-|-|
|Christ the Redeemer|8|938|
|Colosseum|8|1132|
|Great Wall of China|8|586|
|Pyramids of Giza|8|533|
|Statue of Liberty|8|434|
|Taj Mahal|8|1081|

!\[reference keypoints](outputs/task2\_reference\_keypoints.png)

### Task 4: Matching (BFMatcher + kNN k=2 + Lowe ratio test)

```python
bf = cv2.BFMatcher(cv2.NORM\_L2)          # Brute-force, Euclidean distance

def ratio\_match(des\_q, des\_r):
    if des\_q is None or des\_r is None or len(des\_q) < 2 or len(des\_r) < 2: return \[]
    out = \[]
    for pair in bf.knnMatch(des\_q, des\_r, k=2):          # k = 2
        if len(pair) == 2 and pair\[0].distance < RATIO \* pair\[1].distance:   # Lowe's ratio test
            out.append(pair\[0])
    return out
```

### Task 5: Localization (RANSAC homography)

```python
def polygon\_valid(poly, q\_shape):
    if poly is None or not np.isfinite(poly).all(): return False
    if not cv2.isContourConvex(poly.reshape(-1, 1, 2).astype(np.float32)): return False
    area = cv2.contourArea(poly.astype(np.float32)); q\_area = q\_shape\[0] \* q\_shape\[1]
    return 0.01 \* q\_area < area < 3.0 \* q\_area

def localize(kp\_q, kp\_r, good, ref\_shape, q\_shape):
    """RANSAC homography ref->query. Returns (poly, inlier\_mask, n\_inliers, ok)"""
    if len(good) < MIN\_HOMOG\_PTS: return None, None, 0, False
    src = np.float32(\[kp\_r\[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)   # reference pts
    dst = np.float32(\[kp\_q\[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)   # query pts
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC\_THRESH)
    if H is None: return None, None, 0, False
    inl = int(mask.sum()); h, w = ref\_shape
    corners = np.float32(\[\[0, 0], \[w, 0], \[w, h], \[0, h]]).reshape(-1, 1, 2)
    poly = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    return poly, mask.ravel().astype(bool), inl, polygon\_valid(poly, q\_shape)
```

### Decision part (not only the biggest match count)

After matching every reference image I take the best reference image of each landmark and rank the landmarks by RANSAC inliers (then by matches). A landmark is accepted only if all the conditions below are true, otherwise the image is rejected:

```python
    # aggregate per landmark: best reference image
    best = {}
    for r in res:
        key = (r\["inl"], len(r\["good"]))
        if r\["label"] not in best or key > (best\[r\["label"]]\["inl"], len(best\[r\["label"]]\["good"])): best\[r\["label"]] = r
    ranking = sorted(best.values(), key=lambda r: (r\["inl"], len(r\["good"])), reverse=True)
    top = ranking\[0]; second\_inl = ranking\[1]\["inl"] if len(ranking) > 1 else 0

    n\_good, n\_inl = len(top\["good"]), top\["inl"]
    margin\_ok = n\_inl >= MIN\_MARGIN \* max(second\_inl, 1)
    recognized = n\_good >= MIN\_GOOD and n\_inl >= MIN\_INLIERS and top\["ok"] and margin\_ok
    reason = ("OK" if recognized else
              "too few reliable matches" if n\_good < MIN\_GOOD else
              "too few RANSAC inliers" if n\_inl < MIN\_INLIERS else
              "unreliable geometry (invalid polygon)" if not top\["ok"] else
              "ambiguous (runner-up too close)")
```

### Test images for Task 6

Rotation is done with `rotate\_bound` so nothing gets cropped. Brightness is reduced with `cv2.convertScaleAbs(img, alpha=0.35)` and resize is 50% with `cv2.INTER\_AREA`.

```python
def rotate\_bound(img, angle=45):
    h, w = img.shape\[:2]; c = (w/2, h/2); M = cv2.getRotationMatrix2D(c, angle, 1.0)
    cos, sin = abs(M\[0, 0]), abs(M\[0, 1]); nw, nh = int(h\*sin + w\*cos), int(h\*cos + w\*sin)
    M\[0, 2] += nw/2 - c\[0]; M\[1, 2] += nh/2 - c\[1]
    return cv2.warpAffine(img, M, (nw, nh))                      # black borders, nothing cropped

variants = {
    "Original":           lambda p, p2: cv2.imread(p),
    "Rotated (45 deg)":   lambda p, p2: rotate\_bound(cv2.imread(p), 45),
    "Resized (50%)":      lambda p, p2: cv2.resize(cv2.imread(p), None, fx=0.5, fy=0.5, interpolation=cv2.INTER\_AREA),
    "Low brightness":     lambda p, p2: cv2.convertScaleAbs(cv2.imread(p), alpha=0.35, beta=0),
    "Different viewpoint":lambda p, p2: cv2.imread(p2),
}
```

## 5\. Application output (screenshots)

Every screenshot has the query image, its SIFT keypoints, and the matches with the best reference image. All the screenshots below are of a Christ the Redeemer query (the notebook took the first landmark in the list).

**Original**

!\[original](outputs/shot\_original.png)

Console output of this kind of run (the demo query):

```
Demo query (true: Christ the Redeemer)
Ranking (landmark | reliable matches | RANSAC inliers):
   Colosseum                  6     0
   Taj Mahal                  6     0
   Great Wall of China        5     0
   Pyramids of Giza           5     0
   Statue of Liberty          5     0
   Christ the Redeemer        4     0
Decision: REJECTED (too few reliable matches) | matches=6 inliers=0 keypoints=135 time=0.236s
```

**Rotated (45 deg)**

!\[rotated](outputs/shot\_rotated\_45\_deg\_.png)

**Resized (50%)**

!\[resized](outputs/shot\_resized\_50\_.png)

**Low brightness**

!\[low brightness](outputs/shot\_low\_brightness.png)

**Different viewpoint**

!\[viewpoint](outputs/shot\_different\_viewpoint.png)

**Unknown landmark (Stonehenge, not in database)**

!\[unknown](outputs/shot\_unknown.png)

```
Demo query (true: stonehenge, NOT in database)
Ranking (landmark | reliable matches | RANSAC inliers):
   Colosseum                  9     6
   Christ the Redeemer       10     5
   Great Wall of China        9     5
   Pyramids of Giza           8     5
   Statue of Liberty          6     0
   Taj Mahal                  6     0
Decision: REJECTED (too few reliable matches) | matches=9 inliers=6 keypoints=243 time=0.371s
```

I also tried one image of my own with the upload cell (`images.jpg`):

```
Ranking (landmark | reliable matches | RANSAC inliers):
   Christ the Redeemer       57     8
   Statue of Liberty         38     7
   Colosseum                 24     7
   Pyramids of Giza          34     6
   Great Wall of China       25     6
   Taj Mahal                 21     6
Decision: REJECTED (too few RANSAC inliers) | matches=57 inliers=8 keypoints=1407 time=2.352s
```

## 6\. Experimental results

This is the full table saved by my code in `experimental\_results.csv`. "Original" means another photo of the same landmark which is not in the reference set. "Different viewpoint" is a second held-out photo. "Correct" is Yes if the right landmark was recognized, or if an unknown image was rejected.

|#|True landmark|Condition|Predicted|Keypoints|Reliable matches|RANSAC inliers|Decision|Correct|Time (s)|
|-|-|-|-|-|-|-|-|-|-|
|1|Christ the Redeemer|Original|Unknown (rejected)|135|6|0|Rejected|No|0.259|
|2|Christ the Redeemer|Rotated (45 deg)|Unknown (rejected)|168|11|5|Rejected|No|0.33|
|3|Christ the Redeemer|Resized (50%)|Unknown (rejected)|107|6|0|Rejected|No|0.122|
|4|Christ the Redeemer|Low brightness|Unknown (rejected)|41|3|0|Rejected|No|0.08|
|5|Christ the Redeemer|Different viewpoint|Unknown (rejected)|319|15|13|Rejected|No|0.378|
|6|Colosseum|Original|Unknown (rejected)|541|14|5|Rejected|No|0.866|
|7|Colosseum|Rotated (45 deg)|Unknown (rejected)|754|12|7|Rejected|No|1.211|
|8|Colosseum|Resized (50%)|Unknown (rejected)|140|8|0|Rejected|No|0.184|
|9|Colosseum|Low brightness|Unknown (rejected)|198|7|0|Rejected|No|0.226|
|10|Colosseum|Different viewpoint|Unknown (rejected)|599|10|5|Rejected|No|0.675|
|11|Great Wall of China|Original|Unknown (rejected)|326|9|5|Rejected|No|0.384|
|12|Great Wall of China|Rotated (45 deg)|Unknown (rejected)|444|8|6|Rejected|No|0.646|
|13|Great Wall of China|Resized (50%)|Unknown (rejected)|102|3|0|Rejected|No|0.136|
|14|Great Wall of China|Low brightness|Unknown (rejected)|146|6|0|Rejected|No|0.163|
|15|Great Wall of China|Different viewpoint|Unknown (rejected)|317|12|5|Rejected|No|0.378|
|16|Pyramids of Giza|Original|Unknown (rejected)|138|6|0|Rejected|No|0.196|
|17|Pyramids of Giza|Rotated (45 deg)|Unknown (rejected)|205|9|7|Rejected|No|0.6|
|18|Pyramids of Giza|Resized (50%)|Unknown (rejected)|62|4|0|Rejected|No|0.141|
|19|Pyramids of Giza|Low brightness|Unknown (rejected)|12|2|0|Rejected|No|0.044|
|20|Pyramids of Giza|Different viewpoint|Unknown (rejected)|2000|26|8|Rejected|No|3.291|
|21|Statue of Liberty|Original|Unknown (rejected)|171|19|16|Rejected|No|0.198|
|22|Statue of Liberty|Rotated (45 deg)|Statue of Liberty|229|24|16|Recognized|Yes|0.306|
|23|Statue of Liberty|Resized (50%)|Unknown (rejected)|61|4|0|Rejected|No|0.07|
|24|Statue of Liberty|Low brightness|Unknown (rejected)|56|5|0|Rejected|No|0.066|
|25|Statue of Liberty|Different viewpoint|Unknown (rejected)|303|14|7|Rejected|No|0.342|
|26|Taj Mahal|Original|Unknown (rejected)|229|9|5|Rejected|No|0.253|
|27|Taj Mahal|Rotated (45 deg)|Unknown (rejected)|282|19|6|Rejected|No|0.384|
|28|Taj Mahal|Resized (50%)|Unknown (rejected)|66|5|0|Rejected|No|0.071|
|29|Taj Mahal|Low brightness|Unknown (rejected)|50|5|0|Rejected|No|0.064|
|30|Taj Mahal|Different viewpoint|Unknown (rejected)|484|16|14|Rejected|No|0.542|
|31|stonehenge (not in DB)|Unknown landmark #1|Unknown (rejected)|243|9|6|Rejected|Yes|0.306|
|32|stonehenge (not in DB)|Unknown landmark #2|Unknown (rejected)|1652|53|7|Rejected|Yes|3.329|

### Summary by condition

|Condition|Tests|Avg keypoints|Avg reliable matches|Avg RANSAC inliers|Avg time (s)|Correct|
|-|-|-|-|-|-|-|
|Original|6|257|10.5|5.2|0.36|0/6|
|Rotated (45 deg)|6|347|13.8|7.8|0.58|1/6|
|Resized (50%)|6|90|5.0|0.0|0.12|0/6|
|Low brightness|6|84|4.7|0.0|0.11|0/6|
|Different viewpoint|6|670|15.5|8.7|0.93|0/6|
|Unknown landmark #1|1|243|9.0|6.0|0.31|1/1|
|Unknown landmark #2|1|1652|53.0|7.0|3.33|1/1|
|Overall|32|331|11.2|4.5|0.51|3/32 (9.4%)|

## 7\. Discussion

**Overall result.** Only 1 of my 30 known-landmark tests was recognized (Statue of Liberty, rotated 45 degrees, 24 reliable matches and 16 inliers). Both Stonehenge images were rejected correctly. So the overall accuracy is 3/32 = 9.4%. The program never gave a wrong landmark name, all the 29 mistakes are rejections, so it is safe but it rejects too much.

**Match count alone is not enough.** The second Stonehenge image got 53 reliable matches, which is more than any of the known landmark tests, but only 7 RANSAC inliers, so it was rejected. This is the reason I used inliers and a margin in the decision and did not only take the biggest match count.

**Rotation.** Rotated images did not do worse than originals (13.8 vs 10.5 average matches, 7.8 vs 5.2 inliers). This agrees with SIFT being rotation invariant. The rotated image also has a bigger canvas so it got a few more keypoints.

**Resize and low brightness.** These two failed completely. All 6 resized and all 6 low brightness images ended with 0 inliers. After resizing to 50% only 61 to 140 keypoints were left and after reducing brightness only 12 to 198 keypoints were left, so there was not enough to match.

**Different viewpoint.** This was the closest to working (avg 15.5 matches, 8.7 inliers). For Christ the Redeemer the best guess was correct with 15 matches and 13 inliers, and Taj Mahal had 16 matches and 14 inliers, but my threshold of 20 matches rejected them. The Statue of Liberty original also had 19 matches and 16 inliers, only one match short. So most of the rejections come from my `MIN\_GOOD = 20` threshold being too high for this dataset.

**Time.** Processing time was between 0.04 s and 3.33 s (average 0.51 s). It is higher when the image has many keypoints (the 2000 keypoint images took more than 3 s).

## 8\. Limitations

* Dataset quality is not good. Many images are small web thumbnails and some are not real photos (a map of the Great Wall, a computer made Colosseum, a slide with text "The Colosseum"). The median number of keypoints in my known-landmark queries was only 184 and 16 out of 30 had less than 200 keypoints.
* Watermarks and logos (Getty Images, BBC News) and people in the front give keypoints that are not part of the landmark, and some wrong matches in the screenshots go to these parts. The Christ the Redeemer original is almost a black shape in clouds so there are very few features.
* Only 8 reference images per landmark, and they are very different from each other. If the query angle is different there may be no reference image close enough. Homography also assumes a flat scene so it is not perfect for 3D buildings.
* Repeating patterns (arches in Colosseum, bricks of the wall, stones of pyramids) give similar descriptors and the ratio test removes many of them.
* I did not tune the thresholds (20 matches, 12 inliers, margin 1.3) on a separate validation set. Brute force matching with every reference image is also slow if the database is big.

## 9\. Conclusion

The system works in the way the lab asked: SIFT, BFMatcher with kNN (k=2), Lowe's ratio test and RANSAC homography. It did not make any false recognition, even for Stonehenge which gave many random matches. But in my experiment the accuracy was low (9.4%) because the dataset images are poor quality and my thresholds were strict. To make it better I would lower the match threshold to around 12 to 15, use cleaner and more reference images, improve dark images with CLAHE or gamma correction, and not downscale small images.

