# Skin Lesion Boundary Detection — Results

**Dataset:** Skin Cancer MNIST: HAM10000 (`kmader/skin-cancer-mnist-ham10000`), 10,015 images across 7 diagnosis classes.

## Lesion Area and Perimeter

| Image | Best Filter | Edge Method | Area (pixels) | Perimeter (pixels) |
|---|---|---|---|---|
| Image 1 (ISIC_0025438, nv) | Gaussian | Canny (50/100) | 19 | 19.1 |
| Image 2 (ISIC_0028847, mel) | Median | Canny (50/100) | 133 | 84.2 |
| Image 3 (ISIC_0028750, bkl) | Average | Canny (50/100) | 132 | 107.1 |
| Image 4 (ISIC_0031597, bcc) | Average | Canny (50/100) | 376 | 95.3 |
| Image 5 (ISIC_0024646, akiec) | Average | Canny (50/100) | 40 | 24.5 |

Every image's best threshold came out as 50/100 — see Question 3 below for why.

## Final Comparison

| Method | Noise Handling | Edge Quality | Boundary Detection | Overall Performance |
|---|---|---|---|---|
| Original + Sobel | Fair | Fair | Fair | Fair |
| Original + Canny | Poor | Fair | Poor | Poor |
| Average + Sobel | Good | Fair | Good | Fair |
| Average + Canny | Poor | Fair | Poor | Poor |
| Gaussian + Sobel | Good | Fair | Good | Fair |
| Gaussian + Canny | Poor | Fair | Poor | Poor |
| Median + Sobel | Good | Fair | Good | Fair |
| Median + Canny | Fair | Fair | Poor | Poor |

Scored over a stratified sample of 21 images across all 7 classes. Overall Performance is the weakest of the three individual ratings for that row.

A clear pattern stands out: every Sobel row beats its Canny counterpart on both noise handling and boundary detection. Averaged across the four preprocessing options, Sobel's noise-handling F1 is 0.67 against 0.34 for Canny, and Sobel finds a usable boundary far more often (up to 76%, with Average + Sobel the best combination) than Canny manages here (at most 29%, for Original + Canny). None of the four preprocessing filters changes that ordering.

## Questions to Answer

**1. Why is Gaussian filtering applied before Canny detection?**

Gaussian filtering is applied before Canny because Canny's gradient step reacts to any sharp intensity change, including the fine-grained noise and faint skin texture that HAM10000 photos contain, not just the lesion boundary. Smoothing those small fluctuations first means the gradient computed afterwards is dominated by the real edge between the lesion and the surrounding skin. This is visible in the comparison table above: Canny methods average an F1 of 0.34 against added noise versus 0.67 for Sobel, and within the Canny rows, no filtering at all ("Original + Canny") is consistently among the weakest of the four preprocessing choices.

**2. How did the three Canny threshold settings affect the result?**

Across the five images used for Tasks 1–6, the three threshold pairs did not agree on a single best setting — all five picked the 50/100 pair. Looking at the per-threshold averages, the 50/100 pair kept 63% of its edge pixels on a few long, continuous curves on average, against 0% for the 150/250 pair, and the largest contour it found covered 0.1% of the image on average versus 0.0% for the weaker pair. In practice, moving between the three thresholds traded off how much of the lesion outline survived against how much hair and skin-texture noise came along with it: the higher pairs (100/200, 150/250) were strict enough that they left almost no surviving edges at all on these particular images, so the trade-off collapsed in favour of the lowest pair.

**3. Which threshold produced the best lesion boundary?**

The 50/100 pair was selected for all 5 of the 5 task images, because it was the only pair that produced a usable amount of edge signal at all — edge coherence of 0.63 at 50/100 compared to essentially zero surviving edge pixels at the higher pairs. Its own boundary-detection success rate on these five images was still 0%, which is a limitation worth being upfront about: a low threshold being the *least bad* option is not the same as it being *good*, and this is picked up again in Questions 5 and 6.

**4. Why are edges useful for detecting skin lesions?**

A skin lesion is usually a different colour and texture from the surrounding healthy skin, which shows up as a ring of high intensity gradient at the boundary between the two regions. Edge detection turns that gradient ring into an explicit curve, which is what makes it possible to measure the lesion's area and perimeter directly, rather than only estimating lesion size by eye.

**5. What problems did you observe in detecting the lesion boundary?**

The two recurring problems were hair and low contrast. Several HAM10000 images contain dark hair strands crossing the lesion, and these produce their own strong edges that get picked up alongside the real boundary, fragmenting the detected contour into more than one piece. The second problem is low contrast between the lesion and the surrounding skin in some images (particularly lighter nevi), which is part of why "Boundary Detection" in the final comparison table never reaches a perfect score (best combination: Average + Sobel, success rate 76%). The two problems also compound the threshold issue in Question 3: because contrast is often low, only the most permissive threshold (50/100) picks up enough of the real boundary to work with, but that same permissiveness is what lets hair and skin texture through too.

**6. How could your method be improved?**

A dedicated hair-removal step before Canny (e.g. a black-hat morphological filter followed by inpainting, which is the standard approach in dermoscopy preprocessing) would likely reduce the fragmentation caused by hair. For low-contrast images, colour information could be brought in instead of relying on grayscale intensity alone, since the lesion and skin often differ more in hue/saturation than in brightness. The comparison table also suggests a more direct fix: Sobel outperformed Canny on every preprocessing option for both noise handling and boundary detection on this dataset, so a Sobel-based pipeline (ideally paired with Average or Median smoothing) would be a stronger starting point than Canny here, with Canny thresholds tuned per image — as Task 4 already does — rather than fixed for the whole dataset.
