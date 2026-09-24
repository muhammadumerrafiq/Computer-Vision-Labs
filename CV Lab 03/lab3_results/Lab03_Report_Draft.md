# Lab 03 - Edge Detection Techniques and Their Impact on Classification Performance

## 1. Introduction
Edges - abrupt changes in image intensity - describe object boundaries and are the classical building block of many vision pipelines. This lab implements the
Sobel, Prewitt, Laplacian, Laplacian-of-Gaussian (LoG) and Canny detectors, studies their behaviour under Gaussian and salt-and-pepper noise and after Gaussian / median
filtering, analyses the Canny thresholds and kernel size, and finally measures whether edge-only images help or hurt classification of skin lesions compared with the raw
images (Lab 01) and the best filtered images (Lab 02).

## 2. Methodology
* **Edge detectors** (grayscale input): Sobel Gx/Gy/magnitude (3x3), Prewitt (3x3), Laplacian (3x3), LoG (Gaussian sigma=1.5 + Laplacian), Canny (Gaussian smoothing + gradient + non-maximum suppression + hysteresis).
* **Noise:** additive Gaussian noise (sigma=25) and salt-and-pepper noise (5% of pixels). **Pre-processing:** Gaussian filter 5x5 (sigma 1) and median filter 3x3 (as in Lab 02).
* **Edge-quality metrics:** each edge map is compared with the edge map of the clean image (1-px tolerance): precision (false edges), recall (missing/broken edges), F1, fragments per 1000 edge pixels (continuity) and edge width (sharpness). Noise sensitivity = 1 - F1.
* **Canny study:** six configurations (three threshold pairs at 3x3, and three at 5x5); the best one was selected with a validation-only HOG + logistic-regression probe. The final edge method was chosen the same way among all detectors: **Canny (30/100, 5x5)**.
* **Classification sets:** A = raw images, B = Sharpening-filtered images (best filter of Lab 02), C = Canny (30/100, 5x5) edge images (replicated to 3 channels).

## 3. Experimental setup
* **Data:** 4-class ISIC subset (basal cell carcinoma, melanoma, nevus, pigmented benign keratosis); 1306 training / 327 validation images (80/20 split, seed 42) and 64 test images (16 per class), all resized to 224x224.
* **Classical models:** SVM (RBF), Random Forest (300 trees), KNN on 32x32 pixels + HOG features reduced with PCA (128 components); hyper-parameters tuned on the validation split.
* **CNNs:** CNN Model 1 = ResNet50, CNN Model 2 = AlexNet (ImageNet-pretrained), 12 epochs, AdamW (backbone 0.0001, head 0.001), cosine schedule with warm-up, class-weighted cross-entropy with label smoothing 0.05, mixed precision, batch 64, seeds [42].
* **Fairness:** identical split, epochs, optimiser, augmentation and metrics for Sets A, B and C. **Hardware:** Google Colab T4 GPU.
* **Metrics:** accuracy, macro precision / recall / F1, training time, single-image inference time.

## 4. Results
### 4.1 Task 1 - comparative edge detection
Figures: `fig1_all_edge_detectors.png`, `fig2_required_edge_comparison.png`.

### 4.2 Task 2 - noise and pre-processing (Table 1)
Figures: `fig3_noise_melanoma.png`, `fig3_noise_nevus.png`, `fig4_noise_robustness.png`.

| Edge Detector   | Input Image   | Noise Type    | Preprocessing      | Edge Quality                                | Noise Sensitivity     | Observations                                                                                                                                                                                                       |   F1 vs clean |
|:----------------|:--------------|:--------------|:-------------------|:--------------------------------------------|:----------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------:|
| Sobel           | Original      | None          | None               | Reference - medium width, partly fragmented | n/a (reference)       | Edge density 16.5% of pixels; width 2.01x skeleton; 56 fragments per 1000 edge px. Design: 1st-order gradient with built-in 3-px weighted smoothing; responses are wide.                                           |        nan    |
| Sobel           | Noisy         | Gaussian      | None               | Poor (F1=0.58)                              | High (drop 42 pp)     | 57% of detected edge px are false edges, 3% of clean edge px are missing/broken; density 16.5%->37.8%; 45 fragments/1000 px (clean 56); width 1.39x.                                                               |          0.58 |
| Sobel           | Noisy         | Salt & Pepper | None               | Poor (F1=0.49)                              | High (drop 51 pp)     | 59% of detected edge px are false edges, 30% of clean edge px are missing/broken; density 16.5%->24.5%; 65 fragments/1000 px (clean 56); width 1.64x.                                                              |          0.49 |
| Sobel           | Noisy         | Gaussian      | Gaussian Filter    | Fair (F1=0.68)                              | Moderate (drop 32 pp) | 45% of detected edge px are false edges, 6% of clean edge px are missing/broken; density 16.5%->32.8%; 52 fragments/1000 px (clean 56); width 1.97x. Filtering changed F1 by +0.10 vs the unfiltered noisy input.  |          0.68 |
| Sobel           | Noisy         | Salt & Pepper | Median Filter      | Good (F1=0.95)                              | Low (drop 5 pp)       | 3% of detected edge px are false edges, 6% of clean edge px are missing/broken; density 16.5%->16.6%; 38 fragments/1000 px (clean 56); width 2.47x. Filtering changed F1 by +0.46 vs the unfiltered noisy input.   |          0.95 |
| Prewitt         | Original      | None          | None               | Reference - medium width, partly fragmented | n/a (reference)       | Edge density 16.6% of pixels; width 2.05x skeleton; 56 fragments per 1000 edge px. Design: 1st-order gradient with unweighted 3-px smoothing; very close to Sobel, slightly noisier.                               |        nan    |
| Laplacian       | Original      | None          | None               | Reference - thin / sharp, highly fragmented | n/a (reference)       | Edge density 15.6% of pixels; width 1.30x skeleton; 192 fragments per 1000 edge px. Design: 2nd-order isotropic operator with no smoothing; thin but amplifies every noisy pixel.                                  |        nan    |
| LoG             | Noisy         | Gaussian      | Gaussian Filter    | Fair (F1=0.67)                              | Moderate (drop 33 pp) | 44% of detected edge px are false edges, 10% of clean edge px are missing/broken; density 17.3%->29.4%; 84 fragments/1000 px (clean 70); width 2.79x. Filtering changed F1 by +0.03 vs the unfiltered noisy input. |          0.67 |
| Canny           | Original      | None          | Built-in smoothing | Reference - thin / sharp, partly fragmented | n/a (reference)       | Edge density 0.4% of pixels; width 1.30x skeleton; 36 fragments per 1000 edge px. Design: smoothing + non-max suppression + hysteresis; 1-px edges linked into contours.                                           |        nan    |
| Canny           | Noisy         | Gaussian      | Gaussian Filter    | Poor (F1=0.17)                              | High (drop 83 pp)     | 67% of detected edge px are false edges, 85% of clean edge px are missing/broken; density 0.4%->0.1%; 30 fragments/1000 px (clean 36); width 1.36x. Filtering changed F1 by -0.15 vs the unfiltered noisy input.   |          0.17 |
| Canny           | Noisy         | Salt & Pepper | Median Filter      | Poor (F1=0.40)                              | High (drop 60 pp)     | 47% of detected edge px are false edges, 65% of clean edge px are missing/broken; density 0.4%->0.2%; 35 fragments/1000 px (clean 36); width 1.30x. Filtering changed F1 by +0.36 vs the unfiltered noisy input.   |          0.4  |

### 4.3 Task 3 - Canny parameters (Table 2)
Figures: `fig5_canny_parameters.png`, `fig6_canny_tradeoff.png`.

| Configuration   |   Low Threshold |   High Threshold | Kernel Size   | Edge Quality          | Number of Detected Edges   | Observation                                                                                                                                                                              |
|:----------------|----------------:|-----------------:|:--------------|:----------------------|:---------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Canny-1         |              30 |              100 | 3x3           | Fair (probe F1 38.0%) | 1186 px in 17 segments     | 2.90x the edge pixels of Canny-2, 17 segments/img (hysteresis ratio 3.3); clearly more weak edges (texture / hair) admitted than Canny-2; noise F1 0.16.                                 |
| Canny-2         |              50 |              150 | 3x3           | Poor (probe F1 35.3%) | 409 px in 6 segments       | 1.00x the edge pixels of Canny-2, 6 segments/img (hysteresis ratio 3.0); very sparse map - only the strongest gradients survive, parts of the lesion border may be lost; noise F1 0.33.  |
| Canny-3         |             100 |              200 | 3x3           | Poor (probe F1 31.6%) | 143 px in 3 segments       | 0.35x the edge pixels of Canny-2, 3 segments/img (hysteresis ratio 2.0); very sparse map - only the strongest gradients survive, parts of the lesion border may be lost; noise F1 0.28.  |
| Canny-4         |              50 |              150 | 5x5           | Poor (probe F1 28.0%) | 219 px in 3 segments       | 0.54x the edge pixels of Canny-2, 3 segments/img (hysteresis ratio 3.0); very sparse map - only the strongest gradients survive, parts of the lesion border may be lost; noise F1 0.31.  |
| Canny-5         |              30 |              100 | 5x5           | Best (probe F1 41.1%) | 672 px in 10 segments      | 1.65x the edge pixels of Canny-2, 10 segments/img (hysteresis ratio 3.3); very sparse map - only the strongest gradients survive, parts of the lesion border may be lost; noise F1 0.35. |
| Canny-6         |             100 |              200 | 5x5           | Poor (probe F1 21.3%) | 59 px in 1 segments        | 0.14x the edge pixels of Canny-2, 1 segments/img (hysteresis ratio 2.0); very sparse map - only the strongest gradients survive, parts of the lesion border may be lost; noise F1 0.06.  |

Detector selection (validation probe):

| Detector                  |   Density (%) |   Noise F1 (Gaussian) |   Probe val macro-F1 (%) |   Probe val acc (%) |
|:--------------------------|--------------:|----------------------:|-------------------------:|--------------------:|
| Canny (best: 30/100, 5x5) |          1.34 |                  0.35 |                    41.09 |               42.2  |
| Prewitt                   |         16.48 |                  0.58 |                    40.18 |               40.06 |
| Sobel                     |         16.4  |                  0.58 |                    38.17 |               38.23 |
| LoG                       |         16.62 |                  0.65 |                    38.14 |               38.23 |
| Laplacian                 |         13.74 |                  0.67 |                    35.9  |               36.09 |

### 4.4 Tasks 4-5 - classification (Table 3)
| Model / Classifier     |   Accuracy Raw (Lab 1) |   Accuracy Filtered (Lab 2) |   Accuracy Edge (Lab 3) |   Precision |   Recall |   F1-Score |   Training Time (s) |   Inference Time (ms) |   Lab 1 reported acc. (ref.) |   Edge - Raw (pp) |
|:-----------------------|-----------------------:|----------------------------:|------------------------:|------------:|---------:|-----------:|--------------------:|----------------------:|-----------------------------:|------------------:|
| SVM                    |                  57.81 |                       60.94 |                   32.81 |       33.01 |    32.81 |      32.38 |                4.09 |                  2.84 |                       nan    |            -25    |
| Random Forest          |                  51.56 |                       54.69 |                   26.56 |       29.51 |    26.56 |      26.17 |                8.52 |                  3.46 |                       nan    |            -25    |
| KNN                    |                  45.31 |                       43.75 |                   35.94 |       34.17 |    35.94 |      34.19 |                3.9  |                  2.64 |                       nan    |             -9.38 |
| CNN Model 1 (ResNet50) |                  70.31 |                       68.75 |                   40.62 |       38.65 |    40.62 |      39.23 |               53.72 |                  7.01 |                        78.12 |            -29.69 |
| CNN Model 2 (AlexNet)  |                  67.19 |                       62.5  |                   45.31 |       43.35 |    45.31 |      43.35 |               14.46 |                  0.83 |                        75    |            -21.88 |

Full per-representation results (Table 3b):

| Model                  | Representation   |   Runs |   Accuracy (%) |   Accuracy std |   Precision (%) |   Recall (%) |   F1-Score (%) |   Val F1 (%) |   Training Time (s) |   Inference Time (ms) |
|:-----------------------|:-----------------|-------:|---------------:|---------------:|----------------:|-------------:|---------------:|-------------:|--------------------:|----------------------:|
| SVM                    | Raw              |      1 |          57.81 |              0 |           62.19 |        57.81 |          57.78 |        66.77 |                4.89 |                  2.89 |
| SVM                    | Filtered         |      1 |          60.94 |              0 |           61.88 |        60.94 |          60.61 |        67.69 |                4.66 |                  2.81 |
| SVM                    | Edge             |      1 |          32.81 |              0 |           33.01 |        32.81 |          32.38 |        45.25 |                4.09 |                  2.84 |
| Random Forest          | Raw              |      1 |          51.56 |              0 |           53.48 |        51.56 |          50.76 |        64.37 |                8.73 |                  4.45 |
| Random Forest          | Filtered         |      1 |          54.69 |              0 |           59.31 |        54.69 |          54.62 |        63.18 |                9.84 |                  3.44 |
| Random Forest          | Edge             |      1 |          26.56 |              0 |           29.51 |        26.56 |          26.17 |        45.72 |                8.52 |                  3.46 |
| KNN                    | Raw              |      1 |          45.31 |              0 |           37.92 |        45.31 |          40.15 |        58.02 |                4.73 |                  2.69 |
| KNN                    | Filtered         |      1 |          43.75 |              0 |           36.71 |        43.75 |          38.43 |        58.87 |                4.54 |                  2.62 |
| KNN                    | Edge             |      1 |          35.94 |              0 |           34.17 |        35.94 |          34.19 |        42.54 |                3.9  |                  2.64 |
| CNN Model 1 (ResNet50) | Raw              |      1 |          70.31 |              0 |           78.93 |        70.31 |          64.91 |        80.17 |               61.47 |                  5.98 |
| CNN Model 1 (ResNet50) | Filtered         |      1 |          68.75 |              0 |           53.65 |        68.75 |          59.68 |        79.43 |               54.41 |                  8.35 |
| CNN Model 1 (ResNet50) | Edge             |      1 |          40.62 |              0 |           38.65 |        40.62 |          39.23 |        47.41 |               53.72 |                  7.01 |
| CNN Model 2 (AlexNet)  | Raw              |      1 |          67.19 |              0 |           73.67 |        67.19 |          65.75 |        82.15 |               21.81 |                  0.79 |
| CNN Model 2 (AlexNet)  | Filtered         |      1 |          62.5  |              0 |           62.7  |        62.5  |          56.53 |        79.81 |               14.54 |                  0.8  |
| CNN Model 2 (AlexNet)  | Edge             |      1 |          45.31 |              0 |           43.35 |        45.31 |          43.35 |        48.85 |               14.46 |                  0.83 |

### 4.5 Task 6 - visual comparison
Figures: `fig7_three_datasets.png`, `fig8_learning_curves.png`, `fig9_confusion_matrices.png` (best model: **CNN Model 2 (AlexNet)**), `fig10_metric_bars.png`, `fig11_all_models.png`.

## 5. Discussion
**Question 1 - Edge detection and noise.** Canny was the most noise-sensitive detector: its edge map kept only F1 = 0.33 (Gaussian noise) and 0.04 (salt-and-pepper) of the clean edge structure (mean 0.19), with 74% / 98% false-edge pixels. Ranking from most to least sensitive (mean F1 vs clean): Canny 0.19, Prewitt 0.54, Sobel 0.54, Laplacian 0.54, LoG 0.60. A derivative filter boosts high spatial frequencies, and noise lives there; the higher the derivative order and the less smoothing built into the operator, the larger the response to noise (Laplacian > Sobel/Prewitt > LoG/Canny). Your ranking differs from this expectation for the un-smoothed Laplacian being the worst.

**Question 2 - Effect of filtering.** Effect of pre-filtering on edge fidelity (F1 vs clean edge map, averaged over the 5 detectors): Gaussian noise + Gaussian filter: mean F1 change +0.02 (largest gain Prewitt +0.10; smallest Canny -0.15) | Gaussian noise + Median filter: mean F1 change +0.03 (largest gain Prewitt +0.05; smallest Laplacian -0.01) | Salt-and-pepper noise + Gaussian filter: mean F1 change +0.09 (largest gain Canny +0.18; smallest LoG +0.02) | Salt-and-pepper noise + Median filter: mean F1 change +0.43 (largest gain Laplacian +0.46; smallest Canny +0.36). For Gaussian noise the Median filter helped most; for salt-and-pepper noise the Median filter helped most - this differs from the usual expectation (Gaussian filter for Gaussian noise, median filter for impulse noise; see Figure 3). Gaussian filtering averages independent noise away but also blurs true edges (thicker, weaker responses); the median filter removes impulse outliers completely without averaging them into their neighbours and keeps edges sharper, which is why it is the natural choice for salt-and-pepper noise.

**Question 3 - Canny parameters.** Lowering the thresholds from 100/200 (Canny-3) to 30/100 (Canny-1) raised the mean edge count from 143 to 1186 pixels per image (8.3x) and the number of segments from 3 to 17: weak gradients (skin texture, hair, noise) are admitted, the map becomes cluttered and its Gaussian-noise fidelity changes from F1 0.28 to 0.16. Raising them keeps only the strongest edges - the map is cleaner, but weak parts of the lesion border can break and small structures vanish. The middle setting (50/150) gave 409 px. Increasing the Gaussian kernel from 3x3 to 5x5 at 50/150 changed the count from 409 to 219 px and noise F1 from 0.33 to 0.31. Selected configuration: Canny-5 (low 30, high 100, kernel 5x5); the best edge representation overall was Canny (30/100, 5x5).

**Question 4 - Edge maps and classification.** On average, edge-only input reduced accuracy by -22.19 pp relative to raw images (5 of 5 models got worse). SVM: 57.8% -> 32.8% (-25.0 pp); Random Forest: 51.6% -> 26.6% (-25.0 pp); KNN: 45.3% -> 35.9% (-9.4 pp); CNN Model 1 (ResNet50): 70.3% -> 40.6% (-29.7 pp); CNN Model 2 (AlexNet): 67.2% -> 45.3% (-21.9 pp). Edge-only images throw away colour (pigment colour and its variegation is a first-order diagnostic cue for melanoma vs nevus), absolute intensity/shading and the inner texture of the lesion; they keep hair, ruler marks and skin creases as if they were lesion structure, and depend on threshold choices. The ImageNet-pretrained CNNs also expect natural RGB statistics, and information deleted in pre-processing can never be recovered by later layers.

**Question 5 - Information loss.** An edge map keeps only *where* intensity changes abruptly. Lost: (1) colour - lesion pigmentation, colour variegation and blue-white structures; (2) absolute intensity / shading - dark vs light lesions, regions of regression; (3) texture and inner structure - smooth gradients and fine pigment networks below the threshold vanish; (4) region information - what is inside and outside a boundary; and (5) edge strength when the map is binarised. What is kept is mainly shape/border geometry, which is only part of the ABCD-type diagnostic cues.

**Question 6 - Classical vs. deep features.** A CNN's early layers learn Gabor-/Sobel-like and colour-opponent filters on their own, so hand-made edge maps are largely redundant. Letting the network learn them (a) keeps all colour and texture, (b) tunes filter scale, orientation and threshold to the task instead of a fixed threshold guessed by hand, (c) learns hierarchical combinations (edges -> textures -> parts -> lesion patterns) end-to-end, (d) is robust because the filters are optimised jointly with noise and augmentation, and (e) removes the manual parameter search (Canny thresholds, kernel size) that Task 3 needed.

**Question 7 - Best representation.** Averaged over all models, mean accuracy / macro-F1 were: Raw: 58.4% / 55.9%; Filtered: 58.1% / 54.0%; Edge: 36.2% / 35.1%. The most useful representation was therefore **Raw**. The single best run was CNN Model 2 (AlexNet) on Raw images (67.2% accuracy, 65.7% macro-F1). Note that the test set has only 64 images (1 image = 1.56 pp), so use the validation columns in Table 3b and, if possible, several seeds (`SEEDS = [42, 43, 44]`) before claiming a difference.

## 6. Conclusion
The edge study showed that detectors that differentiate without smoothing are the most noise-sensitive, that Gaussian smoothing and median filtering repair different kinds of noise,
and that the Canny thresholds trade completeness against clutter (selected: Canny-5). For classification, edge-only images changed mean accuracy by -22.19 pp relative to raw images
(filtered vs raw: -0.31 pp), and the best-performing model was CNN Model 2 (AlexNet). Because the test set contains only 64 images, differences of a few
percentage points are not statistically reliable; the conclusions above should be read together with the validation results and, ideally, repeated over several seeds.

## 7. References
1. J. Canny, "A computational approach to edge detection," *IEEE Trans. Pattern Analysis and Machine Intelligence*, vol. 8, no. 6, pp. 679-698, 1986.
2. D. Marr and E. Hildreth, "Theory of edge detection," *Proc. Royal Society of London B*, vol. 207, pp. 187-217, 1980.
3. R. C. Gonzalez and R. E. Woods, *Digital Image Processing*, 4th ed., Pearson, 2018.
4. K. He, X. Zhang, S. Ren and J. Sun, "Deep residual learning for image recognition," *CVPR*, 2016.
5. A. Krizhevsky, I. Sutskever and G. Hinton, "ImageNet classification with deep convolutional neural networks," *NeurIPS*, 2012.
6. G. Bradski, "The OpenCV library," *Dr. Dobb's Journal of Software Tools*, 2000.
7. N. Dalal and B. Triggs, "Histograms of oriented gradients for human detection," *CVPR*, 2005.
8. Skin Cancer ISIC dataset (Kaggle: nodoubttome/skin-cancer9-classesisic), derived from the International Skin Imaging Collaboration archive.
