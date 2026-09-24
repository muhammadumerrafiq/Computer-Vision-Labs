# Lab 03 Tables and Discussion Questions

## Table 1. Effect of Noise and Preprocessing on Edge Detection

  -------------------------------------------------------------------------------------------------------------------------
  Edge        Input      Noise Type Preprocessing   Edge Quality  Noise         Observations
  Detector    Image                                               Sensitivity   
  ----------- ---------- ---------- --------------- ------------- ------------- -------------------------------------------
  Sobel       Original   None       None            Reference -   n/a           Edge density 16.5% of pixels; width 2.01x
                                                    medium width, (reference)   skeleton; 56 fragments per 1000 edge px.
                                                    partly                      Design: 1st-order gradient with built-in
                                                    fragmented                  3-px weighted smoothing; responses are
                                                                                wide.

  Sobel       Noisy      Gaussian   None            Poor          High (drop 42 57% of detected edge px are false edges, 3%
                                                    (F1=0.58)     pp)           of clean edge px are missing/broken;
                                                                                density 16.5%-\>37.8%; 45 fragments/1000 px
                                                                                (clean 56); width 1.39x.

  Sobel       Noisy      Salt &     None            Poor          High (drop 51 59% of detected edge px are false edges,
                         Pepper                     (F1=0.49)     pp)           30% of clean edge px are missing/broken;
                                                                                density 16.5%-\>24.5%; 65 fragments/1000 px
                                                                                (clean 56); width 1.64x.

  Sobel       Noisy      Gaussian   Gaussian Filter Fair          Moderate      45% of detected edge px are false edges, 6%
                                                    (F1=0.68)     (drop 32 pp)  of clean edge px are missing/broken;
                                                                                density 16.5%-\>32.8%; 52 fragments/1000 px
                                                                                (clean 56); width 1.97x. Filtering changed
                                                                                F1 by +0.10 vs the unfiltered noisy input.

  Sobel       Noisy      Salt &     Median Filter   Good          Low (drop 5   3% of detected edge px are false edges, 6%
                         Pepper                     (F1=0.95)     pp)           of clean edge px are missing/broken;
                                                                                density 16.5%-\>16.6%; 38 fragments/1000 px
                                                                                (clean 56); width 2.47x. Filtering changed
                                                                                F1 by +0.46 vs the unfiltered noisy input.

  Prewitt     Original   None       None            Reference -   n/a           Edge density 16.6% of pixels; width 2.05x
                                                    medium width, (reference)   skeleton; 56 fragments per 1000 edge px.
                                                    partly                      Design: 1st-order gradient with unweighted
                                                    fragmented                  3-px smoothing; very close to Sobel,
                                                                                slightly noisier.

  Laplacian   Original   None       None            Reference -   n/a           Edge density 15.6% of pixels; width 1.30x
                                                    thin / sharp, (reference)   skeleton; 192 fragments per 1000 edge px.
                                                    highly                      Design: 2nd-order isotropic operator with
                                                    fragmented                  no smoothing; thin but amplifies every
                                                                                noisy pixel.

  LoG         Noisy      Gaussian   Gaussian Filter Fair          Moderate      44% of detected edge px are false edges,
                                                    (F1=0.67)     (drop 33 pp)  10% of clean edge px are missing/broken;
                                                                                density 17.3%-\>29.4%; 84 fragments/1000 px
                                                                                (clean 70); width 2.79x. Filtering changed
                                                                                F1 by +0.03 vs the unfiltered noisy input.

  Canny       Original   None       Built-in        Reference -   n/a           Edge density 0.4% of pixels; width 1.30x
                                    smoothing       thin / sharp, (reference)   skeleton; 36 fragments per 1000 edge px.
                                                    partly                      Design: smoothing + non-max suppression +
                                                    fragmented                  hysteresis; 1-px edges linked into
                                                                                contours.

  Canny       Noisy      Gaussian   Gaussian Filter Poor          High (drop 83 67% of detected edge px are false edges,
                                                    (F1=0.17)     pp)           85% of clean edge px are missing/broken;
                                                                                density 0.4%-\>0.1%; 30 fragments/1000 px
                                                                                (clean 36); width 1.36x. Filtering changed
                                                                                F1 by -0.15 vs the unfiltered noisy input.

  Canny       Noisy      Salt &     Median Filter   Poor          High (drop 60 47% of detected edge px are false edges,
                         Pepper                     (F1=0.40)     pp)           65% of clean edge px are missing/broken;
                                                                                density 0.4%-\>0.2%; 35 fragments/1000 px
                                                                                (clean 36); width 1.30x. Filtering changed
                                                                                F1 by +0.36 vs the unfiltered noisy input.
  -------------------------------------------------------------------------------------------------------------------------

## Table 2. Canny Parameter Analysis

  -----------------------------------------------------------------------------------------------------------------
  Configuration           Low        High Kernel   Edge      Number of  Observation
                    Threshold   Threshold Size     Quality   Detected   
                                                             Edges      
  --------------- ----------- ----------- -------- --------- ---------- -------------------------------------------
  Canny-1                  30         100 3x3      Fair      1186 px in 2.90x the edge pixels of Canny-2, 17
                                                   (probe F1 17         segments/img (hysteresis ratio 3.3);
                                                   38.0%)    segments   clearly more weak edges (texture / hair)
                                                                        admitted than Canny-2; noise F1 0.16.

  Canny-2                  50         150 3x3      Poor      409 px in  1.00x the edge pixels of Canny-2, 6
                                                   (probe F1 6 segments segments/img (hysteresis ratio 3.0); very
                                                   35.3%)               sparse map - only the strongest gradients
                                                                        survive, parts of the lesion border may be
                                                                        lost; noise F1 0.33.

  Canny-3                 100         200 3x3      Poor      143 px in  0.35x the edge pixels of Canny-2, 3
                                                   (probe F1 3 segments segments/img (hysteresis ratio 2.0); very
                                                   31.6%)               sparse map - only the strongest gradients
                                                                        survive, parts of the lesion border may be
                                                                        lost; noise F1 0.28.

  Canny-4                  50         150 5x5      Poor      219 px in  0.54x the edge pixels of Canny-2, 3
                                                   (probe F1 3 segments segments/img (hysteresis ratio 3.0); very
                                                   28.0%)               sparse map - only the strongest gradients
                                                                        survive, parts of the lesion border may be
                                                                        lost; noise F1 0.31.
  -----------------------------------------------------------------------------------------------------------------

## Table 3. Cross-Lab Classification Performance Comparison

  -----------------------------------------------------------------------------------------------------
  Model /        Accuracy    Accuracy   Accuracy   Precision   Recall   F1-Score   Training   Inference
  Classifier     Raw (Lab    Filtered  Edge (Lab                                   Time (s)   Time (ms)
                       1)     (Lab 2)         3)                                            
  ------------ ---------- ----------- ---------- ----------- -------- ---------- ---------- -----------
  SVM               57.81       60.94      32.81       33.01    32.81      32.38       4.09        2.84

  Random            51.56       54.69      26.56       29.51    26.56      26.17       8.52        3.46
  Forest                                                                                    

  KNN               45.31       43.75      35.94       34.17    35.94      34.19        3.9        2.64

  CNN Model 1       70.31       68.75      40.62       38.65    40.62      39.23      53.72        7.01
  (ResNet50)                                                                                

  CNN Model 2       67.19        62.5      45.31       43.35    45.31      43.35      14.46        0.83
  (AlexNet)                                                                                 
  -----------------------------------------------------------------------------------------------------

## Discussion Questions

### Question 1: Edge Detection and Noise

Canny was the most sensitive to noise in this experiment. Its mean F1
against the clean edge structure under unfiltered Gaussian and
salt-and-pepper noise was 0.19. For Gaussian noise, the F1 was 0.17, and
for salt-and-pepper noise it was 0.40. The notebook also showed that
Canny produced many false edges under noisy conditions. This happened
because noise creates small intensity changes that can also be detected
as edges.

### Question 2: Effect of Filtering

Filtering generally improved the quality of the detected edges,
especially for salt-and-pepper noise. With salt-and-pepper noise, Median
filtering gave a large improvement, for example Sobel F1 increased from
0.49 to 0.95. For Gaussian noise, both filters helped in several cases,
although the improvement was not the same for every detector. Gaussian
filtering reduced noise but could also blur real edges, while Median
filtering was better at removing isolated noisy pixels and keeping the
main edges.

### Question 3: Canny Parameters

Changing the Canny thresholds had a clear effect on the number of
detected edges. With thresholds 30/100 and a 3x3 kernel, the average was
1186 edge pixels per image. At 50/150 it dropped to 409 pixels, and at
100/200 it dropped further to 143 pixels. Lower thresholds allowed more
weak edges such as texture and hair to appear, while higher thresholds
kept only stronger edges and could remove parts of the lesion boundary.
The best configuration in the experiment was Canny-5, with low threshold
30, high threshold 100, and a 5x5 kernel. It had a probe validation
macro-F1 of 41.09%.

### Question 4: Edge Maps and Classification

Using only edge images reduced classification accuracy compared with raw
images in all five tested models. The accuracy changed from 57.81% to
32.81% for SVM, 51.56% to 26.56% for Random Forest, 45.31% to 35.94% for
KNN, 70.31% to 40.62% for ResNet50, and 67.19% to 45.31% for AlexNet.
Overall, edge-only images reduced the average accuracy by 22.19
percentage points. A likely reason is that edge maps remove useful
colour, intensity, shading, and texture information that is present in
the original images.

### Question 5: Information Loss

When an image is converted to an edge map, mainly the boundaries and
sudden intensity changes are kept. Information such as colour,
pigmentation, brightness, shading, texture, and the internal structure
of the lesion can be lost. In this dataset, these details can be useful
for distinguishing between different skin-lesion classes. The edge map
therefore gives useful shape information, but it does not contain all
the information available in the original image.

### Question 6: Classical vs. Deep Features

A CNN can learn edge-like features automatically in its early layers.
This has the advantage that the network can learn which edges, textures,
colours, and patterns are useful for the classification task instead of
depending on manually selected edge detectors and thresholds. The
learned features can also be combined in later layers to identify more
complex patterns. Because of this, manually converting the complete
image into an edge-only image can remove information that a CNN could
have used.

### Question 7: Best Representation

Based on the results, raw images produced the most useful overall
classification results. Across the tested models, the average accuracy
was 58.4% for raw images, 58.1% for filtered images, and 36.2% for edge
images. The single best result was obtained by CNN Model 2, AlexNet,
using raw images, with 67.19% accuracy and 65.75% macro F1-score. The
results therefore show that edge images were useful for studying image
structure, but they did not perform as well as the raw images for
classification in this experiment.
