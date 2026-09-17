# Lab Task 02: Effect of Image Filtering on Skin-Lesion Classification

**Course:** Computer Vision
**Dataset:** 4-class ISIC subset (basal cell carcinoma, melanoma, nevus, pigmented benign keratosis)
**Split:** 1633 train / 409 validation / 64 test, same seed-42 split as Lab 01
**Runs:** 3 models x 6 filter conditions = 18 training runs

---

## 1. Setup

I took the three best models from Lab 01 and retrained each one six times: once on the original
images, and once for each of the five filters. Everything except the filter was kept fixed. Same
split, same augmentation, same optimiser, same schedule, same seed, same evaluation code. So any
difference in the numbers below comes from the filter and nothing else.

Filters used:

| Filter | Implementation | Kernel |
|---|---|---|
| Average | `cv2.blur` | 3x3 box |
| Gaussian | `cv2.GaussianBlur` | 5x5, sigma = 1.0 |
| Median | `cv2.medianBlur` | 3x3 |
| Sharpening | `cv2.filter2D` | `[[0,-1,0],[-1,5,-1],[0,-1,0]]` |
| Sobel | gradient magnitude on grayscale, min-max scaled, copied to 3 channels | 3x3 Sobel x and y |

Every filter was applied to the 224x224 image, which is the resolution the network actually sees,
and applied to train, validation and test alike.

---

## 2. Main results table (test set, 64 images)

| Model | Filter | Accuracy | Precision | Recall | F1-score | Macro-F1 | AUC |
|---|---|---|---|---|---|---|---|
| ResNet50 | No Filter | 67.19 | 78.01 | 67.19 | 60.47 | 60.47 | 92.94 |
| ResNet50 | Average | 68.75 | 76.93 | 68.75 | 64.84 | 64.84 | 87.60 |
| ResNet50 | Gaussian | 67.19 | 75.63 | 67.19 | 61.86 | 61.86 | 89.84 |
| ResNet50 | Median | 65.62 | 75.08 | 65.62 | 58.91 | 58.91 | 89.42 |
| **ResNet50** | **Sharpening** | **71.88** | **79.96** | **71.88** | **66.17** | **66.17** | **94.76** |
| ResNet50 | Sobel | 60.94 | 61.72 | 60.94 | 59.72 | 59.72 | 83.33 |
| AlexNet | No Filter | 67.19 | 73.20 | 67.19 | 64.85 | 64.85 | 90.20 |
| AlexNet | Average | 64.06 | 50.35 | 64.06 | 55.58 | 55.58 | 83.56 |
| AlexNet | Gaussian | 65.62 | 77.51 | 65.62 | 61.23 | 61.23 | 84.70 |
| AlexNet | Median | 65.62 | 77.11 | 65.62 | 60.66 | 60.66 | 88.57 |
| AlexNet | Sharpening | 60.94 | 61.74 | 60.94 | 54.87 | 54.87 | 89.36 |
| AlexNet | Sobel | 57.81 | 48.93 | 57.81 | 51.18 | 51.18 | 87.99 |
| ResNet101 | No Filter | 67.19 | 76.54 | 67.19 | 60.31 | 60.31 | 93.59 |
| ResNet101 | Average | 70.31 | 79.64 | 70.31 | 63.05 | 63.05 | 88.15 |
| ResNet101 | Gaussian | 64.06 | 73.90 | 64.06 | 57.51 | 57.51 | 86.30 |
| ResNet101 | Median | 68.75 | 79.59 | 68.75 | 61.98 | 61.98 | 91.31 |
| ResNet101 | Sharpening | 68.75 | 79.59 | 68.75 | 61.98 | 61.98 | 91.96 |
| ResNet101 | Sobel | 57.81 | 61.72 | 57.81 | 57.01 | 57.01 | 81.35 |

Best run overall: **ResNet50 + Sharpening**, 71.88% accuracy, 66.17 macro-F1, 94.76 AUC.

Two notes on this table. The Precision, Recall and F1-score columns are weighted averages and
Macro-F1 is the macro average. Because the test set has exactly 16 images per class, the weighted
and macro averages come out identical, so the F1-score and Macro-F1 columns are the same numbers.
Same reason the Recall column equals the Accuracy column, since weighted recall is accuracy by
definition. That is not a copy error, it is what a perfectly balanced test set does.

---

## 3. Change relative to the unfiltered baseline (test set)

| Model | Filter | dAccuracy | dMacro-F1 | dBalancedAcc | dAUC |
|---|---|---|---|---|---|
| ResNet50 | Average | +1.56 | +4.37 | +1.56 | -5.34 |
| ResNet50 | Gaussian | 0.00 | +1.38 | 0.00 | -3.09 |
| ResNet50 | Median | -1.56 | -1.56 | -1.56 | -3.52 |
| ResNet50 | Sharpening | +4.69 | +5.70 | +4.69 | +1.82 |
| ResNet50 | Sobel | -6.25 | -0.75 | -6.25 | -9.60 |
| AlexNet | Average | -3.12 | -9.27 | -3.12 | -6.64 |
| AlexNet | Gaussian | -1.56 | -3.62 | -1.56 | -5.50 |
| AlexNet | Median | -1.56 | -4.19 | -1.56 | -1.63 |
| AlexNet | Sharpening | -6.25 | -9.98 | -6.25 | -0.85 |
| AlexNet | Sobel | -9.38 | -13.67 | -9.38 | -2.21 |
| ResNet101 | Average | +3.12 | +2.74 | +3.12 | -5.44 |
| ResNet101 | Gaussian | -3.12 | -2.80 | -3.12 | -7.29 |
| ResNet101 | Median | +1.56 | +1.68 | +1.56 | -2.28 |
| ResNet101 | Sharpening | +1.56 | +1.68 | +1.56 | -1.63 |
| ResNet101 | Sobel | -9.38 | -3.30 | -9.38 | -12.24 |

---

## 4. Validation results (409 images)

The test set is only 64 images, so one image is worth 1.5625 percentage points. A 3-point gap on
that table is a two-image difference and I do not trust it. I am putting the validation results
here as well because 409 images gives roughly 6x the resolution, and the pattern there is far
cleaner. Where the two tables disagree I go with the validation numbers.

| Model | Filter | Accuracy | Macro-F1 | Balanced Acc | AUC | dMacro-F1 |
|---|---|---|---|---|---|---|
| ResNet50 | No Filter | 80.73 | 80.40 | 80.21 | 95.42 | 0.00 |
| ResNet50 | Average | 81.96 | 81.42 | 81.44 | 95.59 | +1.02 |
| ResNet50 | Gaussian | 81.65 | 81.20 | 81.16 | 94.80 | +0.80 |
| ResNet50 | Median | 77.37 | 76.31 | 77.32 | 94.21 | -4.09 |
| ResNet50 | Sharpening | 82.87 | 82.33 | 82.45 | 94.52 | +1.93 |
| ResNet50 | Sobel | 63.61 | 62.93 | 63.66 | 83.96 | **-17.47** |
| AlexNet | No Filter | 83.49 | 82.97 | 83.00 | 95.23 | 0.00 |
| AlexNet | Average | 85.93 | 85.59 | 85.44 | 96.08 | +2.62 |
| AlexNet | Gaussian | 84.10 | 83.80 | 83.83 | 95.81 | +0.83 |
| AlexNet | Median | 85.93 | 85.65 | 85.54 | 96.03 | +2.68 |
| AlexNet | Sharpening | 79.51 | 79.40 | 79.63 | 94.45 | -3.57 |
| AlexNet | Sobel | 59.94 | 59.62 | 59.67 | 82.97 | **-23.35** |
| ResNet101 | No Filter | 83.18 | 82.74 | 82.67 | 94.95 | 0.00 |
| ResNet101 | Average | 84.71 | 84.38 | 84.41 | 95.50 | +1.64 |
| ResNet101 | Gaussian | 84.10 | 83.85 | 83.92 | 95.19 | +1.11 |
| ResNet101 | Median | 79.51 | 79.18 | 78.98 | 95.37 | -3.56 |
| ResNet101 | Sharpening | 81.96 | 81.66 | 81.66 | 94.29 | -1.08 |
| ResNet101 | Sobel | 61.16 | 60.56 | 60.98 | 82.97 | **-22.18** |

Averaged over the three models, on validation:

| Filter | mean dMacro-F1 | std across models | min | max |
|---|---|---|---|---|
| Average | **+1.76** | 0.81 | +1.02 | +2.62 |
| Gaussian | **+0.91** | 0.17 | +0.80 | +1.11 |
| Median | -1.66 | 3.77 | -4.09 | +2.68 |
| Sharpening | -0.91 | 2.75 | -3.57 | +1.93 |
| Sobel | **-21.00** | 3.11 | -23.35 | -17.47 |

One caveat I should be honest about. Validation was also used for early stopping and best-epoch
selection, so these numbers are slightly optimistic in absolute terms. But the bias is applied
identically to all 18 runs, so the comparison between filters is still fair.

---

## 5. Per-class F1 (test set)

| Model | Filter | BCC | MEL | NV | PBK |
|---|---|---|---|---|---|
| ResNet50 | No Filter | 86.7 | 11.8 | 73.7 | 69.8 |
| ResNet50 | Average | 87.5 | 31.6 | 70.3 | 70.0 |
| ResNet50 | Gaussian | 84.8 | 22.2 | 73.7 | 66.7 |
| ResNet50 | Median | 88.2 | 11.8 | 72.2 | 63.4 |
| ResNet50 | Sharpening | 93.8 | 22.2 | 71.8 | 76.9 |
| ResNet50 | Sobel | 75.9 | 38.5 | 68.3 | 56.2 |
| AlexNet | No Filter | 75.9 | 38.1 | 66.7 | 78.8 |
| AlexNet | Average | 83.9 | 0.0 | 66.7 | 71.8 |
| AlexNet | Gaussian | 83.9 | 22.2 | 63.8 | 75.0 |
| AlexNet | Median | 75.9 | 22.2 | 68.1 | 76.5 |
| AlexNet | Sharpening | 71.4 | 11.1 | 66.7 | 70.3 |
| AlexNet | Sobel | 80.0 | 0.0 | 66.7 | 58.1 |
| ResNet101 | No Filter | 90.9 | 11.8 | 70.3 | 68.3 |
| ResNet101 | Average | 90.3 | 11.8 | 73.2 | 76.9 |
| ResNet101 | Gaussian | 84.8 | 11.8 | 68.4 | 65.0 |
| ResNet101 | Median | 93.3 | 11.8 | 68.4 | 74.4 |
| ResNet101 | Sharpening | 93.3 | 11.8 | 68.4 | 74.4 |
| ResNet101 | Sobel | 66.7 | 41.7 | 68.4 | 51.3 |

Spread of per-class F1 across the six filter conditions:

| Class | ResNet50 (mean / std) | AlexNet (mean / std) | ResNet101 (mean / std) |
|---|---|---|---|
| BCC | 86.1 / 5.9 | 78.5 / 5.0 | 86.6 / 10.2 |
| MEL | 23.0 / 10.6 | 15.6 / 14.8 | 16.7 / 12.2 |
| NV | 71.7 / **2.1** | 66.4 / **1.4** | 69.5 / **1.9** |
| PBK | 67.2 / 7.0 | 71.7 / 7.4 | 68.4 / 9.5 |

Average change in per-class F1 versus the baseline, averaged over the three models:

| Filter | BCC | MEL | NV | PBK |
|---|---|---|---|---|
| Average | +2.8 | -6.1 | -0.2 | +0.6 |
| Gaussian | 0.0 | -1.8 | -1.6 | -3.4 |
| Median | +1.3 | -5.3 | -0.6 | -0.9 |
| Sharpening | +1.7 | -5.5 | -1.2 | +1.6 |
| Sobel | **-10.3** | +6.2 | -2.4 | **-17.1** |

---

## 6. The melanoma problem

This is the biggest thing in the whole experiment and it is not really about filtering at all, so I
am putting it in its own section.

Confusion matrix summed over all 18 runs (rows are true class, columns are predicted):

| True \\ Pred | BCC | MEL | NV | PBK | Recall |
|---|---|---|---|---|---|
| BCC | **232** | 2 | 7 | 47 | 80.6% |
| MEL | 0 | **33** | 174 | 81 | **11.5%** |
| NV | 0 | 3 | **253** | 32 | 87.8% |
| PBK | 32 | 8 | 11 | **237** | 82.3% |

Melanoma recall is 11.5% across every run. Of 288 melanoma test images seen across the 18 runs, 174
were called nevus and 81 were called pigmented benign keratosis. Not a single melanoma was ever
called basal cell carcinoma. The other three classes all sit above 80% recall.

What makes this strange is that validation accuracy is 80 to 86% while test accuracy is 60 to 72%, a
gap of 13.45 points on average. If the models had simply failed to learn melanoma, validation
macro-F1 could not be sitting at 82. So the models do learn melanoma on the training distribution
and then fail on the melanoma images specifically in the Test folder. That points at a distribution
shift between the Train and Test folders of this dataset for the melanoma class, not at a modelling
failure, and definitely not at anything the filters did.

This also explains why my No-Filter baselines here (67.19% for all three models) came out below the
Lab 01 numbers (78.12 / 75.00 / 75.00) even though I used a stronger recipe. In Lab 01 the loss was
plain cross-entropy and the best epoch was picked on validation accuracy. Here I used class-weighted
cross-entropy with label smoothing and picked the best epoch on validation macro-F1. Both of those
changes deliberately push the model to spend capacity on the hard minority behaviour instead of
riding the easy classes. On a test set where melanoma is shifted, that trade costs raw accuracy. It
is the right choice methodologically and it cost me points on this particular table.

---

## 7. Answers to the questions

### 1. Which three pretrained models performed best in Lab Activity 1?

ResNet50, AlexNet and ResNet101.

| Rank | Model | Lab 01 accuracy | Lab 01 macro-F1 | Lab 01 AUC |
|---|---|---|---|---|
| 1 | ResNet50 | 78.12% | 76.40 | 90.98 |
| 2 | AlexNet | 75.00% | 74.15 | 90.66 |
| 3 | ResNet101 | 75.00% | 70.78 | 92.02 |

AlexNet and ResNet101 tied on accuracy at exactly 75.00%, so I broke the tie on macro-F1, which put
AlexNet second and ResNet101 third. EfficientNet-B0 was next at 71.88% and did not make the cut,
even though it actually had the best AUC of the whole set at 93.72.

I also carried over the classifier-head result from Lab 01. Fine-tuning a plain linear head beat
every classical head on ResNet50 deep features (78.1% linear vs 76.6% logistic regression, 75.0%
SVM and XGBoost, 73.4% random forest, 67.2% KNN), so I used one fixed head, `Dropout(0.3) ->
Linear(features, 4)`, for all 18 runs rather than searching heads again.

### 2. How does filtering affect each of the three models?

The three models react differently, and that is itself the finding.

**ResNet50** benefits from most filtering. On test, sharpening gave it the single best result in the
whole experiment (+4.69 accuracy, +5.70 macro-F1, +1.82 AUC) and the average filter also helped
(+1.56 accuracy, +4.37 macro-F1). Validation agrees: sharpening +1.93, average +1.02, Gaussian
+0.80. Only median (-4.09 on validation) and Sobel (-17.47) hurt it.

**AlexNet** is the one that gets hurt on test by every single filter without exception, from -3.62
(Gaussian) down to -13.67 macro-F1 (Sobel). But on validation AlexNet actually likes the smoothing
filters, with median at +2.68 and average at +2.62 being its two best conditions. That
contradiction is the clearest sign that the 64-image test set is too small to read filter effects
off directly. The one thing both sets agree on for AlexNet is that sharpening hurts it, -9.98 on
test and -3.57 on validation.

**ResNet101** is mildly positive for average (+2.74 test, +1.64 val), roughly neutral for median and
sharpening, mildly negative for Gaussian, and badly hurt by Sobel (-3.30 test, -22.18 val).

One pattern holds for all three, though. AUC drops under almost every filter even when accuracy goes
up: 13 of the 15 filtered runs have a lower AUC than their own baseline. AUC measures the quality of
the ranking of the probabilities, not just the argmax. So filtering is making the models less
calibrated and less confident even when it occasionally nudges the hard decisions in the right
direction.

### 3. Which filter produces the greatest change compared with the unfiltered baseline?

**Sobel**, and it is not close.

On the test set the mean absolute change in macro-F1 puts Sobel first at 5.91 points, but sharpening
is right behind at 5.79 and average at 5.46, which is too tight to call on 64 images.

The validation set settles it completely. Sobel averages **-21.00** macro-F1 across the three models,
with a range of -17.47 to -23.35. The next largest effect of any kind is median at -1.66. So Sobel is
roughly **twelve times** the magnitude of anything else, and it is the only filter that moves the
result by more than a few points.

Sobel is also the only filter that changes what kind of thing the image is, rather than just
re-weighting its frequency content. The other five hand back an RGB photograph with different
sharpness. Sobel hands back a grayscale gradient-magnitude map copied across three channels. Colour
is gone, absolute brightness is gone, and only edge strength survives. AUC confirms the damage:
Sobel drops it to 83.33 / 87.99 / 81.35 on test and to about 83 on validation for all three models,
from baselines in the mid-90s.

### 4. Does the effect of a filter remain consistent across all three models?

Only for two of the five filters. For the rest, no.

Standard deviation of dMacro-F1 across the three models:

| Filter | std (validation) | std (test) | Verdict |
|---|---|---|---|
| Gaussian | **0.17** | 2.68 | Consistent |
| Average | 0.81 | 7.45 | Consistent on validation |
| Sobel | 3.11 | 6.84 | Consistent in sign, varies in size |
| Sharpening | 2.75 | 8.14 | Not consistent |
| Median | 3.77 | 2.94 | Not consistent |

Gaussian is the most consistent filter I tested. All three models gained between +0.80 and +1.11
macro-F1 on validation, a spread of 0.3 points. That is as uniform as it gets.

Sobel is consistent in direction but not magnitude. It hurt every model in every condition on both
sets, which is a genuinely reliable finding, but the size ranged from -17.47 to -23.35 on validation.

Median and sharpening are where consistency breaks down properly, and they break down by flipping
sign, not just by changing size. Median helped AlexNet (+2.68) while hurting both ResNets (-4.09 and
-3.56). Sharpening helped ResNet50 (+1.93) while hurting AlexNet (-3.57) and ResNet101 (-1.08). So
for these two you cannot say "this filter helps" or "this filter hurts" at all without naming the
architecture.

That result makes sense architecturally. AlexNet's first layer is an 11x11 convolution with stride 4,
which is already a low-pass, heavily downsampling operator, so it was never using the fine detail
that a median filter removes, and it gets the denoising benefit for free. Sharpening then gives it
amplified high-frequency content it has no machinery to exploit and that is partly aliased away by
that stride-4 stem, so it is pure added noise. The ResNets start with a 7x7 stride-2 convolution and
keep far more spatial resolution through the residual stages, so they can actually use sharpened
detail and they genuinely lose something when median filtering removes it. Same filter, opposite
sign, because the receptive field of the first layer is different.

So the answer is that the effect is architecture-dependent, and the deciding factor seems to be how
much high-frequency information the model's early layers were using to begin with.

### 5. Does filtering improve or decrease macro-F1 and balanced accuracy?

Mostly it decreases both, but with two real exceptions.

On the test set, 9 of the 15 filtered runs have lower macro-F1 than their baseline and 8 of 15 have
lower balanced accuracy. On validation, 8 of 15 are lower on both. So the majority verdict is
"decreases", but it is closer to a coin flip than I expected before running this.

The two exceptions are consistent enough to state as findings. **Average filtering improves macro-F1
for all three models on validation** (+1.02, +2.62, +1.64, mean +1.76). **Gaussian filtering does the
same** (+0.80, +0.83, +1.11, mean +0.91). These are small, but they are positive for every model on
the larger set, which is more than I can say for median or sharpening.

Sobel decreases both metrics catastrophically and without exception: -21.00 macro-F1 and -20.52
balanced accuracy on average across the three models on validation.

A detail worth noting. Macro-F1 and balanced accuracy move together almost perfectly in my results,
and on the test table balanced accuracy is numerically identical to plain accuracy. That is because
the test set has exactly 16 images per class, and on a perfectly balanced set balanced accuracy
reduces to accuracy by definition. So on this test set those two columns cannot disagree, which is
another reason I leaned on validation.

The more interesting observation is that macro-F1 falls faster than accuracy under filtering. Take
AlexNet with the average filter: accuracy dropped 3.12 points but macro-F1 dropped 9.27, three times
as much. The reason is visible in the per-class table. That run pushed melanoma F1 from 38.1 all the
way to 0.0, meaning it stopped predicting melanoma entirely, while the three easy classes held up and
kept plain accuracy looking respectable. Accuracy hid a total class collapse that macro-F1 caught
immediately. This is exactly why I selected models on validation macro-F1 rather than accuracy.

### 6. Which lesion classes are most affected by filtering?

It depends on whether I measure instability or damage, and the two answers are different.

By raw variance across filter conditions, **melanoma** swings the most: standard deviation of 10.6
(ResNet50), 14.8 (AlexNet) and 12.2 (ResNet101), with its F1 ranging from 0.0 all the way to 41.7.
But I do not think this is really a filtering effect. Melanoma F1 is sitting between 0 and 42 in
every condition including the unfiltered baseline, and melanoma recall is 11.5% overall. The class is
near the floor already, so it is bouncing around on the noise of one or two images flipping, not
responding to the filters in any meaningful way.

By actual damage done, the answer is **pigmented benign keratosis** and **basal cell carcinoma**, and
specifically under Sobel. PBK loses 17.1 F1 points on average under Sobel and BCC loses 10.3. Those
are the two largest genuine per-class drops anywhere in the experiment. In ResNet101's Sobel run PBK
falls from 68.3 to 51.3 and BCC from 90.9 to 66.7.

That fits what those two classes look like. BCC is normally the easiest class here, sitting around 86
to 91 F1, because it is identified by pearly translucency and arborising telangiectasia, which are
colour and texture cues. Sobel throws away every bit of colour and leaves only edge strength, so BCC
loses its main discriminator and collapses into PBK. The confusion matrix supports this: BCC was
misread as PBK 47 times across all runs, and PBK as BCC 32 times, and those two directions are the
largest non-melanoma confusions in the whole experiment.

**Nevus is by far the most robust class.** Its per-class F1 standard deviation across all six
conditions is 2.1, 1.4 and 1.9 for the three models, which is five to ten times more stable than any
other class. It never drops below 63.8 in any run. Nevus is defined largely by regular, symmetric,
large-scale structure, and that survives a 3x3 blur, a sharpening kernel and even a Sobel transform.
Whatever the network uses to recognise a nevus is apparently low-frequency and shape-based, which is
precisely why filtering cannot touch it.

Melanoma also does something odd that is worth flagging: it is the only class whose F1 goes *up*
under Sobel, by +6.2 on average, reaching 38.5 for ResNet50 and 41.7 for ResNet101. I do not read
this as Sobel helping. Under Sobel the models get much less confident overall, start spreading their
predictions around instead of defaulting to nevus, and melanoma picks up a few correct predictions by
accident. It is a degenerate side effect of a broken model, not a genuine gain, and the fact that
overall macro-F1 still dropped in those same runs confirms it.

### 7. Why might smoothing remove useful lesion texture or morphological information?

Because average, Gaussian and median filters are all low-pass operators, and in dermoscopy a large
part of the diagnostic signal lives in exactly the high frequencies they remove.

The features a dermatologist actually uses, the pigment network, dots and globules, streaks, blue-white
veil, and the sharpness or abruptness of the lesion border, are all fine-scale, high-frequency
structure. The ABCD rule and the 7-point checklist are essentially formalised statements about
high-frequency structure and colour variegation. A 3x3 box kernel replaces each pixel with the mean
of its neighbourhood, which is a convolution with a kernel whose frequency response falls off sharply
above a cutoff. Everything above that cutoff is attenuated, and attenuated information is gone
permanently. No downstream layer can recover it, because the data processing inequality says
post-processing cannot increase the information a signal carries about the label.

What is left after smoothing is mostly colour and gross shape, which separates BCC from nevus fine
but is much weaker for telling melanoma from a benign nevus, since that distinction rests on
irregularity of the network rather than on overall colour or outline.

My results do fit this, but with a nuance I did not predict. Smoothing did not hurt overall accuracy
much and average and Gaussian actually helped slightly on validation. But look at what it did per
class: average filtering cost melanoma 6.1 F1 points and median cost it 5.3, while BCC *gained* 2.8
and 1.3. So smoothing did exactly what the theory predicts, it removed the fine texture that melanoma
depends on, and it simultaneously removed hair artefacts, sensor noise and JPEG blocking that were
hurting the easier classes. The two effects partly cancelled in the aggregate number and only the
per-class breakdown revealed what was going on.

Median is the interesting case among the three. It should in principle be the gentlest, since it is
edge-preserving and removes impulse noise rather than averaging across boundaries, so it should
remove hair and speckle while keeping the lesion border crisp. On AlexNet it behaved exactly like
that, giving its joint-best validation result at +2.68. On both ResNets it was the worst non-Sobel
filter, at -4.09 and -3.56. So edge preservation helped the model with the coarse stem and hurt the
models that were using the detail it removed inside the lesion, which is the same architecture
argument as in question 4.

### 8. Why might sharpening or edge detection help or hurt classification?

These are two quite different operations and my results separate them very clearly.

**Sharpening can help, and for ResNet50 it did.** The kernel `[[0,-1,0],[-1,5,-1],[0,-1,0]]` is an
identity plus a Laplacian, so it boosts high frequencies while keeping the underlying image. It makes
faint pigment networks more salient and increases local contrast at the lesion border, which is
genuinely useful extra visibility for structures the network is already looking for, and it acts a
bit like a fixed contrast augmentation. ResNet50 got its best result of the whole experiment this
way: 71.88% accuracy, 66.17 macro-F1, 94.76 AUC, with BCC F1 rising to 93.8 and PBK to 76.9.

**It can also hurt, and for AlexNet it did badly**, at -9.98 macro-F1 on test and -3.57 on validation.
Sharpening is indiscriminate. It amplifies sensor noise, hair, bubbles and JPEG compression artefacts
by the same factor as the real structure, and it pushes the image statistics away from the ImageNet
distribution the pretrained weights were calibrated on. If a model was not exploiting those high
frequencies anyway, all it receives is amplified noise. AlexNet's stride-4 11x11 stem means the
amplified detail is largely aliased away before it can be used, so AlexNet paid the noise cost
without collecting the signal benefit.

**Sobel is a different thing entirely and it hurt every time**, by -21.00 macro-F1 on average across
the three models. Sharpening returns a photograph. Sobel returns a gradient-magnitude map, which
throws away all colour and all absolute intensity and keeps only edge strength. Three things break at
once. First, colour is a first-order diagnostic cue in dermoscopy, since the blue-white veil, the
brown of a pigment network and the pearly pink of a BCC are all colour facts, and all of them are
gone. Second, a CNN's first-layer filters after ImageNet pretraining are largely colour-opponent
detectors and oriented Gabor-like edge detectors, and feeding them an identical value in all three
channels means every colour-opponent unit outputs approximately zero, so a large fraction of the
pretrained first layer is dead on arrival. Third, running an edge detector before a network whose job
is to learn edge detectors is redundant at best, because it hard-codes one fixed 3x3 kernel in place
of the thousands the network would have learned.

AUC shows the damage most clearly. Sobel dropped it to 83.33, 87.99 and 81.35 on test, and to roughly
83 for all three models on validation, from baselines of 92.94, 90.20 and 93.59. The models are not
just getting the argmax wrong, their whole probability ranking has degraded.

### 9. What is the difference between convolution and correlation?

Cross-correlation slides the kernel over the image and takes a dot product directly:

$$(f \star g)(x,y) = \sum_{i}\sum_{j} f(x+i,\; y+j)\, g(i,j)$$

Convolution flips the kernel 180 degrees in both axes first:

$$(f * g)(x,y) = \sum_{i}\sum_{j} f(x-i,\; y-j)\, g(i,j)$$

The two give identical results whenever the kernel is symmetric under a 180-degree rotation. That
covers the box kernel, the Gaussian, the Laplacian and my sharpening kernel
`[[0,-1,0],[-1,5,-1],[0,-1,0]]`, which is why the distinction never showed up in most of this lab.
They differ for asymmetric kernels, and the Sobel kernels are the example right here in my own
experiment:

$$G_x = \begin{bmatrix} -1 & 0 & +1 \\ -2 & 0 & +2 \\ -1 & 0 & +1 \end{bmatrix}$$

Flipping that gives the negated kernel, so convolving with Sobel-x gives the negative of
correlating with it. My results are unaffected because I take the gradient magnitude
$\sqrt{G_x^2 + G_y^2}$, and squaring destroys the sign. If I had used the signed gradient the
distinction would have mattered.

The properties differ too. Convolution is commutative and associative and corresponds to the response
of a linear shift-invariant system, which is what makes the convolution theorem work
($f * g \leftrightarrow F \cdot G$ in the Fourier domain) and what lets you cascade two blurs into
one. Correlation is neither commutative nor associative, and it is the natural operation for template
matching, since correlating an image with a template peaks where the template appears.

The practical footnote is that what deep learning frameworks call convolution is actually
cross-correlation. PyTorch's `nn.Conv2d` does not flip the kernel. It makes no difference, because the
weights are learned from scratch, so the network simply learns the flipped version of whatever kernel
it would otherwise have learned and the two formulations are equivalent in expressive power. It only
matters if you are hand-loading a designed kernel into a conv layer and expecting textbook behaviour.

### 10. Based on your results, explain the relationship between classical image processing and deep-learning-based feature extraction.

My results point to one conclusion: a CNN does not replace classical image processing, it learns it,
and then keeps going.

The first convolutional layer of a pretrained network, when you visualise it, contains things that
look strikingly like the filters in this lab: oriented edge detectors that are close to Gabor
functions, centre-surround blob detectors that resemble a Laplacian of Gaussian, and colour-opponent
kernels. Decades of hand-designing edge and blur operators produced roughly the same first-layer
filter bank that gradient descent finds on its own from data. The difference is that the network then
stacks dozens more layers on top, each composing the previous ones into progressively more abstract
and task-specific features, which is the part classical pipelines never had.

That explains the shape of my results directly. **Applying a fixed classical filter beforehand is
largely redundant**, because if that filter is useful the network can learn it, and if it is not, the
network can learn not to use it. This is why four of my five filters moved macro-F1 by less than 2
points on average on validation, at +1.76, +0.91, -1.66 and -0.91. I paid six times the training cost
to hand the network an operation it was perfectly capable of discovering by itself, and got back
almost nothing.

**But it becomes actively harmful when the filter is destructive and irreversible.** This is the Sobel
result and the -21.00 macro-F1. The argument is information-theoretic, not empirical. Pre-processing
sits before the network in the pipeline, so by the data processing inequality it can never increase
the mutual information between the input and the label. It can only preserve or destroy. A learned
layer inside the network is different because it is optimised jointly with the objective and can be
adjusted or bypassed; a fixed pre-processing filter cannot be undone by anything downstream. Sobel
deletes colour and absolute intensity, so no amount of depth afterwards can bring them back. This is
the sharpest lesson in the lab for me: the cost of a bad pre-processing choice is unbounded, while the
benefit of a good one is capped at whatever the network could not have learned itself.

There is a second, subtler cost that my ResNet50 and AlexNet split exposed. Pre-processing moves the
input off the distribution the pretrained weights were calibrated on. ImageNet pretraining is only
valuable because the statistics of the new images resemble the old ones. The further a filter pushes
the input away, the more of that transfer value is thrown away, and Sobel pushes it furthest of all,
which is why every colour-opponent unit in the first layer receives three identical channels and
effectively switches off.

So when is classical filtering actually worth doing in a deep pipeline? My per-class numbers suggest
the answer. It earns its place when it removes a known **nuisance** factor rather than transforming
the signal itself. Average and Gaussian filtering were the only two filters to help all three models
on validation, and the per-class breakdown shows why: they cost melanoma 6.1 and 1.8 F1 points, which
is the genuine texture loss, but they helped BCC and left the rest roughly flat, which is the noise
and hair removal. The net was positive because for this dataset the nuisance removed slightly
outweighed the signal lost. That is the correct model for when to use classical pre-processing.
Hair removal, specular highlight suppression, illumination and shading correction and colour
constancy are all cases where you know something in the image is not signal, and removing it is a
genuine gain the network cannot easily achieve for itself. Sobel is the opposite case, throwing away
the signal along with the noise, and the 21-point drop is the price.

The last thing my results say is about evaluation rather than filtering. The filter effects I was
trying to measure are mostly in the range of 1 to 4 points, while the melanoma distribution shift
between the Train and Test folders costs 13.45 points, and a single test image is worth 1.56 points.
The thing I was measuring was smaller than the noise floor of the thing I was measuring it with. In
practice the data and the evaluation protocol mattered far more than any choice of pre-processing
filter, and if I had only looked at the 64-image test table I would have concluded that AlexNet hates
all filtering, which the 409-image validation set shows is not true.

---

## 8. Limitations

I want to be straight about what this experiment can and cannot support.

The test set is 64 images, 16 per class. One image is 1.5625 percentage points. Almost every
non-Sobel difference in my main table is a one or two image difference and is not statistically
meaningful. I have reported the validation results (409 images) alongside throughout for this reason,
and where the two disagree I trusted validation.

Validation was used for early stopping and best-epoch selection, so validation numbers are
optimistically biased in absolute terms. The bias is identical across all 18 runs so the between-filter
comparison stands, but the absolute validation figures should not be quoted as held-out performance.

Each configuration was trained once with a fixed seed. A proper version of this study would repeat
the grid with three or more seeds and report mean and standard deviation, which would turn most of
these comparisons from suggestive into defensible. That is the single most valuable extension and it
is the one I would do next.

The melanoma distribution shift between the Train and Test folders is a property of this dataset, not
of my method, and it dominates the test results. Any conclusion drawn from melanoma's test numbers
specifically should be treated with suspicion.

---

## 9. Summary

- The three best Lab 01 models were ResNet50 (78.12%), AlexNet (75.00%) and ResNet101 (75.00%).
- Best run in this lab: ResNet50 with sharpening, 71.88% accuracy, 66.17 macro-F1, 94.76 AUC.
- Sobel produces by far the greatest change, -21.00 macro-F1 on average, roughly twelve times any
  other filter, and it hurt all three models in every condition.
- Average (+1.76) and Gaussian (+0.91) were the only filters to improve macro-F1 for all three models.
- Filter effects are architecture-dependent. Median and sharpening flip sign between AlexNet and the
  ResNets, which tracks how much high-frequency detail each model's first layer was using.
- Nevus is the most robust class (per-class F1 std of 1.4 to 2.1). PBK and BCC take the most real
  damage, mainly from Sobel, at -17.1 and -10.3 F1 points.
- Melanoma recall is 11.5% across all runs, with 174 of 288 melanoma images called nevus. This is a
  Train/Test distribution shift in the dataset, not a filtering effect.
- Overall: classical filtering is mostly redundant in front of a CNN, occasionally useful when it
  removes a known nuisance, and badly damaging when it destroys information irreversibly.
