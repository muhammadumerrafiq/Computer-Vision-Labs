# Computer Vision Lab 01

## Skin Cancer Image Classification using the ISIC Dataset

### 1. Introduction

In this lab, I worked on skin cancer image classification using the ISIC dataset. I downloaded the dataset from Kaggle. The original dataset contains 9 classes, but for this lab I selected only four classes:

- Basal cell carcinoma
- Melanoma
- Nevus
- Pigmented benign keratosis

I selected these four classes because they had a reasonable number of training images and each selected class had exactly 16 test images. This made the experiment more manageable.

### 2. Dataset

The selected dataset contained 1,633 training images and 64 test images.

| Class | Training Images | Test Images |
| --- | ---: | ---: |
| Basal cell carcinoma | 376 | 16 |
| Melanoma | 438 | 16 |
| Nevus | 357 | 16 |
| Pigmented benign keratosis | 462 | 16 |
| **Total** | **1633** | **64** |

I split the training data into training and validation data using a fixed random seed of 42.

- **Training:** 1306 images
- **Validation:** 327 images
- **Test:** 64 images

The test set was kept separate and was used only for the final evaluation.

### 3. Preprocessing

All images were resized to **224 × 224**.

For training, I used the following preprocessing and augmentation:

- Resize
- Random horizontal flip
- Random vertical flip
- Random rotation up to 15 degrees
- ToTensor
- ImageNet normalization

For validation and testing, I used:

- Resize
- ToTensor
- ImageNet normalization

No random augmentation was used for validation or testing.

### 4. Deep Learning Models

I used ImageNet pretrained versions of the following models:

- AlexNet
- VGG16
- VGG19
- ResNet18
- ResNet50
- ResNet101
- DenseNet121
- EfficientNet-B0

I changed the final classification layer of each model so that it could classify the four selected classes.

#### Training Settings

| Setting | Value |
| --- | --- |
| Image size | 224 × 224 |
| Batch size | 32 |
| Optimizer | Adam |
| Learning rate | 0.0001 |
| Loss | Cross Entropy Loss |
| Maximum epochs | 10 |
| Early stopping patience | 3 |
| Model selection | Best validation accuracy |

### 5. Table 1 — Deep Learning Results

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | AUC (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| AlexNet | 75.00 | 77.61 | 75.00 | 74.15 | 90.66 |
| VGG16 | 70.31 | 72.63 | 70.31 | 66.12 | 90.43 |
| VGG19 | 68.75 | 71.47 | 68.75 | 65.11 | 92.35 |
| ResNet18 | 70.31 | 74.38 | 70.31 | 66.02 | 93.20 |
| ResNet50 | 78.12 | 80.74 | 78.12 | 76.40 | 90.98 |
| ResNet101 | 75.00 | 83.17 | 75.00 | 70.78 | 92.02 |
| DenseNet121 | 70.31 | 74.42 | 70.31 | 66.51 | 90.79 |
| EfficientNet-B0 | 71.88 | 80.64 | 71.88 | 67.92 | 93.72 |

### 6. Best Deep Learning Model

I selected **ResNet50** as the best overall model because it achieved the highest accuracy of **78.12%** and the highest F1-score of **76.40%** among the compared models.

However, ResNet50 was not the best in every metric:

- **ResNet101** had the highest precision: **83.17%**
- **EfficientNet-B0** had the highest AUC: **93.72%**

Based on the overall results, I selected ResNet50 for the deep feature experiment.

### 7. Deep Feature Experiment

After selecting ResNet50, I used it as a feature extractor. ResNet50 generated **2048 deep features per image**.

I then used these deep features with traditional machine learning classifiers:

- Logistic Regression
- Decision Tree
- Random Forest
- KNN
- Linear SVM
- RBF-SVM
- XGBoost

For this experiment, I combined the training and validation data for fitting the traditional classifiers. The test set remained separate for final evaluation.

### 8. Table 2 — Traditional Classifier Results

| Feature Extractor | Classifier | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | AUC (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Deep Features (ResNet50) | Logistic Regression | 76.56 | 80.57 | 76.56 | 74.42 | 90.98 |
| Deep Features (ResNet50) | Decision Tree | 68.75 | 68.59 | 68.75 | 67.07 | 79.17 |
| Deep Features (ResNet50) | Random Forest | 73.44 | 77.20 | 73.44 | 70.55 | 92.22 |
| Deep Features (ResNet50) | KNN | 67.19 | 71.91 | 67.19 | 63.71 | 89.42 |
| Deep Features (ResNet50) | Linear SVM | 75.00 | 77.18 | 75.00 | 73.13 | 91.60 |
| Deep Features (ResNet50) | RBF-SVM | 75.00 | 79.72 | 75.00 | 72.36 | 94.11 |
| Deep Features (ResNet50) | XGBoost | 75.00 | 79.72 | 75.00 | 72.36 | 90.76 |

### 9. Traditional Classifier Results

From the results, I found that:

- **Logistic Regression** had the highest accuracy: **76.56%**
- **RBF-SVM** had the highest AUC: **94.11%**
- The end-to-end **ResNet50** was still slightly better in accuracy, with **78.12%**

This shows that the deep features extracted by ResNet50 can also be used with traditional machine learning classifiers, but in this experiment the complete ResNet50 model gave better accuracy.

### 10. Computational Comparison

I also compared the deep learning models based on:

- Number of parameters
- Model size
- FLOPs
- Inference time
- Accuracy

The experiments were run using a **Tesla T4 GPU**.

The model size is an estimated size based on float32 parameter storage. Inference time can depend on the hardware and the environment used for testing.

### 11. Table 3 — Computational Results

| Model | Parameters (M) | Model Size (MB) | FLOPs (G) | Inference Time (ms) | Accuracy (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| AlexNet | 57.02 | 217.51 | 0.71 | 3.75 | 75.00 |
| VGG16 | 134.28 | 512.23 | 15.47 | 10.58 | 70.31 |
| VGG19 | 139.59 | 532.48 | 19.63 | 12.82 | 68.75 |
| ResNet18 | 11.18 | 42.64 | 1.82 | 8.56 | 70.31 |
| ResNet50 | 23.52 | 89.71 | 4.13 | 13.84 | 78.12 |
| DenseNet121 | 6.96 | 26.54 | 2.90 | 17.78 | 70.31 |
| EfficientNet-B0 | 4.01 | 15.31 | 0.41 | 12.06 | 71.88 |

### 12. Computational Comparison Results

The main observations from the computational comparison were:

- **AlexNet** had the fastest inference time: **3.75 ms**
- **EfficientNet-B0** was the smallest model with **4.01M parameters**, **15.31 MB**, and **0.41G FLOPs**
- **VGG19** was the largest model with **139.59M parameters** and **19.63G FLOPs**
- **ResNet50** achieved the highest accuracy: **78.12%**

This comparison shows that model size and speed are not the only important factors. A smaller or faster model does not always give the highest accuracy.

### 13. Conclusion

In this lab, I learned how pretrained CNN models can be used for image classification. I used four classes from the ISIC skin cancer dataset and compared different pretrained models.

I found that different models gave different results. **ResNet50 performed best overall in my experiment**, with an accuracy of **78.12%** and an F1-score of **76.40%**. I also used ResNet50 as a feature extractor and tested its deep features with traditional machine learning classifiers.

Overall, this lab helped me understand how transfer learning, deep feature extraction, and traditional machine learning classifiers can be used together for image classification.
