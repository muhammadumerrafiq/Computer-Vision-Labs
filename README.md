# Computer Vision Labs & Assignments

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8.svg)](https://opencv.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626.svg)](https://jupyter.org/)

A comprehensive collection of practical laboratory experiments, benchmarks, and assignments for **Introduction to Computer Vision (7th Semester)**. This repository explores classical image processing pipelines, spatial filtering, edge detection, mathematical morphology, deep Convolutional Neural Networks (CNNs) benchmarked on dermoscopic medical imaging datasets (ISIC and HAM10000), feature-descriptor-based industrial defect detection pipelines (HOG + Kernel SVM) with real-time web deployment, scale-invariant feature transform (SIFT) landmark recognition with RANSAC homography localization and open-set rejection, and evolutionary feature selection using Genetic Algorithms (GA-HOG) for medical image classification (PneumoniaMNIST).

---

## 📑 Table of Contents

- [Repository Structure](#-repository-structure)
- [Lab Overview & Modules](#-lab-overview--modules)
  - [Lab 01: Deep CNN Architecture Benchmarking](#lab-01-deep-cnn-architecture-benchmarking)
  - [Lab 02: Spatial Filtering & Feature Enhancement](#lab-02-spatial-filtering--feature-enhancement)
  - [Lab 03: Edge Detection, Noise Sensitivity & Edge Representations](#lab-03-edge-detection-noise-sensitivity--edge-representations)
  - [Lab Assignment 01 (Lab 04): Skin Lesion Boundary Detection & Morphometry](#lab-assignment-01-lab-04-skin-lesion-boundary-detection--morphometry)
  - [Lab 05: Industrial Defect Detection using HOG & Kernel SVM](#lab-05-industrial-defect-detection-using-hog--kernel-svm)
  - [Lab 06: Landmark Recognition & Localization using SIFT & RANSAC](#lab-06-landmark-recognition--localization-using-sift--ransac)
  - [Lab 07: Evolutionary Feature Selection for Medical Image Classification (GA-HOG)](#lab-07-evolutionary-feature-selection-for-medical-image-classification-ga-hog)
- [Key Empirical Results](#-key-empirical-results)
  - [Filter Benchmark Summary (Lab 02)](#filter-benchmark-summary-lab-02--test-set)
  - [Classifier Benchmark Summary (Lab 05)](#classifier-benchmark-summary-lab-05--industrial-defect-detection)
  - [SIFT Landmark Recognition Summary (Lab 06)](#sift-landmark-recognition-summary-lab-06--test-perturbations--open-set-queries)
  - [GA-HOG Feature Selection Benchmark (Lab 07)](#ga-hog-feature-selection-benchmark-lab-07--pneumoniamnist)
- [Tech Stack & Dependencies](#-tech-stack--dependencies)
- [Getting Started](#-getting-started)
- [Author & Academic Info](#-author--academic-info)
- [License](#-license)

---

## 📂 Repository Structure

```text
Computer-Vision-Labs/
├── CV Lab 01/
│   ├── CV_LAB_01.ipynb             # Deep CNN architectures benchmarking
│   └── FA23-BAI-013.docx           # Lab report and documentation
│
├── CV Lab 02/
│   ├── CV_LAB_02.ipynb             # Image filtering impact on CNN classification
│   ├── Lab02.md                    # Detailed results, metrics, and discussion
│   └── lab02_results/              # Metric tables, confusion matrices, filter response curves
│
├── CV Lab 03/
│   ├── CV_LAB_03.ipynb             # Comprehensive edge detection and noise robustness
│   ├── Lab-03.md                   # Metric tables and theoretical discussion
│   ├── Report.pdf                  # Complete formal academic lab report
│   └── lab3_results/               # Plots, confusion matrices, and benchmark tables
│
├── Lab Assignment 01_CV Lab 04/
│   ├── Skin_Lesion_Boundary_Detection.ipynb   # Boundary extraction & morphology
│   ├── Skin_Lesion_Boundary_Detection.md      # Quantitative morphometrics & answers
│   └── skin_lesion_results/                   # Visualizations and area/perimeter logs
│
├── CV Lab 05/
│   ├── HOG_Defect_Detection.ipynb  # End-to-end HOG pipeline, sweeps & classifier benchmarks
│   ├── Lab_Report.docx             # Formal academic lab report and analysis
│   ├── Outputs/                    # Heatmaps, confusion matrices, and CSV evaluation benchmarks
│   └── Bonus Task/                 # Real-time HOG industrial quality inspection prototype (Flask web app)
│       ├── app.py                  # Flask inspection server with webcam, video & image routes
│       ├── predictor.py            # Sliding window HOG inference & bounding box visualizer
│       ├── hog_utils.py            # HOG feature descriptor extraction & visualization helpers
│       ├── hog_defect_model.joblib # Pre-trained Kernel SVM (RBF) industrial defect model
│       ├── templates/ & static/    # Modern dark-mode UI with real-time inspection indicators
│       ├── sample_images/          # Normal & defective steel samples and conveyor belt test video
│       └── requirements.txt        # Bonus app dependencies
│
├── CV Lab 06/
│   ├── SIFT_Landmark_Recognition.ipynb # End-to-end SIFT landmark recognition & homography pipeline
│   ├── Lab_Report.docx                 # Formal academic lab report and analysis
│   ├── Lab_Report.md                   # Complete Markdown report, metrics, and screenshots
│   ├── landmark_database/              # Reference gallery across 6 world landmarks (8 views each)
│   └── submission_files/               # Visualizations, test condition outputs, and experimental CSV
│
├── CV Lab 07/
│   ├── Lab07.ipynb                     # Executed GA-HOG feature selection & SVM training pipeline
│   ├── Lab 07.docx                     # Formal academic lab report documentation
│   ├── Lab07_Report.md                 # Complete Markdown report, metrics, and discussion
│   └── lab07_outputs/                  # Confusion matrices, feature distributions, GA fitness curves & CSVs
│
├── LICENSE                             # MIT License
└── README.md                           # Repository documentation
```

---

## 🔬 Lab Overview & Modules

### Lab 01: Deep CNN Architecture Benchmarking
- **Objective:** Systematically evaluate and compare deep neural network vision backbones on dermoscopic skin lesion classification.
- **Architectures Evaluated:**
  - Classic Deep Models: **AlexNet**, **VGG16**, **VGG19**
  - Residual Networks: **ResNet18**, **ResNet50**, **ResNet101**
  - Dense & Efficient Networks: **DenseNet121**, **EfficientNet-B0**
- **Analysis:** Feature extraction, downstream classifiers (Softmax, SVM, Random Forest) on deep embeddings, FLOPs/parameter counts, inference latency, and convergence characteristics.

---

### Lab 02: Spatial Filtering & Feature Enhancement
- **Objective:** Quantify the exact impact of spatial filtering and preprocessing transformations on downstream deep learning classification performance.
- **Dataset:** 4-class ISIC subset (Basal Cell Carcinoma, Melanoma, Nevus, Pigmented Benign Keratosis).
- **Experimental Design:** 18 controlled training runs (3 backbone models $\times$ 6 filter variations) with identical splits, random seeds, and hyperparameters:
  - **No Filter (Baseline)**
  - **Average Blur** ($3\times3$ box filter)
  - **Gaussian Blur** ($5\times5$, $\sigma = 1.0$)
  - **Median Filter** ($3\times3$)
  - **Custom Sharpening Kernel** ($\begin{bmatrix} 0 & -1 & 0 \\ -1 & 5 & -1 \\ 0 & -1 & 0 \end{bmatrix}$)
  - **Sobel Gradient Magnitude** ($3\times3$ Sobel $x + y$)
- **Key Takeaway:** Custom high-pass sharpening on ResNet50 yielded top classification performance (**71.88% Accuracy**, **66.17% Macro-F1**, **94.76% AUC**), enhancing micro-texture and pigment networks.

---

### Lab 03: Edge Detection, Noise Sensitivity & Edge Representations
- **Objective:** Evaluate classical first-order, second-order, and multi-stage edge operators under clean and corrupted environments, and examine their utility as direct input representations for neural networks.
- **Algorithms Analyzed:**
  - **Sobel** & **Prewitt** (1st-order gradient operators with directional smoothing)
  - **Laplacian** & **Laplacian of Gaussian (LoG)** (2nd-order zero-crossing isotropic operators)
  - **Canny Edge Detector** (Gaussian smoothing, gradient magnitude/direction, Non-Maximum Suppression, Hysteresis thresholding)
- **Controlled Noise Experiments:**
  - Gaussian Noise ($\sigma = 25$) vs. Salt-and-Pepper Noise ($p = 0.05$).
  - Preprocessing restoration: Median filtering recovers up to **+0.46 F1** against impulse noise; Gaussian filtering mitigates gradient noise variance.
- **Representation Study:** Comparing classification models trained purely on raw RGB vs. edge maps vs. filtered inputs.

---

### Lab Assignment 01 (Lab 04): Skin Lesion Boundary Detection & Morphometry
- **Objective:** Design an end-to-end computer vision pipeline for automated lesion boundary extraction, artifact handling, and quantitative morphometric feature calculation.
- **Dataset:** HAM10000 (`kmader/skin-cancer-mnist-ham10000`) across all 7 diagnostic categories (*nv, mel, bkl, bcc, akiec, vasc, df*).
- **Key Techniques:**
  - Spatial pre-filtering (Gaussian, Median, Box blur)
  - Gradient vs. Multi-stage boundary extraction (Sobel vs. Canny with threshold pairs: 50/100, 100/200, 150/250)
  - Mathematical morphology (dilation, closing to bridge fragmented boundaries)
  - Contour extraction and shape morphometrics: **Lesion Area**, **Perimeter**, and **Circularity / Compactness**
- **Analysis:** In-depth evaluation of dermoscopy artifacts (hair occlusions, faint boundaries, low contrast) and clinical trade-offs between operator sensitivity and specificity.

---

### Lab 05: Industrial Defect Detection using HOG & Kernel SVM
- **Objective:** Detect manufacturing surface defects on steel sheets (hot-rolled plates) using **Histogram of Oriented Gradients (HOG)** feature extraction and machine learning classifiers.
- **Key Components & Experiments:**
  - **HOG Parameter Optimization:** Systematic parameter sweep evaluating cell sizes ($4\times4$, $8\times8$, $16\times16$), orientations ($6, 9, 12$), and block sizes ($1\times1$, $2\times2$, $3\times3$) with L2-Hys normalization. The optimal configuration ($8\times8$ cells, 9 bins, $2\times2$ blocks, $128\times128$ window) yields an 8,100-dimensional descriptor.
  - **Classifier Benchmarking:** Comparative analysis of **Kernel SVM (RBF)**, **Linear SVM**, **Random Forest**, **k-NN**, and **XGBoost**. Kernel SVM achieved top performance with **77.74% Accuracy** and **81.95% F1-Score** in binary defect classification, and **83.89% Macro-Accuracy** in multi-class defect categorization across 6 classes (*Crazing, Inclusion, Patches, Pitted Surface, Rolled-in Scale, Scratches*).
  - **Sliding-Window Industrial Decision Module:** Full-image surface scanning with $64\times64$ windows (stride 32px), local patch classification, and thresholded spatial anomaly aggregation to automate **ACCEPT (PASS)** or **REJECT (DEFECTIVE)** decisions.
  - **Robustness & Perturbation Analysis:** Stress testing under synthetic noise, contrast shifts, and spatial rotations to evaluate descriptor stability under production floor lighting variations.
- **Bonus Challenge Prototype (Real-Time Deployment):**
  - Interactive Flask web application simulating industrial quality control.
  - **Single & Batch Image Inspection:** Upload or select sample steel plates with automatic bounding box localization around detected defect patches.
  - **Video Stream Inspection:** Frame-by-frame analysis of simulated conveyor belt footage with defect rate counters and rejection logs.
  - **Live Webcam Inspection:** Real-time $200\times200$ ROI video stream extraction displaying dynamic PASS/DEFECTIVE status and confidence scoring.

---

### Lab 06: Landmark Recognition & Localization using SIFT & RANSAC
- **Objective:** Construct a feature-based landmark recognition and geometric localization system using **Scale-Invariant Feature Transform (SIFT)** and **RANSAC Homography**, featuring strict open-set rejection for unseen queries.
- **Reference Gallery & Dataset:**
  - Curated database of 6 global landmarks from Kaggle's Wonders of the World dataset: *Taj Mahal, Great Wall of China, Statue of Liberty, Roman Colosseum, Pyramids of Giza, Christ the Redeemer*.
  - Reference gallery composed of 8 distinct viewpoints per landmark (48 reference images total), multithreaded SIFT feature extraction (up to 2,000 features per image), downscaled to max dimension 800 px.
- **Matching & Localization Pipeline:**
  - **Feature Extraction:** Grayscale conversion and keypoint/128-dimensional descriptor extraction via `cv2.SIFT_create(nfeatures=2000)`.
  - **Descriptor Matching:** Brute-force Euclidean distance matching (`cv2.BFMatcher(cv2.NORM_L2)`) with $k=2$ nearest neighbors.
  - **Ambiguity Filtering:** Lowe's ratio test ($d_1 / d_2 < 0.75$) to discard non-distinctive repetitive matches.
  - **Geometric Verification & Localization:** Multi-point homography estimation (`cv2.findHomography`) with RANSAC reprojection error threshold ($\le 5.0$ px); projected bounding polygon validation ensuring convexity and plausible scene area ($0.01 \times A_{\text{query}} < A_{\text{poly}} < 3.0 \times A_{\text{query}}$).
  - **Dual-Threshold Decision Rule:** Multi-level acceptance criterion requiring $\ge 20$ reliable matches, $\ge 12$ RANSAC inliers, convex geometry, and a distinct inlier margin ($\ge 1.3\times$ over the runner-up landmark).
- **Stress Testing & Perturbation Analysis:**
  - Evaluated across 32 query configurations: unaltered held-out images, 45° boundary-preserving rotations, 50% spatial downscaling, low illumination ($\alpha = 0.35$), viewpoint disparities, and negative/unseen landmarks (*Stonehenge*).
  - **Key Finding:** SIFT proved robust to rotation (13.8 avg matches, 7.8 inliers), but sensitive to extreme resolution reduction and low illumination. The multi-stage inlier and margin gating completely eliminated false recognitions (0% false acceptance on open-set queries).

---

### Lab 07: Evolutionary Feature Selection for Medical Image Classification (GA-HOG)
- **Objective:** Modify the classical Histogram of Oriented Gradients (HOG) pipeline by incorporating a Genetic Algorithm (GA) wrapper for feature selection to reduce descriptor dimensionality and boost classification on medical imaging.
- **Dataset:** **PneumoniaMNIST** from MedMNIST v2 (pediatric chest X-ray binary classification: *Normal* vs. *Pneumonia*; 5,030 total images with fixed splits: 3,882 train / 524 validation / 624 test).
- **Pipeline & Architecture:**
  - **Feature Extraction:** Standard HOG descriptor ($128\times128$ resolution, 9 orientations, $8\times8$ pixel cells, $2\times2$ cell blocks, L2-Hys normalization) yielding **8,100 features** per image.
  - **GA Optimization Wrapper:** Searches a binary feature selection mask over 8,100 dimensions. Fitness balances validation macro-F1 against feature sparsity: $\text{Fitness} = \text{Macro-F1}_{\text{val}} - 0.05 \times \frac{|\text{selected features}|}{8,100}$.
  - **Genetic Operators:** Tournament selection (size 3), uniform crossover ($p = 0.9$), bit-flip mutation, and elitism (2 best individuals preserved) evolved over 15 generations.
  - **GPU-Accelerated Kernel SVM:** Fast RBF kernel Gram matrices evaluated on GPU (PyTorch) with scikit-learn precomputed `SVC(kernel="precomputed")`, exploring three GA configurations (GA-A, GA-B, GA-C).
- **Key Findings:**
  - The optimal configuration (**GA-A**) reduced feature dimensionality by **73.7%** (retaining only **2,129** out of 8,100 features).
  - Outperformed standard baseline HOG on the unseen test set across all primary metrics: **Macro-F1 +4.58%** (77.36% $\to$ **81.95%**), **Accuracy +3.21%** (81.57% $\to$ **84.78%**), and **Normal class recall +8.98%** (51.28% $\to$ **60.26%**).

---

## 📊 Key Empirical Results

### Filter Benchmark Summary (Lab 02 — Test Set)

| Backbone Architecture | Filter Applied | Accuracy (%) | Precision (%) | Recall (%) | Macro-F1 (%) | AUC (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **ResNet50** | **Sharpening** | **71.88** | **79.96** | **71.88** | **66.17** | **94.76** |
| ResNet50 | Average Blur | 68.75 | 76.93 | 68.75 | 64.84 | 87.60 |
| ResNet50 | Gaussian Blur | 67.19 | 75.63 | 67.19 | 61.86 | 89.84 |
| ResNet50 | Baseline (Raw) | 67.19 | 78.01 | 67.19 | 60.47 | 92.94 |
| ResNet101 | Average Blur | 70.31 | 79.64 | 70.31 | 63.05 | 88.15 |
| ResNet101 | Sharpening | 68.75 | 79.59 | 68.75 | 61.98 | 91.96 |
| AlexNet | Baseline (Raw) | 67.19 | 73.20 | 67.19 | 64.85 | 90.20 |

### Classifier Benchmark Summary (Lab 05 — Industrial Defect Detection)

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | Latency (ms/sample) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **HOG + SVM (RBF)** | **77.74** | **80.43** | **83.52** | **81.95** | **0.14 ms** |
| HOG + XGBoost (GPU) | 76.39 | 79.71 | 81.78 | 80.73 | 0.12 ms |
| HOG + Random Forest | 75.79 | 75.85 | 87.98 | 81.47 | 0.24 ms |
| HOG + SVM (Linear) | 71.06 | 77.95 | 72.74 | 75.26 | 0.13 ms |
| HOG + kNN ($k=5$) | 51.57 | 90.05 | 22.43 | 35.91 | 1.07 ms |

### SIFT Landmark Recognition Summary (Lab 06 — Test Perturbations & Open-Set Queries)

| Query Condition | Test Count | Avg Keypoints | Avg Reliable Matches | Avg RANSAC Inliers | Avg Inference Time (s) | Recognition / Correct Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Rotated (45°)** | 6 | 347 | 13.8 | 7.8 | 0.58 s | 1/6 (Recognized) |
| **Different Viewpoint** | 6 | 670 | 15.5 | 8.7 | 0.93 s | 0/6 (Rejected - Strict Margins) |
| **Original (Held-out)** | 6 | 257 | 10.5 | 5.2 | 0.36 s | 0/6 (Rejected - Below Threshold) |
| **Resized (50% Scale)** | 6 | 90 | 5.0 | 0.0 | 0.12 s | 0/6 (Rejected - Few Features) |
| **Low Brightness ($\alpha=0.35$)** | 6 | 84 | 4.7 | 0.0 | 0.11 s | 0/6 (Rejected - Low Contrast) |
| **Open-Set Negative (Stonehenge)** | 2 | 948 | 31.0 | 6.5 | 1.82 s | **2/2 (Correctly Rejected - 100%)** |
| **Overall Stress Benchmark** | **32** | **331** | **11.2** | **4.5** | **0.51 s** | **3/32 (9.4% strictly passed; 0% false positives)** |

### GA-HOG Feature Selection Benchmark (Lab 07 — PneumoniaMNIST)

| Pipeline | Feature Dimension | Reduction (%) | Accuracy (%) | Macro-Precision (%) | Macro-Recall (%) | Macro-F1 (%) | Normal Recall (%) | Pneumonia Recall (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GA-HOG + SVM (Proposed)** | **2,129** | **73.7%** | **84.78** | **89.63** | **79.87** | **81.95** | **60.26** | **99.49** |
| HOG + SVM (Baseline) | 8,100 | 0.0% | 81.57 | 88.25 | 75.51 | 77.36 | 51.28 | **99.74** |
| **Net Improvement** | **-5,971 features** | **-73.7%** | **+3.21%** | **+1.38%** | **+4.36%** | **+4.58%** | **+8.98%** | -0.25% |

---

## 🛠️ Tech Stack & Dependencies

The experiments in this repository utilize standard scientific Python libraries, machine learning algorithms, and deep learning frameworks:

- **Language:** Python 3.10+
- **Deep Learning:** PyTorch, Torchvision
- **Computer Vision & Image Processing:** OpenCV (`cv2`), scikit-image, Albumentations, PIL
- **Machine Learning & Evaluation:** scikit-learn, XGBoost, Joblib
- **Data & Visualization:** NumPy, Pandas, Matplotlib, Seaborn
- **Web Prototype (Bonus):** Flask, HTML5, Vanilla CSS, JavaScript
- **Environment:** Jupyter Notebook / Google Colab

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/muhammadumerrafiq/Computer-Vision-Labs.git
cd Computer-Vision-Labs
```

### 2. Git LFS Setup (Download Model Weights)
This repository uses **Git LFS** for the large model file (`hog_defect_model.joblib`, ~183 MB) in Lab 05. Run:
```bash
git lfs install
git lfs pull
```
This is required so the actual `hog_defect_model.joblib` model binary is downloaded instead of only the Git LFS pointer.

### 3. Set Up a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install torch torchvision opencv-python numpy pandas matplotlib seaborn scikit-learn scikit-image xgboost joblib flask jupyter
```

### 5. Launch Jupyter Notebooks
```bash
jupyter notebook
```
Navigate to any lab folder (`CV Lab 01`, `CV Lab 02`, `CV Lab 03`, `Lab Assignment 01_CV Lab 04`, `CV Lab 05`, `CV Lab 06`, or `CV Lab 07`) and execute the respective `.ipynb` notebook.

### 6. Run the Real-Time Quality Inspection Prototype (Lab 05 Bonus)
```bash
cd "CV Lab 05/Bonus Task"
pip install -r requirements.txt
python app.py
```
Open your web browser at `http://127.0.0.1:5000` to interact with the image inspection, conveyor belt video stream, and live webcam inspection modules.

---

## 👤 Author & Academic Info

- **Course:** Introduction to Computer Vision
- **Student ID:** FA23-BAI-013
- **Author:** [Muhammad Umer Rafiq](https://github.com/muhammadumerrafiq)
- **Program:** BS Artificial Intelligence (Semester 7)

---

## 📜 License

This project is licensed under the [MIT License](LICENSE) — feel free to use and adapt this code for academic and educational purposes.
