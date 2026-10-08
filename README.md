# Computer Vision Labs & Assignments

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8.svg)](https://opencv.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626.svg)](https://jupyter.org/)

A comprehensive collection of practical laboratory experiments, benchmarks, and assignments for **Introduction to Computer Vision (7th Semester)**. This repository explores classical image processing pipelines, spatial filtering, edge detection, mathematical morphology, deep Convolutional Neural Networks (CNNs) benchmarked on dermoscopic medical imaging datasets (ISIC and HAM10000), and feature-descriptor-based industrial defect detection pipelines (HOG + Kernel SVM) with real-time web deployment.

---

## 📑 Table of Contents

- [Repository Structure](#-repository-structure)
- [Lab Overview & Modules](#-lab-overview--modules)
  - [Lab 01: Deep CNN Architecture Benchmarking](#lab-01-deep-cnn-architecture-benchmarking)
  - [Lab 02: Spatial Filtering & Feature Enhancement](#lab-02-spatial-filtering--feature-enhancement)
  - [Lab 03: Edge Detection, Noise Sensitivity & Edge Representations](#lab-03-edge-detection-noise-sensitivity--edge-representations)
  - [Lab Assignment 01 (Lab 04): Skin Lesion Boundary Detection & Morphometry](#lab-assignment-01-lab-04-skin-lesion-boundary-detection--morphometry)
  - [Lab 05: Industrial Defect Detection using HOG & Kernel SVM](#lab-05-industrial-defect-detection-using-hog--kernel-svm)
- [Key Empirical Results](#-key-empirical-results)
  - [Filter Benchmark Summary (Lab 02)](#filter-benchmark-summary-lab-02--test-set)
  - [Classifier Benchmark Summary (Lab 05)](#classifier-benchmark-summary-lab-05--industrial-defect-detection)
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
├── LICENSE                         # MIT License
└── README.md                       # Repository documentation
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
Navigate to any lab folder (`CV Lab 01`, `CV Lab 02`, `CV Lab 03`, `Lab Assignment 01_CV Lab 04`, or `CV Lab 05`) and execute the respective `.ipynb` notebook.

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
