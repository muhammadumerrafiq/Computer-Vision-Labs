# HOG Industrial Defect Detection

> **Real-Time HOG-Based Industrial Quality Inspection Prototype**  
> Developed for **Computer Vision Lab 05 Bonus Challenge**

---

## 1. Project Purpose

In industrial manufacturing, manual surface inspection of steel plates and hot-rolled coils is labor-intensive, error-prone, and slow. This project implements a real-time computer vision quality inspection system that automatically identifies defective and non-defective industrial surfaces using **Histogram of Oriented Gradients (HOG)** feature extraction and a trained **Support Vector Machine (Kernel SVM with RBF kernel)** classifier.

The application provides a web-based industrial interface supporting:
1. **Single & Batch Image Inspection** with defective-region bounding box visualization.
2. **Video Stream Inspection** for moving conveyor belt footage with frame-by-frame analysis and summary reports.
3. **Live Webcam Inspection** with designated 200×200 industrial inspection region and real-time PASS / DEFECTIVE status updates.

---

## 2. Lab 05 Objective & Bonus Challenge

- **Lab 05 Objective:** Apply Histogram of Oriented Gradients (HOG) to a real-world industrial computer vision problem to distinguish normal/non-defective products from defective steel surfaces.
- **Original Pipeline:**
  $$\text{Image} \longrightarrow \text{Preprocessing (Grayscale / Scaling)} \longrightarrow \text{HOG Feature Extraction} \longrightarrow \text{SVM Classifier} \longrightarrow \text{PASS / DEFECTIVE Decision}$$
- **Bonus Challenge Requirement:**
  > *"Develop a real-time prototype using a webcam or video stream. The system should extract HOG features from each frame and display the predicted industrial product status as either PASS or DEFECTIVE."*

This prototype directly satisfies the bonus challenge by deploying the exact trained HOG + SVM pipeline into a real-time responsive stream.

---

## 3. How the HOG + SVM Model Works

The model uses the pre-trained weights from `hog_defect_model.joblib` and feature extraction from `hog_utils.py`:

- **HOG Cell Size:** $8 \times 8$ pixels per cell.
- **Orientations:** 9 gradient orientation bins in $[0, \pi)$ (unsigned gradients with linear soft-binning).
- **Block Normalization:** $2 \times 2$ cell blocks with stride of 1 cell and L2-Hys clipping (max 0.2).
- **Patch Resolution:** Resized to $128 \times 128$ for HOG feature extraction, yielding a feature vector of **8,100 dimensions** ($15 \times 15 \times 4 \times 9 = 8,100$).
- **Sliding Window:** Scans surfaces with $64 \times 64$ sliding windows (stride 32px).
- **Classifier:** `KernelSVC` with an **RBF kernel** ($C = 10$) accelerated via GPU/PyTorch Gram matrix computation and solved by libsvm.
- **Decision Logic:**
  - Probability threshold: $\text{thr} = 0.50$.
  - Minimum defective windows required for rejection: $\text{min\_windows} = 2$.
  - If $\ge 2$ windows have $P(\text{defect}) \ge 0.50$, the product is **DEFECTIVE** $\rightarrow$ **ACTION: REJECT PRODUCT**.
  - Otherwise, the product is **PASS** $\rightarrow$ **ACTION: ACCEPT PRODUCT**.

---

## 4. Installation

Clone or open the project folder in your terminal:

```bash
cd "HOG Defect Detection"
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## 5. How to Run the Application

Start the local Flask inspection server:

```bash
python app.py
```

Once started, open your web browser at:

```
http://127.0.0.1:5000
```

---

## 6. How Image Prediction Works

1. Select **Image Inspection** tab.
2. Click **Browse Image File** or drag & drop any image (`.jpg`, `.jpeg`, `.png`, `.webp`), or click any of the preset sample chips (**Normal Steel**, **Scratch Defect**, **Oxidation Patch**, **Unseen Surface**).
3. Click **Run Quality Inspection**.
4. The system:
   - Converts the image to grayscale and slides $64 \times 64$ windows across the surface.
   - Resizes each window to $128 \times 128$ and computes the 8,100-dimensional HOG descriptor.
   - Evaluates each patch through `model.predict_proba()`.
   - Displays the **Prediction** (PASS / DEFECTIVE), **Confidence %**, **Action** (ACCEPT / REJECT), and overlays red bounding boxes over the exact detected defect windows.

---

## 7. How Video Prediction Works

1. Select **Video Inspection** tab.
2. Upload a video file (`.mp4`, `.avi`, `.mov`, `.mkv`) or click **Conveyor Belt Stream (Mixed PASS & DEFECT)**.
3. Click **Start Frame-by-Frame Inspection**.
4. The system:
   - Decodes video frames sequentially using OpenCV.
   - Preprocesses each frame and runs the unified HOG + SVM predictor.
   - Tracks defective frames vs. pass frames in real time with a progress bar.
   - Displays a comprehensive **VIDEO INSPECTION COMPLETE** report with total frames, defect rate %, and an annotated keyframe gallery highlighting defect moments.

---

## 8. How Live Webcam Prediction Works

1. Select **Live Webcam Inspection** tab.
2. Click **START WEBCAM** and allow camera permissions in your browser.
3. Align the test product or sample image inside the clearly delineated **INSPECTION AREA** ($200 \times 200$ pixels):
   ```
   ┌─────────────────────────────┐
   │      INSPECTION AREA        │
   │     PLACE PRODUCT HERE      │
   └─────────────────────────────┘
   ```
4. The system streams the inspection ROI to the backend prediction module, dynamically overlays red bounding boxes on defects, and updates the large real-time status banner:
   - **PASS** (Green glow) $\rightarrow$ **ACCEPT PRODUCT**
   - **DEFECTIVE** (Pulsing Red) $\rightarrow$ **REJECT PRODUCT**
5. Live FPS and inference latency (ms) are displayed continuously. Click **STOP WEBCAM** to terminate the stream.

---

## 9. Model Limitations & Safety Notice

> **Notice:** This prototype was trained on industrial steel surface images (from the NEU Surface Defect Database). Predictions are intended for metallic surfaces similar to the training data and may be unreliable for unrelated objects or arbitrary domestic environments. Out-of-distribution textures will trigger an "Unseen Image Test" notice.
