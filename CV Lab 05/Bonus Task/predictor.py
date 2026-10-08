"""
predictor.py
============
Unified HOG + SVM Defect Detection Module.
Loads the existing `hog_defect_model.joblib` and uses `hog_utils.py`
to provide consistent predictions for:
  - Uploaded Images
  - Uploaded Videos (frame-by-frame)
  - Live Webcam Streams
"""

import os
import time
import base64
import cv2
import numpy as np
import joblib
import hog_utils

MODEL_PATH = os.path.join(os.path.dirname(__file__), "hog_defect_model.joblib")


class DefectPredictor:
    """Singleton predictor that wraps the trained HOG + SVM model bundle."""

    _instance = None

    def __init__(self, model_path=MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        print(f"[DefectPredictor] Loading trained model bundle from {model_path}...")
        self.bundle = joblib.load(model_path)
        self.model = self.bundle["model"]
        self.cell = self.bundle["cell"]
        self.orient = self.bundle["orient"]
        self.size = self.bundle["size"]
        self.patch = self.bundle["patch"]
        self.thr = float(self.bundle["thr"])
        self.min_windows = int(self.bundle["min_windows"])

        print("[DefectPredictor] Model loaded successfully.")
        print(f"  - Model Type: {self.model.__class__.__name__} ({self.model.kernel} kernel, C={self.model.C})")
        print(f"  - HOG Cell Size: {self.cell}x{self.cell}")
        print(f"  - HOG Orientations: {self.orient}")
        print(f"  - HOG Resize Size: {self.size}x{self.size}")
        print(f"  - Sliding Patch Size: {self.patch}x{self.patch}")
        print(f"  - Defect Threshold: {self.thr}")
        print(f"  - Min Defective Windows: {self.min_windows}")
        print(f"  - Training Vectors: {self.model.Xtr_.shape[0]} vectors of {self.model.Xtr_.shape[1]} features")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = DefectPredictor()
        return cls._instance

    def predict_image(self, img, stride=32, top_k=3):
        """
        Core unified prediction function.
        Reused identically across:
          - Image Upload
          - Video Frame
          - Webcam Frame

        Parameters:
          img: numpy array (BGR or Grayscale uint8)
          stride: sliding window step in pixels (default 32)
          top_k: top k probabilities used for non-defective confidence

        Returns:
          dict containing:
            - is_defective (bool)
            - prediction: "PASS" or "DEFECTIVE"
            - action: "ACCEPT PRODUCT" or "REJECT PRODUCT"
            - confidence: float in [0.0, 1.0]
            - defective_windows: int count of defective windows
            - total_windows: int total sliding windows evaluated
            - boxes: list of [x, y, w, h, prob] for all windows
            - defective_boxes: list of [x, y, w, h, prob] for positive windows
            - latency_ms: float execution time in milliseconds
        """
        t0 = time.perf_counter()

        if img is None or img.size == 0:
            raise ValueError("Input image is empty or invalid.")

        # Ensure 2D/3D uint8
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        # Run inspect_product from hog_utils
        raw_res = hog_utils.inspect_product(img, self.bundle, stride=stride, top_k=top_k)
        latency = (time.perf_counter() - t0) * 1000.0

        is_defective = bool(raw_res["defective"])
        coords = raw_res["coords"]
        probs = raw_res["probs"]
        positives = raw_res["positive"]

        all_boxes = []
        defective_boxes = []

        for (x, y, w, h), p, is_pos in zip(coords, probs, positives):
            box_info = {
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "prob": round(float(p), 4),
                "is_defective": bool(is_pos),
            }
            all_boxes.append(box_info)
            if is_pos:
                defective_boxes.append(box_info)

        prediction_label = "DEFECTIVE" if is_defective else "PASS"
        action_label = "REJECT PRODUCT" if is_defective else "ACCEPT PRODUCT"

        return {
            "is_defective": is_defective,
            "prediction": prediction_label,
            "action": action_label,
            "confidence": round(float(raw_res["confidence"]), 4),
            "confidence_pct": round(float(raw_res["confidence"]) * 100.0, 1),
            "defective_windows": int(positives.sum()),
            "total_windows": len(coords),
            "min_windows_required": self.min_windows,
            "threshold": self.thr,
            "boxes": all_boxes,
            "defective_boxes": defective_boxes,
            "latency_ms": round(latency, 1),
        }

    def annotate_frame(self, img, result, show_all_boxes=False):
        """
        Renders visual annotations on the image:
          - Red bounding boxes around defective windows
          - Green status badge or outline for PASS
          - Defect probability badges
        """
        annotated = img.copy()
        if annotated.ndim == 2:
            annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)

        h, w = annotated.shape[:2]

        if result["is_defective"]:
            # Draw red bounding boxes for defective regions
            for box in result["defective_boxes"]:
                bx, by, bw, bh = box["x"], box["y"], box["w"], box["h"]
                p = box["prob"]
                # Box
                cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (0, 0, 220), 2)
                # Label badge
                badge_text = f"DEFECT {int(p*100)}%"
                (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(
                    annotated,
                    (bx, max(by - 16, 0)),
                    (bx + tw + 6, max(by, 16)),
                    (0, 0, 220),
                    -1,
                )
                cv2.putText(
                    annotated,
                    badge_text,
                    (bx + 3, max(by - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            # Red corner indicator
            cv2.rectangle(annotated, (0, 0), (w - 1, h - 1), (0, 0, 220), 3)
        else:
            # Subtle green border / corner accents for PASS
            cv2.rectangle(annotated, (0, 0), (w - 1, h - 1), (0, 200, 70), 3)

        return annotated

    def encode_base64(self, img_bgr, quality=85):
        """Encodes an OpenCV image to base64 JPEG string."""
        success, buffer = cv2.imencode(
            ".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        )
        if not success:
            return ""
        return base64.b64encode(buffer).decode("utf-8")


# Global convenient accessor
def get_predictor():
    return DefectPredictor.get_instance()
