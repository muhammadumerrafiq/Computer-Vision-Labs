"""
app.py
======
Flask Application for HOG Industrial Defect Detection.
Computer Vision Lab 05 Bonus Challenge.

Provides:
  - Professional Industrial Inspection UI
  - Image Inspection (upload or sample)
  - Video Inspection (frame-by-frame processing + summary)
  - Live Webcam Inspection (real-time stream + designated inspection area)
"""

import os
import time
import uuid
import json
import base64
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory, Response

from predictor import get_predictor

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB max upload

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
SAMPLES_DIR = os.path.join(BASE_DIR, "sample_images")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)

# Pre-load predictor singleton on server start
predictor = get_predictor()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/sample_images/<path:filename>")
def serve_sample(filename):
    return send_from_directory(SAMPLES_DIR, filename)


@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/api/model_info", methods=["GET"])
def model_info():
    """Returns trained model architecture & HOG parameters."""
    return jsonify({
        "status": "success",
        "model_type": "KernelSVC (libsvm precomputed Gram matrix)",
        "kernel": predictor.model.kernel,
        "C": predictor.model.C,
        "cell_size": f"{predictor.cell}x{predictor.cell}",
        "orientations": predictor.orient,
        "input_size": f"{predictor.size}x{predictor.size}",
        "patch_size": f"{predictor.patch}x{predictor.patch}",
        "threshold": predictor.thr,
        "min_defective_windows": predictor.min_windows,
        "num_training_vectors": int(predictor.model.Xtr_.shape[0]),
        "feature_dimension": int(predictor.model.Xtr_.shape[1]),
        "classes": [int(c) for c in predictor.model.classes_],
    })


@app.route("/api/samples", methods=["GET"])
def list_samples():
    """Returns list of pre-configured sample files for quick testing."""
    samples = [
        {
            "id": "pass_steel",
            "name": "Normal Steel Surface (PASS)",
            "type": "image",
            "filename": "pass_steel.jpg",
            "description": "Clean cold-rolled industrial steel surface with uniform grain.",
            "expected": "PASS",
        },
        {
            "id": "defect_scratch",
            "name": "Scratched Steel Surface (DEFECTIVE)",
            "type": "image",
            "filename": "defect_scratch.jpg",
            "description": "Industrial steel surface with linear scratches across surface.",
            "expected": "DEFECTIVE",
        },
        {
            "id": "defect_patch",
            "name": "Oxidized Patch Defect (DEFECTIVE)",
            "type": "image",
            "filename": "defect_patch.jpg",
            "description": "Steel plate exhibiting local oxidation patch defects.",
            "expected": "DEFECTIVE",
        },
        {
            "id": "unseen_surface",
            "name": "Unseen Non-Steel Surface (UNSEEN TEST)",
            "type": "image",
            "filename": "unseen_surface.jpg",
            "description": "Arbitrary non-metallic texture to demonstrate out-of-distribution evaluation.",
            "expected": "DEFECTIVE",
        },
        {
            "id": "sample_conveyor",
            "name": "Conveyor Belt Stream (VIDEO)",
            "type": "video",
            "filename": "sample_conveyor.mp4",
            "description": "Simulated industrial conveyor sequence transitioning from pass to defective product.",
            "expected": "MIXED",
        },
    ]
    return jsonify({"samples": samples})


def _decode_image_payload(req):
    """Decodes image from either multipart file upload or JSON base64 string."""
    if "file" in req.files:
        file = req.files["file"]
        in_memory = file.read()
        nparr = np.frombuffer(in_memory, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        filename = file.filename
        return img, filename, False
    elif req.is_json:
        data = req.get_json()
        if "sample_filename" in data:
            sample_path = os.path.join(SAMPLES_DIR, data["sample_filename"])
            if os.path.exists(sample_path):
                img = cv2.imread(sample_path)
                return img, data["sample_filename"], False
        elif "image_base64" in data:
            b64_str = data["image_base64"]
            if "," in b64_str:
                b64_str = b64_str.split(",", 1)[1]
            img_bytes = base64.b64decode(b64_str)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            is_unseen = data.get("is_unseen", False)
            return img, "uploaded_image.jpg", is_unseen
    return None, None, False


@app.route("/api/predict_image", methods=["POST"])
def predict_image_endpoint():
    """Inspects a single uploaded or sample image."""
    try:
        img, filename, is_unseen = _decode_image_payload(request)
        if img is None:
            return jsonify({"status": "error", "message": "No valid image provided."}), 400

        # Run central predictor
        result = predictor.predict_image(img)
        annotated = predictor.annotate_frame(img, result)

        # Generate base64 images for immediate display
        annotated_b64 = predictor.encode_base64(annotated)
        original_b64 = predictor.encode_base64(img)

        # Save record copy to uploads
        save_name = f"{uuid.uuid4().hex[:8]}_{filename}"
        save_path = os.path.join(UPLOAD_DIR, save_name)
        cv2.imwrite(save_path, annotated)

        return jsonify({
            "status": "success",
            "filename": filename,
            "prediction": result["prediction"],
            "action": result["action"],
            "confidence": result["confidence"],
            "confidence_pct": result["confidence_pct"],
            "is_defective": result["is_defective"],
            "defective_windows": result["defective_windows"],
            "total_windows": result["total_windows"],
            "threshold": result["threshold"],
            "min_windows_required": result["min_windows_required"],
            "latency_ms": result["latency_ms"],
            "is_unseen": is_unseen,
            "boxes": result["defective_boxes"],
            "annotated_image": f"data:image/jpeg;base64,{annotated_b64}",
            "original_image": f"data:image/jpeg;base64,{original_b64}",
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/predict_frame", methods=["POST"])
def predict_frame_endpoint():
    """
    Lightweight endpoint for real-time webcam & video streaming.
    Accepts base64 frame, crops to inspection area if provided,
    and returns predictions with defective bounding boxes.
    """
    try:
        data = request.get_json(force=True)
        b64_str = data.get("frame", "")
        if not b64_str:
            return jsonify({"status": "error", "message": "Missing frame"}), 400

        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]

        img_bytes = base64.b64decode(b64_str)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"status": "error", "message": "Failed to decode frame"}), 400

        # Check if crop ROI is requested
        roi = data.get("roi")
        offset_x, offset_y = 0, 0
        if roi:
            rx = max(0, int(roi.get("x", 0)))
            ry = max(0, int(roi.get("y", 0)))
            rw = int(roi.get("w", frame.shape[1]))
            rh = int(roi.get("h", frame.shape[0]))
            # Bound check
            rx = min(rx, frame.shape[1] - 1)
            ry = min(ry, frame.shape[0] - 1)
            rw = min(rw, frame.shape[1] - rx)
            rh = min(rh, frame.shape[0] - ry)
            if rw > 10 and rh > 10:
                crop = frame[ry : ry + rh, rx : rx + rw]
                offset_x, offset_y = rx, ry
            else:
                crop = frame
        else:
            crop = frame

        # Run central predictor on cropped inspection area
        # For real-time streaming, stride 32 provides fast response
        result = predictor.predict_image(crop, stride=32)

        # Shift box coordinates back to full-frame space if ROI was used
        adjusted_defects = []
        for box in result["defective_boxes"]:
            adjusted_defects.append({
                "x": box["x"] + offset_x,
                "y": box["y"] + offset_y,
                "w": box["w"],
                "h": box["h"],
                "prob": box["prob"],
            })

        return jsonify({
            "status": "success",
            "prediction": result["prediction"],
            "action": result["action"],
            "confidence_pct": result["confidence_pct"],
            "is_defective": result["is_defective"],
            "defective_windows": result["defective_windows"],
            "total_windows": result["total_windows"],
            "latency_ms": result["latency_ms"],
            "boxes": adjusted_defects,
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/inspect_video", methods=["POST"])
def inspect_video_endpoint():
    """
    Processes an uploaded video file frame by frame.
    Generates summary report:
      - Frames Processed
      - Defective Frames
      - Pass Frames
      - Overall Decision
      - Sample annotated keyframes
    """
    try:
        video_file = request.files.get("video")
        sample_filename = request.form.get("sample_filename")

        if video_file:
            input_ext = os.path.splitext(video_file.filename)[1].lower()
            if input_ext not in [".mp4", ".avi", ".mov", ".mkv"]:
                return jsonify({"status": "error", "message": "Unsupported video format."}), 400
            video_id = uuid.uuid4().hex[:8]
            video_path = os.path.join(UPLOAD_DIR, f"{video_id}_{video_file.filename}")
            video_file.save(video_path)
            orig_name = video_file.filename
        elif sample_filename:
            video_path = os.path.join(SAMPLES_DIR, sample_filename)
            if not os.path.exists(video_path):
                return jsonify({"status": "error", "message": "Sample video not found."}), 404
            orig_name = sample_filename
        else:
            return jsonify({"status": "error", "message": "No video provided."}), 400

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return jsonify({"status": "error", "message": "Could not open video file."}), 400

        total_source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        processed_count = 0
        defective_count = 0
        pass_count = 0
        frame_history = []
        keyframes = []

        # Process every frame (or skip frames if video is very long)
        frame_step = 1
        if total_source_frames > 150:
            frame_step = 2
        if total_source_frames > 300:
            frame_step = 3

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_step == 0:
                # Run central predictor
                # Resize if frame is huge to ensure consistent patch resolution
                h, w = frame.shape[:2]
                if max(h, w) > 400:
                    scale = 400.0 / max(h, w)
                    proc_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                else:
                    proc_frame = frame

                res = predictor.predict_image(proc_frame, stride=32)
                processed_count += 1

                if res["is_defective"]:
                    defective_count += 1
                else:
                    pass_count += 1

                frame_summary = {
                    "frame_idx": frame_idx,
                    "prediction": res["prediction"],
                    "confidence_pct": res["confidence_pct"],
                    "defective_windows": res["defective_windows"],
                }
                frame_history.append(frame_summary)

                # Keep up to 6 representative keyframes (3 pass, 3 defective)
                if len(keyframes) < 8 and (
                    processed_count == 1
                    or (res["is_defective"] and defective_count <= 4)
                    or (not res["is_defective"] and pass_count <= 4)
                ):
                    annotated = predictor.annotate_frame(proc_frame, res)
                    b64 = predictor.encode_base64(annotated, quality=75)
                    keyframes.append({
                        "frame_idx": frame_idx,
                        "prediction": res["prediction"],
                        "confidence_pct": res["confidence_pct"],
                        "image": f"data:image/jpeg;base64,{b64}",
                    })

            frame_idx += 1

        cap.release()

        overall_defective = defective_count > 0
        overall_prediction = "DEFECTIVE" if overall_defective else "PASS"
        overall_action = "REJECT PRODUCT" if overall_defective else "ACCEPT PRODUCT"

        return jsonify({
            "status": "success",
            "video_name": orig_name,
            "total_source_frames": total_source_frames,
            "processed_frames": processed_count,
            "defective_frames": defective_count,
            "pass_frames": pass_count,
            "overall_prediction": overall_prediction,
            "overall_action": overall_action,
            "overall_is_defective": overall_defective,
            "defect_rate_pct": round((defective_count / max(processed_count, 1)) * 100.0, 1),
            "keyframes": keyframes,
            "frame_history": frame_history[-30:],  # last 30 frames for chart/log
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    print("\n=======================================================")
    print("  HOG Industrial Defect Detection Application Starting ")
    print("  Computer Vision Lab 05 Bonus Challenge Prototype     ")
    print("  Serving at: http://127.0.0.1:5000                   ")
    print("=======================================================\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
