"""Flask web app: upload an ovarian ultrasound and compare the quantum and metaheuristic branches.

Run `python 04_train_final.py` first (it writes models/ and results/comparative_metrics.json).
"""
import base64
import io
import json
from importlib import import_module

import cv2
import joblib
import numpy as np
import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image

from src.config import IMG_SIZE, MODELS, RESULTS
from src.dataset import clahe_rgb
from src.hybrid_model import HybridPCOSClassifier, predict_hybrid
from src.quantum_branch import QuantumFeatureExtractor

features = import_module("02_extract_features")

app = Flask(__name__, template_folder="templates", static_folder="templates", static_url_path="/static")

backbone = features.load_backbone()
q_extractor = QuantumFeatureExtractor()
masks = np.load(MODELS / "masks.npz")
classifiers = {name: joblib.load(MODELS / f"{name}_clf.joblib") for name in ["quantum", *masks.files]}


def load_hybrid():
    """Final diagnosis: fusion MLP over quantum + selected classical features (5-seed ensemble)."""
    ckpt = torch.load(MODELS / "hybrid.pt", weights_only=False)
    members = []
    for m in ckpt["members"]:
        model = HybridPCOSClassifier(quantum_dim=len(m["mu_q"]), classical_dim=len(m["mu_c"]))
        model.load_state_dict(m["state_dict"])
        model.eval()
        members.append((model, {k: m[k].numpy() for k in ("mu_q", "sd_q", "mu_c", "sd_c")}))
    return ckpt["selector"], members


hybrid_selector, hybrid_members = load_hybrid()


def looks_like_ultrasound(rgb):
    """Cheap sanity check: ultrasounds are (near-)grayscale and not blank."""
    r, g, b = (rgb[..., i].astype(float) for i in range(3))
    if np.mean(np.abs(r - g)) + np.mean(np.abs(g - b)) > 12.0:
        return False, "the image is in colour, ultrasounds are grayscale"
    if not 10 <= rgb.mean() <= 190:
        return False, "the image is too dark or too bright"
    return True, ""


def dark_blobs(rgb, max_regions=15):
    """Heuristic visual aid (NOT a model output): outlines round regions that are clearly darker than
    the image, which in an ovarian ultrasound are often follicles. Classical image processing only."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 6)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tissue = gray[gray > 20].mean() if np.any(gray > 20) else gray.mean()
    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        if not (60 < area < 2500 and 0.6 < w / h < 1.65):
            continue
        circularity = 4 * np.pi * area / (cv2.arcLength(cnt, True) ** 2 + 1e-6)
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        if circularity > 0.55 and cv2.mean(blur, mask)[0] < 0.6 * tissue:
            boxes.append((area, x, y, w, h))
    out = rgb.copy()
    for _, x, y, w, h in sorted(boxes, reverse=True)[:max_regions]:
        cv2.rectangle(out, (x, y), (x + w, y + h), (244, 63, 94), 1)
    return out, min(len(boxes), max_regions)


def branch(p):
    return {"label": "Ultrasound shows signs of PCOS" if p >= 0.5 else "Ultrasound does not show signs of PCOS",
            "is_infected": bool(p >= 0.5), "confidence": round(100 * max(p, 1 - p), 1)}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/metrics/<algo>")
def get_metrics(algo):
    data = json.loads((RESULTS / "comparative_metrics.json").read_text())
    return jsonify({"quantum": data["quantum"], "hybrid": data["hybrid"],
                    "metaheuristic": data.get(algo.upper(), data["PSO"])})


@app.route("/api/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    algo = request.form.get("algorithm", "PSO").upper()
    if algo not in masks.files:
        algo = "PSO"

    img = Image.open(io.BytesIO(request.files["file"].read())).convert("RGB")
    rgb = np.asarray(img.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC), dtype=np.uint8)
    ok, why = looks_like_ultrasound(rgb)
    if not ok:
        return jsonify({"error": f"Invalid image: {why}. Please upload a pelvic ultrasound scan.",
                        "status": "rejected"}), 422

    enhanced = clahe_rgb(rgb)
    c_feats = features.classical_features(enhanced[None], backbone)
    q_feats = q_extractor(features.small_gray(enhanced[None])).numpy()
    p_q = float(classifiers["quantum"].predict_proba(q_feats)[0, 1])
    p_m = float(classifiers[algo].predict_proba(c_feats[:, masks[algo]])[0, 1])
    sel = c_feats[:, masks[hybrid_selector]]
    p_h = float(np.mean([predict_hybrid(m, st, q_feats, sel)[0] for m, st in hybrid_members]))

    shown, n_blobs = dark_blobs(enhanced) if p_h >= 0.5 else (enhanced, 0)
    _, buf = cv2.imencode(".png", cv2.cvtColor(shown, cv2.COLOR_RGB2BGR))
    return jsonify({"hybrid_pred": branch(p_h),
                    "quantum_pred": branch(p_q),
                    "metaheuristic_pred": {**branch(p_m), "algorithm": algo},
                    "follicles_detected": n_blobs,
                    "annotated_image": "data:image/png;base64," + base64.b64encode(buf).decode()})


if __name__ == "__main__":
    app.run(port=5000)
