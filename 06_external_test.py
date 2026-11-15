"""Step 6: external test on ultrasound images from outside the dataset.

Put images in two folders and run:
    python 06_external_test.py path/to/external        # contains pcos/ and normal/

Each image is first checked against the training dataset. Many PCOS ultrasounds found online come
from the same public Kaggle dataset; such copies (same scan, possibly cropped or flipped) would
inflate the score, so they are reported separately and excluded from the external accuracy.

Output: results/external_test.json, results/external_test.csv
"""
import json
import sys
from importlib import import_module
from pathlib import Path

import joblib
import numpy as np
from PIL import Image, ImageOps

from src.config import ARTIFACTS, IMG_SIZE, MODELS, RESULTS
from src.dataset import clahe_rgb
from src.quantum_branch import QuantumFeatureExtractor

prep = import_module("01_prepare_data")
feats = import_module("02_extract_features")
LABELS = {"normal": 0, "notinfected": 0, "pcos": 1, "infected": 1}


def load_folder(root):
    files, labels, images = [], [], []
    for sub in sorted(Path(root).iterdir()):
        if not sub.is_dir() or sub.name.lower() not in LABELS:
            continue
        for f in sorted(sub.iterdir()):
            if f.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue
            try:
                img = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
            except Exception:
                print("skipping unreadable", f)
                continue
            images.append(np.asarray(img.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC), dtype=np.uint8))
            labels.append(LABELS[sub.name.lower()])
            files.append(f"{sub.name}/{f.name}")
    return np.stack(images), np.array(labels), files


def in_training_data(images):
    """Same-scan check against the dataset: embedding neighbours + SIFT/RANSAC verification."""
    train_imgs = np.load(ARTIFACTS / "images.npy")
    train_emb = np.load(ARTIFACTS / "emb.npy")
    emb = prep.embed(images)
    sim = emb @ train_emb.T
    ext_kp = prep.keypoint_features(images)
    flags = []
    for i in range(len(images)):
        nn = np.argsort(-sim[i])[:10]
        kp = prep.keypoint_features(train_imgs[nn])
        best = max(prep.inliers(ext_kp[i], k) for k in kp)
        flags.append(bool(sim[i, nn[0]] >= prep.SURE_SIMILARITY or best >= prep.MIN_INLIERS))
    return np.array(flags), sim.max(1)


def main(root):
    images, y, files = load_folder(root)
    print(f"{len(images)} external images (normal={np.sum(y == 0)}, PCOS={np.sum(y == 1)})")
    seen, max_sim = in_training_data(images)
    print(f"{seen.sum()} of them are copies of scans in the training dataset (excluded)")

    enhanced = np.stack([clahe_rgb(im) for im in images])
    c = feats.classical_features(enhanced)
    q = QuantumFeatureExtractor()(feats.small_gray(enhanced)).numpy()
    masks = np.load(MODELS / "masks.npz")
    probs = {"quantum": joblib.load(MODELS / "quantum_clf.joblib").predict_proba(q)[:, 1]}
    for a in masks.files:
        probs[a] = joblib.load(MODELS / f"{a}_clf.joblib").predict_proba(c[:, masks[a]])[:, 1]

    new = ~seen
    out = {"n_images": int(len(y)), "n_copies_of_training_data": int(seen.sum()), "n_new": int(new.sum()),
           "new_label_counts": {"normal": int(np.sum(y[new] == 0)), "pcos": int(np.sum(y[new] == 1))}}
    for name, p in probs.items():
        pred = (p >= 0.5).astype(int)
        out[name] = {"accuracy_new_images": round(float(np.mean(pred[new] == y[new])), 4) if new.any() else None,
                     "correct_new_images": f"{int(np.sum(pred[new] == y[new]))}/{int(new.sum())}",
                     "accuracy_copies": round(float(np.mean(pred[seen] == y[seen])), 4) if seen.any() else None}
        print(f"{name:8s} new images: {out[name]['correct_new_images']} correct")
    (RESULTS / "external_test.json").write_text(json.dumps(out, indent=2))
    with open(RESULTS / "external_test.csv", "w") as fh:
        fh.write("file,label,copy_of_training_scan,max_similarity," + ",".join(f"p_pcos_{k}" for k in probs) + "\n")
        for i, f in enumerate(files):
            fh.write(f"\"{f}\",{y[i]},{seen[i]},{max_sim[i]:.3f}," + ",".join(f"{probs[k][i]:.3f}" for k in probs) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
