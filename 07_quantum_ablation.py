"""Step 7: sensitivity of the quantum branch to input resolution, pooling and regularisation.

Every configuration is scored with the same leave-one-scan-out protocol (logistic regression).
The default (64x64 input, 8x8 pooling, C=0.1) was chosen after this comparison, so the table is
published in full: the result is not one lucky setting.

Output: results/quantum_ablation.json
"""
import json

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import ARTIFACTS, RESULTS
from src.dataset import clahe_rgb
from src.quantum_branch import quanv_image


def main():
    images = np.load(ARTIFACTS / "images.npy")
    y, groups = np.load(ARTIFACTS / "labels.npy"), np.load(ARTIFACTS / "groups.npy")
    gray = torch.tensor(np.stack([clahe_rgb(im) for im in images])).permute(0, 3, 1, 2).float().mean(1, True) / 255

    def angles(size):
        return F.interpolate(gray, size=(size, size), mode="bilinear", antialias=True, align_corners=False) * np.pi

    def quantum(size, pool):
        maps = torch.tensor(quanv_image(angles(size)[:, 0].numpy().astype(np.float64)), dtype=torch.float32)
        return F.adaptive_avg_pool2d(maps, pool).flatten(1).numpy()

    def loso(X, C):
        p = np.zeros(len(y))
        for s in np.unique(groups):
            tr, te = groups != s, groups == s
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=4000)).fit(X[tr], y[tr])
            p[te] = clf.predict_proba(X[te])[:, 1]
        scans = sum(int(p[groups == s].mean() >= .5) == y[groups == s][0] for s in np.unique(groups))
        return {"accuracy": round(float(np.mean((p >= .5) == y)), 4),
                "roc_auc": round(float(roc_auc_score(y, p)), 4), "scans_correct": f"{scans}/8"}

    rows = {}
    for size, pool in [(32, 4), (32, 8), (64, 4), (64, 8), (64, 16), (128, 8)]:
        rows[f"input {size}x{size}, pool {pool}x{pool}, C=0.1"] = loso(quantum(size, pool), 0.1)
    X = quantum(64, 8)
    for C in [0.01, 0.03, 0.3, 1.0]:
        rows[f"input 64x64, pool 8x8, C={C}"] = loso(X, C)
    for k, v in rows.items():
        print(f"{k:36s} {v}")
    (RESULTS / "quantum_ablation.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
