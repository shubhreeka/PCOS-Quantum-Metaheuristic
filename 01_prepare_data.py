"""Step 1: leak-free dataset preparation.

The public PCOS ultrasound dataset ships with a train/ and a test/ folder, but 1,920 of the
1,922 test files are byte-identical copies of training files, so the "test" score measures
memorisation. Many of the remaining images are also augmented copies (rotations, flips, zooms)
of one another.

This script:
  1. pools train/ and test/, drops unreadable files and exact duplicates (MD5),
  2. finds candidate look-alikes: the 20 nearest neighbours by CNN embedding (flip-averaged),
  3. verifies them geometrically (SIFT + RANSAC homography, also against the mirror image),
  4. treats each connected component as one "scan group",
  5. saves the group of every image, used for leave-one-scan-out evaluation.

Outputs data/processed/{images.npy, labels.npy, groups.npy, emb.npy, files.json},
results/data_split.json and results/scan_groups.jpg (8 random images per group).
"""
import hashlib
import json

import numpy as np
from PIL import Image, ImageOps
from scipy.sparse.csgraph import connected_components

from src.config import ARTIFACTS, CLASSES, DATA_ROOT, IMG_SIZE, RESULTS

K_NEIGHBOURS = 20
MIN_INLIERS = 12
SURE_SIMILARITY = 0.95


def load_all():
    seen, files, labels, images = {}, [], [], []
    stats = {"files": 0, "unreadable": 0, "exact_duplicates": 0, "label_conflicts": 0}
    for split in ("train", "test"):
        for label, cls in enumerate(CLASSES):
            folder = DATA_ROOT / split / cls
            for f in sorted(folder.iterdir()):
                if f.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                    continue
                stats["files"] += 1
                raw = f.read_bytes()
                h = hashlib.md5(raw).hexdigest()
                if h in seen:
                    stats["exact_duplicates"] += 1
                    if labels[seen[h]] != label:
                        stats["label_conflicts"] += 1
                    continue
                try:
                    img = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
                except Exception:
                    stats["unreadable"] += 1
                    continue
                seen[h] = len(images)
                images.append(np.asarray(img.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC), dtype=np.uint8))
                labels.append(label)
                files.append(f"{split}/{cls}/{f.name}")
    return np.stack(images), np.array(labels), files, stats


def embed(images):
    import torch
    from torchvision import models

    net = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT).features.eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    out = []
    with torch.no_grad():
        for i in range(0, len(images), 64):
            x = torch.tensor(images[i:i + 64]).permute(0, 3, 1, 2).float() / 255
            x = (x - mean) / std
            e = net(x).mean((2, 3)) + net(torch.flip(x, dims=[3])).mean((2, 3))
            out.append(e.numpy())
    e = np.concatenate(out)
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def keypoint_features(images):
    import cv2

    sift = cv2.SIFT_create(nfeatures=600)
    feats = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        feats.append((sift.detectAndCompute(gray, None),
                      sift.detectAndCompute(np.ascontiguousarray(gray[:, ::-1]), None)))
    return feats


def inliers(fa, fb):
    import cv2

    (ka, da), _ = fa
    best = 0
    for kb, db in fb:
        if da is None or db is None or len(da) < 8 or len(db) < 8:
            continue
        matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
        good = [m for m, n in (p for p in matches if len(p) == 2) if m.distance < 0.75 * n.distance]
        if len(good) < 8:
            continue
        src = np.float32([ka[m.queryIdx].pt for m in good])
        dst = np.float32([kb[m.trainIdx].pt for m in good])
        _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if mask is not None:
            best = max(best, int(mask.sum()))
    return best


def save_montage(images, groups, per_group=8):
    rng = np.random.default_rng(1)
    rows = []
    for g in range(groups.max() + 1):
        idx = np.where(groups == g)[0]
        idx = rng.choice(idx, min(per_group, len(idx)), replace=False)
        tiles = [np.asarray(Image.fromarray(images[i]).resize((112, 112))) for i in idx]
        tiles += [np.zeros((112, 112, 3), np.uint8)] * (per_group - len(tiles))
        rows.append(np.hstack(tiles))
    Image.fromarray(np.vstack(rows)).save(RESULTS / "scan_groups.jpg", quality=85)


def main():
    images, labels, files, stats = load_all()
    print(f"Files: {stats['files']}  unreadable: {stats['unreadable']}  exact duplicates removed: "
          f"{stats['exact_duplicates']}  (label conflicts: {stats['label_conflicts']})")
    print(f"Unique images: {len(images)}  (no PCOS={np.sum(labels == 0)}, PCOS={np.sum(labels == 1)})")

    emb = embed(images)
    sim = emb @ emb.T
    np.fill_diagonal(sim, 0)
    feats = keypoint_features(images)

    n = len(images)
    adj = np.zeros((n, n), dtype=bool)
    checked, pair_log = set(), []
    for i in range(n):
        for j in np.argsort(-sim[i])[:K_NEIGHBOURS]:
            a, b = min(i, j), max(i, j)
            if (a, b) in checked:
                continue
            checked.add((a, b))
            k = inliers(feats[a], feats[b])
            pair_log.append((a, b, float(sim[a, b]), k))
            if sim[a, b] >= SURE_SIMILARITY or k >= MIN_INLIERS:
                adj[a, b] = adj[b, a] = True
    np.save(ARTIFACTS / "pair_log.npy", np.array(pair_log))
    print(f"Checked {len(checked)} candidate pairs, {int(adj.sum() // 2)} confirmed near-duplicates")

    n_groups, groups = connected_components(adj, directed=False)
    sizes = np.bincount(groups)
    print(f"Groups: {n_groups}  (largest = {sizes.max()} images, {np.sum(sizes > 1)} groups have copies)")
    mixed = [g for g in range(n_groups) if len(set(labels[groups == g])) > 1]
    print(f"Groups containing both labels: {len(mixed)}")

    # Each group is one source scan. With so few scans, a fixed train/val/test split is not
    # meaningful (a group-wise split leaves one class out of val/test), so later steps use
    # leave-one-scan-out cross-validation over these groups.
    for g in range(n_groups):
        print(f"  scan {g}: {sizes[g]:4d} images  label={CLASSES[labels[groups == g][0]]}")
    save_montage(images, groups)

    for name, arr in [("images", images), ("labels", labels), ("groups", groups), ("emb", emb)]:
        np.save(ARTIFACTS / f"{name}.npy", arr)
    (ARTIFACTS / "files.json").write_text(json.dumps(files))
    out = dict(stats, unique_images=int(n), n_groups=int(n_groups), largest_group=int(sizes.max()),
               groups_with_copies=int(np.sum(sizes > 1)), mixed_label_groups=len(mixed),
               k_neighbours=K_NEIGHBOURS, min_inliers=MIN_INLIERS, sure_similarity=SURE_SIMILARITY,
               scans=[{"scan": g, "images": int(sizes[g]), "label": CLASSES[int(labels[groups == g][0])]}
                      for g in range(n_groups)])
    (RESULTS / "data_split.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
