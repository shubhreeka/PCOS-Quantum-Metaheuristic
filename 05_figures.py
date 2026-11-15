"""Step 5: README figure - accuracy on each unseen scan (leave-one-scan-out)."""
import json

import matplotlib
import numpy as np

from src.config import RESULTS

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SHOW = [("(leaky) random image-level 5-fold CV, all features", "Random image split (leaky)", "#bbbbbb"),
        ("MobileNetV2, all features (1280)", "MobileNetV2, all features", "#4c78a8"),
        ("MobileNetV2 + GA selection (20)", "MobileNetV2 + GA-selected 20", "#72b7b2"),
        ("quantum (64)", "Quantum features only", "#b279a2"),
        ("Same MLP without quantum", "MLP without quantum", "#f58518"),
        ("Hybrid MLP: quantum + best selection", "Hybrid quantum + classical", "#54a24b")]


def main():
    r = json.loads((RESULTS / "loso_results.json").read_text())
    scans = sorted(r[SHOW[0][0]]["per_scan_accuracy"], key=int)
    labels = json.loads((RESULTS / "data_split.json").read_text())["scans"]
    names = [f"Scan {s}\n({'PCOS' if labels[int(s)]['label'] == 'infected' else 'normal'})" for s in scans]
    x, w = np.arange(len(scans)), 0.8 / len(SHOW)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for i, (key, lab, col) in enumerate(SHOW):
        acc = [r[key]["per_scan_accuracy"][s] for s in scans]
        ax.bar(x + (i - len(SHOW) / 2 + .5) * w, acc, w, label=f"{lab} ({r[key]['image_accuracy']:.0%})", color=col)
    ax.axhline(.5, color="k", lw=.8, ls="--")
    ax.set_xticks(x, names)
    ax.set_ylabel("Accuracy on the held-out scan")
    ax.set_ylim(0, 1.05)
    ax.set_title("Leave-one-scan-out: accuracy on each scan the model never saw")
    ax.legend(fontsize=8, ncol=3, loc="lower center", bbox_to_anchor=(.5, -.42))
    fig.tight_layout()
    fig.savefig(RESULTS / "loso_per_scan.png", dpi=150)


if __name__ == "__main__":
    main()
