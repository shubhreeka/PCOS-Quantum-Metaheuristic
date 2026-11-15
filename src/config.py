"""Shared paths and constants."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Folder that contains train/ and test/ sub-folders, each with infected/ and notinfected/
DATA_ROOT = Path(os.environ.get("PCOS_DATA_ROOT", ROOT / "data"))
ARTIFACTS = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
for _d in (ARTIFACTS, RESULTS, MODELS):
    _d.mkdir(parents=True, exist_ok=True)

CLASSES = ["notinfected", "infected"]   # label 0 = no PCOS, 1 = PCOS
IMG_SIZE = 224        # classical branch input
Q_SIZE = 64           # quantum branch input (grayscale)
Q_POOL = 8            # quanvolution maps are average-pooled to Q_POOL x Q_POOL
SEED = 42
