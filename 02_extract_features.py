"""Step 2: feature extraction (all splits), after CLAHE contrast enhancement.

  classical : MobileNetV2 (ImageNet, frozen) global-average-pooled features      -> 1280-d
  quantum   : 4-qubit quanvolution on the 32x32 grayscale image, pooled to 4x4    -> 64-d
  control   : the same 32x32 input, the same 2x2/stride-2 patching and 4x4 pooling,
              but the raw patch pixels instead of the quantum circuit             -> 64-d
              (isolates what the quantum transform itself adds)

Output: data/processed/features.npz
"""
import numpy as np
import torch
import torch.nn.functional as F

from src.config import ARTIFACTS, Q_SIZE
from src.dataset import clahe_rgb
from src.quantum_branch import QuantumFeatureExtractor

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_backbone():
    from torchvision import models

    return models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT).features.eval()


def classical_features(images, net=None):
    net = net or load_backbone()
    out = []
    with torch.no_grad():
        for i in range(0, len(images), 64):
            x = torch.tensor(images[i:i + 64]).permute(0, 3, 1, 2).float() / 255
            out.append(net((x - MEAN) / STD).mean((2, 3)).numpy())
    return np.concatenate(out)


def small_gray(images):
    """(N, H, W, 3) uint8 -> (N, 1, 32, 32) angles in [0, pi]."""
    x = torch.tensor(images).permute(0, 3, 1, 2).float().mean(1, keepdim=True) / 255
    return F.interpolate(x, size=(Q_SIZE, Q_SIZE), mode="bilinear", antialias=True, align_corners=False) * np.pi


def control_features(q):
    patches = F.pixel_unshuffle(q, 2)          # (N, 4, 16, 16): the 4 pixels of every 2x2 patch
    return F.adaptive_avg_pool2d(patches, 4).flatten(1).numpy()


def main():
    images = np.load(ARTIFACTS / "images.npy")
    enhanced = np.stack([clahe_rgb(im) for im in images])
    q = small_gray(enhanced)
    feats = {"classical": classical_features(enhanced),
             "quantum": QuantumFeatureExtractor()(q).numpy(),
             "control": control_features(q)}
    np.savez(ARTIFACTS / "features.npz", **feats)
    print({k: v.shape for k, v in feats.items()})


if __name__ == "__main__":
    main()
