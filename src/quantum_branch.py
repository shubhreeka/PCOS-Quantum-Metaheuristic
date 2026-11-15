"""Quanvolutional (quantum convolution) feature extractor.

A 2x2 grayscale patch (pixel values scaled to [0, pi]) is angle-encoded into 4 qubits with RY
rotations, entangled with a ring of CNOTs, and measured as <Z> on every qubit
(Henderson et al., 2019, "Quanvolutional Neural Networks"). Sliding the circuit over a 32x32
image with stride 2 gives a 4-channel 16x16 map, average-pooled to 4x4 -> 64 features.

Two equivalent implementations are provided:
  * `quanv_circuit`   - the reference PennyLane QNode (needs `pennylane`)
  * `quanv_numpy`     - an exact, vectorised state-vector simulation in NumPy, ~10,000x faster on CPU.
`tests/test_quantum_equivalence.py` checks that both give identical outputs.
"""
import numpy as np
import torch
import torch.nn as nn

N_QUBITS = 4


# --------------------------------------------------------------------------- reference (PennyLane)
def make_pennylane_circuit():
    import pennylane as qml

    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev)
    def quanv_circuit(patch):
        for i in range(N_QUBITS):
            qml.RY(patch[i], wires=i)
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 3])
        qml.CNOT(wires=[3, 0])
        return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

    return quanv_circuit


# --------------------------------------------------------------------------- exact NumPy simulator
def _cnot_ring_permutation():
    """Basis-state permutation implemented by CNOT(0,1) CNOT(1,2) CNOT(2,3) CNOT(3,0).
    Wire 0 is the most significant bit (PennyLane convention)."""
    perm = np.arange(2 ** N_QUBITS)
    for c, t in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        cb, tb = N_QUBITS - 1 - c, N_QUBITS - 1 - t
        perm = np.array([s ^ (1 << tb) if (s >> cb) & 1 else s for s in perm])
    return perm


_PERM = _cnot_ring_permutation()
_BITS = np.array([[(s >> (N_QUBITS - 1 - q)) & 1 for q in range(N_QUBITS)] for s in range(2 ** N_QUBITS)])
_ZSIGN = 1 - 2 * _BITS[_PERM]          # (16 states, 4 qubits): sign of Z_q on the state each basis state is mapped to


def quanv_numpy(patches):
    """patches: (..., 4) angles. Returns (..., 4) expectation values <Z_q>, exactly as the circuit."""
    c, s = np.cos(patches / 2) ** 2, np.sin(patches / 2) ** 2          # P(qubit q = 0 / 1) after RY
    probs = np.ones(patches.shape[:-1] + (2 ** N_QUBITS,))
    for q in range(N_QUBITS):
        probs = probs * np.where(_BITS[:, q] == 0, c[..., q:q + 1], s[..., q:q + 1])
    return probs @ _ZSIGN


def quanv_image(img, kernel=2, stride=2):
    """img: (N, H, W) angles in [0, pi]. Returns (N, 4, H', W')."""
    n, h, w = img.shape
    oh, ow = (h - kernel) // stride + 1, (w - kernel) // stride + 1
    patches = np.stack([img[:, i:i + stride * oh:stride, j:j + stride * ow:stride]
                        for i in range(kernel) for j in range(kernel)], axis=-1)   # (N, oh, ow, 4)
    return np.moveaxis(quanv_numpy(patches), -1, 1)


class QuantumFeatureExtractor(nn.Module):
    """(N, 1, 32, 32) tensor of angles -> (N, 64) quanvolutional features."""

    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

    def forward(self, q_img):
        maps = quanv_image(q_img[:, 0].detach().cpu().numpy().astype(np.float64))
        return self.pool(torch.tensor(maps, dtype=torch.float32)).flatten(1)
