"""Checks the NumPy simulator against the PennyLane reference circuit (needs pennylane)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.quantum_branch import make_pennylane_circuit, quanv_numpy  # noqa: E402


def test_matches_pennylane():
    rng = np.random.default_rng(0)
    circuit = make_pennylane_circuit()
    for patch in rng.uniform(0, np.pi, size=(200, 4)):
        ref = np.array(circuit(patch), dtype=float)
        assert np.allclose(ref, quanv_numpy(patch), atol=1e-10), (patch, ref, quanv_numpy(patch))




def test_closed_form():
    """Product-state input + CNOT ring => each <Z_q> is a product of cosines (Heisenberg picture)."""
    th = np.random.default_rng(1).uniform(0, np.pi, size=(500, 4))
    c = np.cos(th)
    expected = np.stack([c[:, 1] * c[:, 2] * c[:, 3],
                         c[:, 0] * c[:, 1],
                         c[:, 0] * c[:, 1] * c[:, 2],
                         c.prod(1)], axis=1)
    assert np.allclose(quanv_numpy(th), expected)


if __name__ == "__main__":
    test_matches_pennylane()
    test_closed_form()
    print("OK: NumPy simulator == PennyLane circuit == closed form")
