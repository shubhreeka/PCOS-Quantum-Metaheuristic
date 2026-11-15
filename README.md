# PCOS-Quantum-Metaheuristic: Hybrid Quantum-Classical PCOS Detection from Ultrasound

Detects **polycystic ovary syndrome (PCOS)** in ovarian ultrasound images by combining a **4-qubit quanvolutional
(quantum convolution) feature extractor**, a **MobileNetV2** transfer-learning backbone and **five metaheuristic
feature selectors** (PSO, GWO, WOA, GA, Firefly). It ships with a Flask web app that compares the quantum and
classical branches side by side.

The project also includes a **data audit**. The popular public dataset turns out to contain only **8 distinct scans**,
so the evaluation here is built around that fact.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![PennyLane](https://img.shields.io/badge/Quantum-PennyLane%20%2F%20exact%20NumPy%20sim-6f42c1)
![Metaheuristics](https://img.shields.io/badge/PSO%20·%20GWO%20·%20WOA%20·%20GA%20·%20FA-feature%20selection-2ea44f)

![Leave-one-scan-out results](results/loso_per_scan.png)

---

## Headline results

| Evaluation protocol | Quantum features only | Hybrid quantum + classical |
|---|---|---|
| Random image-level split (5-fold CV) | **100%** | **100%** |
| **Unseen scans** (leave-one-scan-out) | **92.1%** (AUC 0.976, 8/8 scans) | **97.5%** (AUC 0.998, 8/8 scans) |

The **unseen-scan** row is the fair estimate of how the model handles a new scan. In the random split, other
augmented copies of every test scan are in the training data (see the audit below).

## Key findings

1. **The dataset is much smaller than it looks.** Of its 3,855 files, 1,925 are byte-identical copies (1,920 of the
   1,922 "test" images also appear in "train") and 9 are empty. The remaining 1,921 images are augmented copies
   (rotated, shifted, zoomed, flipped) of **just 8 original scans: 4 normal and 4 PCOS**. So every model here is
   evaluated **leave-one-scan-out**: train on 7 scans, test on the unseen 8th, eight times.
2. **The hybrid quantum-classical model generalises to unseen scans: 97.5% accuracy, ROC-AUC 0.998**, with all 8
   scans classified correctly (96.5 ± 0.9% across 5 random seeds).
3. **The quantum circuit is what makes it work.** The same network without the quantum branch gets 74.8%. With the
   same 64×64 input and pooling, the quanvolutional features reach **92.1%**, while raw pixels get 62.6% and the
   circuit **without its entangling CNOTs** gets 57.5%.
4. **Metaheuristic feature selection overfits with this little data.** Selecting 20 of 1,280 features reaches
   ~99% on the training scans, but on unseen scans it does no better than 20 random features. With only 7 training
   scans there isn't enough signal to pick features reliably. In the hybrid, the quantum branch makes up for this.

## Pipeline

```
Ultrasound ─► CLAHE contrast enhancement ─┬─► 64×64 gray ─► 4-qubit quanvolution (2×2 patches) ─► 256 quantum features ─┐
                                          │                                                                             ├─► fusion MLP ─► PCOS / normal
                                          └─► 224×224 ─► MobileNetV2 (ImageNet) ─► 1,280 features ─► PSO/GWO/WOA/GA/FA ─┘
                                                                                                     select 20
```

| Step | Script | What it does |
|---|---|---|
| 1 | [`01_prepare_data.py`](01_prepare_data.py) | Removes exact duplicates (MD5), then finds augmented copies with CNN-embedding neighbours + **SIFT/RANSAC** geometric verification, and groups them into source scans |
| 2 | [`02_extract_features.py`](02_extract_features.py) | CLAHE, then MobileNetV2 features, quanvolutional features and two classical controls |
| 3 | [`03_loso_evaluation.py`](03_loso_evaluation.py) | **Leave-one-scan-out** evaluation of every model; feature selection runs inside each fold |
| 4 | [`04_train_final.py`](04_train_final.py) | Trains the models served by the web app on all scans |
| 5 | [`05_figures.py`](05_figures.py) | Per-scan results figure |
| 6 | [`06_external_test.py`](06_external_test.py) | Scores a folder of outside ultrasound images, after checking that none is a copy of a training scan |
| 7 | [`07_quantum_ablation.py`](07_quantum_ablation.py) | Sensitivity of the quantum branch to resolution, pooling and regularisation |

### The dataset audit

![The 8 source scans](results/scan_groups.jpg)
*Each row shows 8 random images from one group found by step 1: every group is one ultrasound scan, augmented
99–391 times. Rows 1–4 are normal and rows 5–8 are PCOS.*

Every candidate pair of look-alike images was checked with SIFT keypoints and a RANSAC homography. Matching copies
share a median of **285 geometrically consistent keypoints**, far above chance. No group mixes the two labels.

## Quantum feature extractor

[`src/quantum_branch.py`](src/quantum_branch.py) implements a **quanvolutional layer** (Henderson et al., 2019). Each
2×2 pixel patch is angle-encoded into 4 qubits with `RY(θ)`, entangled by a ring of CNOTs, and measured as `⟨Z⟩` on
every qubit. Sliding it over a 64×64 image gives a 4-channel 32×32 map, which is pooled to 8×8 → 256 features.

- The reference circuit is written in **PennyLane**. For speed, the same circuit is simulated **exactly with
  vectorised NumPy**, which takes seconds on CPU for the whole dataset instead of hours.
  [`tests/test_quantum_equivalence.py`](tests/test_quantum_equivalence.py) checks that both give identical outputs.
- Because the input is a product state and the CNOTs only permute basis states, the circuit has a **closed form**
  (verified in the same test):

  `⟨Z₀⟩ = cos θ₁ cos θ₂ cos θ₃`, `⟨Z₁⟩ = cos θ₀ cos θ₁`, `⟨Z₂⟩ = cos θ₀ cos θ₁ cos θ₂`, `⟨Z₃⟩ = cos θ₀ cos θ₁ cos θ₂ cos θ₃`

  The entanglement is what creates the **products** of neighbouring pixels' cosines. Without the CNOTs, each output
  would be just `cos θ_q`, and that control gets only 57.5% on unseen scans vs 92.1% with entanglement. The circuit
  runs on a simulator; no quantum hardware is used.

**Sensitivity** ([`07_quantum_ablation.py`](07_quantum_ablation.py), quantum features + logistic regression,
leave-one-scan-out). The original 32×32 / 4×4 setting gave 79.4%. Higher resolution helps consistently, and the
result is stable across regularisation strengths:

| Input | Pooling | Features | Accuracy | ROC-AUC | Scans correct |
|---|---|---|---|---|---|
| 32×32 | 4×4 | 64 | 79.4% | 0.884 | 7/8 |
| 32×32 | 8×8 | 256 | 82.2% | 0.900 | 8/8 |
| 64×64 | 4×4 | 64 | 82.5% | 0.908 | 7/8 |
| **64×64** | **8×8** | **256** | **92.1%** | **0.976** | **8/8** |
| 64×64 | 16×16 | 1,024 | 92.3% | 0.981 | 8/8 |
| 128×128 | 8×8 | 256 | 94.6% | 0.984 | 8/8 |

With 64×64 / 8×8, varying the regularisation C from 0.01 to 1.0 gives 89.0–93.8%, always 8/8 scans. The default
(64×64, 8×8, C = 0.1) was chosen after this comparison, so the whole table is shown, not just the best row.

## Metaheuristic feature selection

[`src/metaheuristic_opt.py`](src/metaheuristic_opt.py) implements PSO (Kennedy & Eberhart, 1995), GWO
(Mirjalili et al., 2014), WOA (Mirjalili & Lewis, 2016), GA with tournament selection and elitism, and Firefly
(Yang, 2008) as **wrapper feature selectors**. Each agent scores all 1,280 features and keeps the top 20.

- **Fitness:** balanced accuracy of logistic regression in **group-aware 3-fold CV over the training scans only**.
- **Equal budget:** 300 unique fitness evaluations per algorithm.

![Convergence](results/convergence.png)

*GA and GWO reach the highest training fitness. FA's curve is short because each FA iteration compares every pair of
fireflies, so it uses its budget in few iterations.*

## Results: leave-one-scan-out (8 folds, 1,921 images)

| Model | Features | Accuracy | Balanced acc. | ROC-AUC | Unseen scans correct* |
|---|---|---|---|---|---|
| *Random image-level 5-fold CV (leaky, for comparison)* | *1,280* | *100%* | *100%* | *1.000* | *8/8* |
| MobileNetV2, all features + logistic regression | 1,280 | 84.4% | 83.4% | 0.915 | 6/8 |
| MobileNetV2, 20 random features | 20 | 81.9% | 80.3% | 0.875 | 6/8 |
| MobileNetV2 + PSO selection | 20 | 73.7% | 72.6% | 0.811 | 6/8 |
| MobileNetV2 + GWO selection | 20 | 77.4% | 74.3% | 0.759 | 6/8 |
| MobileNetV2 + WOA selection | 20 | 67.9% | 67.5% | 0.732 | 6/8 |
| MobileNetV2 + GA selection | 20 | 71.3% | 72.3% | 0.838 | 5/8 |
| MobileNetV2 + Firefly selection | 20 | 63.6% | 65.2% | 0.754 | 4/8 |
| Control: raw 2×2 patch pixels | 256 | 62.6% | 61.4% | 0.658 | 5/8 |
| Control: circuit without entanglement, cos(θ) | 256 | 57.5% | 56.8% | 0.586 | 4/8 |
| **Quantum (quanvolution) features** | 256 | **92.1%** | **91.3%** | **0.976** | **8/8** |
| Fusion MLP **without** quantum branch | 20 | 74.8% | 75.0% | 0.837 | 6/8 |
| **Hybrid fusion MLP: quantum + selected classical** | 276 | **97.5%** | **97.3%** | **0.998** | **8/8** |

\*A scan counts as correct when the mean predicted probability over all its augmented copies is on the right side
of 0.5. The MLP rows average 5 random seeds (single seeds: hybrid 95.1–97.5%, without quantum 71.2–77.3%). Per-scan numbers are in [`results/loso_per_scan.csv`](results/loso_per_scan.csv).

**How to read these numbers:** with only 8 independent scans, every figure has a wide confidence interval. They show
that the hybrid model transfers to scans it has never seen, but they are not a clinical accuracy estimate.

### Note on earlier results
An earlier version of this project reported **98.9%** (quantum branch) on the dataset's own `test/` folder. That
folder is a copy of `train/`, so the comparable figure today is the random-split row above (100%). This version adds
the stricter unseen-scan evaluation, on which the hybrid model reaches 97.5%.

## External test (your own images)

```bash
python 06_external_test.py path/to/external     # folders: pcos/ and normal/
```
The script first checks every image against the training dataset with the same SIFT/RANSAC test used in step 1.
Many PCOS ultrasounds online come from this same public dataset, so those copies are reported separately and
excluded. It then reports accuracy on the genuinely new images (`results/external_test.json`).

## Web app

```bash
python app.py        # http://127.0.0.1:5000
```
![Web app](results/app_screenshot.jpg)

Upload an ultrasound and choose a selector (PSO / GWO / WOA / GA / FA). The app shows:
- the **final diagnosis from the hybrid model** (97.5% on unseen scans), an ensemble of the 5 seeds
- the quantum and the metaheuristic branch predictions on their own, with confidence
- leave-one-scan-out metrics for each model
- a rough outline of dark, round regions (often follicles) on positive cases

The outlines come from **classical image processing (a heuristic visual aid)**, not from the model. Non-grayscale
or blank images are rejected.

## Reproduce

```bash
git clone https://github.com/shubhreeka/PCOS-Quantum-Metaheuristic.git && cd PCOS-Quantum-Metaheuristic
python -m venv venv && venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt

# Dataset: "PCOS Detection using Ultrasound Images" (Kaggle), with train/ and test/ folders,
# each containing infected/ and notinfected/
set PCOS_DATA_ROOT=path\to\data                        # macOS/Linux: export PCOS_DATA_ROOT=...

python 01_prepare_data.py       # ~5 min on CPU (SIFT verification)
python 02_extract_features.py   # <1 min
python 03_loso_evaluation.py    # ~15 min
python 04_train_final.py
python 05_figures.py
python 07_quantum_ablation.py   # ~2 min
python tests/test_quantum_equivalence.py   # needs pennylane
```
The dataset and trained models are not stored in this repository.

## Limitations & future work
- **Only 8 independent scans.** External validation on a larger, multi-centre dataset with patient-level metadata is
  the most important next step.
- The quantum circuit is fixed (not trained) and simulated classically. Next steps: trainable variational circuits
  and a run on real quantum hardware.
- Feature selection needs more independent scans to beat random selection. Group-aware stability selection is a
  natural next step.
- **Research prototype, not a medical device.** Predictions must not be used for diagnosis.

## Author

**Shubhreeka Saxena**, B.Tech AI & ML, VIPS-TC (GGSIPU)
[LinkedIn](https://www.linkedin.com/in/shubhreeka-saxena/) · [GitHub](https://github.com/shubhreeka)
