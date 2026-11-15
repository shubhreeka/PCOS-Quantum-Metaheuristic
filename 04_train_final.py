"""Step 4: train the models served by the web app on all 8 scans.

Same recipe as one LOSO fold (step 3), but using every scan: each metaheuristic selects k=20
MobileNetV2 features with group-aware CV, then a logistic regression is fitted. A quantum-only
classifier is trained on the 64 quanvolutional features.

The metrics shown in the app come from the leave-one-scan-out evaluation (results/loso_results.json),
never from training data.

Outputs: models/{quantum,PSO,GWO,WOA,GA,FA}_clf.joblib, models/masks.npz,
         results/comparative_metrics.json
"""
import json

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import ARTIFACTS, MODELS, RESULTS, SEED
from src.metaheuristic_opt import ALGORITHMS, run_selector

K, BUDGET, C = 20, 300, 0.1


def logreg():
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))


def main():
    f = np.load(ARTIFACTS / "features.npz")
    y, groups = np.load(ARTIFACTS / "labels.npy"), np.load(ARTIFACTS / "groups.npy")
    joblib.dump(logreg().fit(f["quantum"], y), MODELS / "quantum_clf.joblib")
    masks = {}
    for a in ALGORITHMS:
        mask, fit, _, _ = run_selector(a, f["classical"], y, groups, k=K, budget=BUDGET, seed=SEED)
        masks[a] = mask
        joblib.dump(logreg().fit(f["classical"][:, mask], y), MODELS / f"{a}_clf.joblib")
        print(f"{a}: cv_fitness={fit:.3f}")
    np.savez(MODELS / "masks.npz", **masks)

    loso = json.loads((RESULTS / "loso_results.json").read_text())
    app = {"quantum": loso["quantum (64)"],
           **{a: loso[f"MobileNetV2 + {a} selection (20)"] for a in ALGORITHMS}}
    (RESULTS / "comparative_metrics.json").write_text(json.dumps(app, indent=2))


if __name__ == "__main__":
    main()
