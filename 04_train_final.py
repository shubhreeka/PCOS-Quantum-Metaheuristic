"""Step 4: train the models served by the web app on all 8 scans.

Same recipe as one LOSO fold (step 3), but using every scan: each metaheuristic selects k=20
MobileNetV2 features with group-aware CV, then a logistic regression is fitted. A quantum-only
classifier is trained on the 256 quanvolutional features, and the hybrid fusion MLP (quantum +
features of the selector with the best CV fitness) is trained with the 5 seeds used in step 3.

The metrics shown in the app come from the leave-one-scan-out evaluation (results/loso_results.json),
never from training data.

Outputs: models/{quantum,PSO,GWO,WOA,GA,FA}_clf.joblib, models/masks.npz, models/hybrid.pt,
         results/comparative_metrics.json
"""
import json

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import torch

from src.config import ARTIFACTS, MODELS, RESULTS, SEED
from src.hybrid_model import fit_hybrid
from src.metaheuristic_opt import ALGORITHMS, run_selector

K, BUDGET, C = 20, 300, 0.1


def logreg():
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))


def main():
    f = np.load(ARTIFACTS / "features.npz")
    y, groups = np.load(ARTIFACTS / "labels.npy"), np.load(ARTIFACTS / "groups.npy")
    joblib.dump(logreg().fit(f["quantum"], y), MODELS / "quantum_clf.joblib")
    masks, fitness = {}, {}
    for a in ALGORITHMS:
        mask, fit, _, _ = run_selector(a, f["classical"], y, groups, k=K, budget=BUDGET, seed=SEED)
        masks[a], fitness[a] = mask, fit
        joblib.dump(logreg().fit(f["classical"][:, mask], y), MODELS / f"{a}_clf.joblib")
        print(f"{a}: cv_fitness={fit:.3f}")
    np.savez(MODELS / "masks.npz", **masks)

    best = max(fitness, key=fitness.get)
    everything = np.ones(len(y), dtype=bool)
    members = []
    for seed in [0, 1, 2, 3, 4]:
        model, stats = fit_hybrid(f["quantum"], f["classical"][:, masks[best]], y, everything, seed)
        members.append({"state_dict": model.state_dict(), **{k: torch.tensor(v) for k, v in stats.items()}})
    torch.save({"selector": best, "members": members}, MODELS / "hybrid.pt")
    print(f"hybrid: quantum + {best}-selected features, 5 seeds")

    loso = json.loads((RESULTS / "loso_results.json").read_text())
    app = {"quantum": loso["quantum (256)"], "hybrid": loso["Hybrid MLP: quantum + best selection"],
           **{a: loso[f"MobileNetV2 + {a} selection (20)"] for a in ALGORITHMS}}
    (RESULTS / "comparative_metrics.json").write_text(json.dumps(app, indent=2))


if __name__ == "__main__":
    main()
