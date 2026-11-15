"""Step 4: train the models served by the web app on all 8 scans.

Same recipes as one LOSO fold (step 3), but using every scan:
  * quantum classifier: logistic regression on the 256 quanvolutional features
  * metaheuristic branch: each of PSO / GWO / WOA / GA / FA selects 20 of the 256 quantum features
    (group-aware CV fitness), then a logistic regression is fitted on them
  * hybrid fusion MLP: all 256 quantum features + the 20 MobileNetV2 features chosen by the selector
    with the best CV fitness, trained with the same 5 seeds as in step 3

The metrics shown in the app come from the leave-one-scan-out evaluation (results/loso_results.json),
never from training data.

Outputs: models/{quantum,PSO,GWO,WOA,GA,FA}_clf.joblib, models/masks.npz (quantum-feature masks),
         models/hybrid.pt, results/comparative_metrics.json
"""
import json

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import ARTIFACTS, MODELS, RESULTS, SEED
from src.hybrid_model import fit_hybrid
from src.metaheuristic_opt import ALGORITHMS, run_selector

K, BUDGET, C = 20, 300, 0.1


def logreg():
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))


def main():
    f = np.load(ARTIFACTS / "features.npz")
    y, groups = np.load(ARTIFACTS / "labels.npy"), np.load(ARTIFACTS / "groups.npy")
    Xq, Xc = f["quantum"], f["classical"]
    joblib.dump(logreg().fit(Xq, y), MODELS / "quantum_clf.joblib")

    masks = {}
    for a in ALGORITHMS:
        mask, fit, _, _ = run_selector(a, Xq, y, groups, k=K, budget=BUDGET, seed=SEED)
        masks[a] = mask
        joblib.dump(logreg().fit(Xq[:, mask], y), MODELS / f"{a}_clf.joblib")
        print(f"{a} (quantum pool): cv_fitness={fit:.3f}")
    np.savez(MODELS / "masks.npz", **masks)

    fitness, c_masks = {}, {}
    for a in ALGORITHMS:
        c_masks[a], fitness[a], _, _ = run_selector(a, Xc, y, groups, k=K, budget=BUDGET, seed=SEED)
    best = max(fitness, key=fitness.get)
    everything = np.ones(len(y), dtype=bool)
    members = []
    for seed in [0, 1, 2, 3, 4]:
        model, stats = fit_hybrid(Xq, Xc[:, c_masks[best]], y, everything, seed)
        members.append({"state_dict": model.state_dict(), **{k: torch.tensor(v) for k, v in stats.items()}})
    torch.save({"selector": best, "classical_mask": torch.tensor(c_masks[best]), "members": members},
               MODELS / "hybrid.pt")
    print(f"hybrid: quantum + {best}-selected MobileNetV2 features, 5 seeds")

    loso = json.loads((RESULTS / "loso_results.json").read_text())
    app = {"quantum": loso["quantum (256)"], "hybrid": loso["Hybrid MLP: quantum + best selection"],
           **{a: loso[f"Quantum + {a} selection (20)"] for a in ALGORITHMS}}
    (RESULTS / "comparative_metrics.json").write_text(json.dumps(app, indent=2))


if __name__ == "__main__":
    main()
