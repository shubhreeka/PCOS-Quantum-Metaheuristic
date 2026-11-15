"""Step 8: which candidate pool and fitness make metaheuristic feature selection transfer to unseen scans?

GA selects k=20 features inside each leave-one-scan-out fold; a logistic regression on the selected
features is then scored on the held-out scan. Compared:
  pools   : 1,280 MobileNetV2 features / MobileNetV2 + 256 quantum / 256 quantum
  fitness : group-aware 3-fold CV ("group3") / inner leave-one-scan-out probability ("loso_prob")

Output: results/selection_pool_study.json
"""
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import ARTIFACTS, RESULTS, SEED
from src.metaheuristic_opt import run_selector

K, BUDGET, ALGO = 20, 300, "GA"
VARIANTS = [("MobileNetV2 (1280)", "group3"), ("MobileNetV2 (1280)", "loso_prob"),
            ("MobileNetV2 + quantum (1536)", "group3"), ("MobileNetV2 + quantum (1536)", "loso_prob"),
            ("quantum (256)", "group3")]


def main():
    f = np.load(ARTIFACTS / "features.npz")
    y, g = np.load(ARTIFACTS / "labels.npy"), np.load(ARTIFACTS / "groups.npy")
    pools = {"MobileNetV2 (1280)": f["classical"], "quantum (256)": f["quantum"],
             "MobileNetV2 + quantum (1536)": np.hstack([f["classical"], f["quantum"]])}
    out = {}
    for pool, mode in VARIANTS:
        X, p = pools[pool], np.zeros(len(y))
        for s in np.unique(g):
            tr, te = g != s, g == s
            mask, *_ = run_selector(ALGO, X[tr], y[tr], g[tr], k=K, budget=BUDGET, seed=SEED, mode=mode)
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
            p[te] = clf.fit(X[tr][:, mask], y[tr]).predict_proba(X[te][:, mask])[:, 1]
        scans = sum(int(p[g == s].mean() >= .5) == y[g == s][0] for s in np.unique(g))
        key = f"{ALGO}, pool={pool}, fitness={mode}"
        out[key] = {"accuracy": round(float(np.mean((p >= .5) == y)), 4),
                    "roc_auc": round(float(roc_auc_score(y, p)), 4), "scans_correct": f"{scans}/8"}
        print(key, out[key], flush=True)
    (RESULTS / "selection_pool_study.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
