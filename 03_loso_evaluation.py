"""Step 3: leave-one-scan-out (LOSO) evaluation of every model.

The dataset contains only 8 distinct ultrasound scans (see step 1). The only honest test of
generalisation is therefore: train on 7 scans (all their augmented copies), predict every copy of
the 8th, unseen scan, and repeat 8 times. Everything is fitted inside each fold:
  * standardisation and logistic-regression weights (C fixed at 0.1, there is no data to tune it)
  * metaheuristic feature selection: k=20 of 1,280 MobileNetV2 features, fitness = group-aware
    3-fold CV balanced accuracy over the 7 training scans, 300 evaluations per algorithm
  * hybrid MLP (quantum 256 + selected classical 20), fixed number of epochs, averaged over 5 seeds

For contrast, the same classifier is also scored with a random image-level 5-fold CV, which is
how the original 98.9% figure arises.

Outputs: results/loso_results.json, results/loso_per_scan.csv, results/convergence.png
"""
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import ARTIFACTS, RESULTS, SEED
from src.hybrid_model import fit_hybrid, predict_hybrid
from src.metaheuristic_opt import ALGORITHMS, run_selector

K, BUDGET, C = 20, 300, 0.1
MLP_EPOCHS = 60
MLP_SEEDS = [0, 1, 2, 3, 4]


def logreg():
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))


def mlp_predict(Xq, Xc, y, tr, te, seed=SEED):
    model, stats = fit_hybrid(Xq, Xc, y, tr, seed, epochs=MLP_EPOCHS)
    return predict_hybrid(model, stats, Xq[te], Xc[te])


def summarise(y, prob, groups, n_features=None):
    pred = (prob >= 0.5).astype(int)
    per_scan = {int(g): float(np.mean(pred[groups == g] == y[groups == g])) for g in np.unique(groups)}
    scan_votes = [int(np.mean(prob[groups == g]) >= 0.5) == y[groups == g][0] for g in np.unique(groups)]
    return {"image_accuracy": round(float(accuracy_score(y, pred)), 4),
            "image_balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
            "image_roc_auc": round(float(roc_auc_score(y, prob)), 4),
            "scans_correct": f"{int(np.sum(scan_votes))}/{len(scan_votes)}",
            "per_scan_accuracy": {g: round(a, 3) for g, a in per_scan.items()},
            # keys read by the web app
            "accuracy": round(float(accuracy_score(y, pred)), 4),
            "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y, pred)), 4),
            "f1": round(float(f1_score(y, pred)), 4),
            "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
            "selected_features": n_features}


def main():
    f = np.load(ARTIFACTS / "features.npz")
    y, groups = np.load(ARTIFACTS / "labels.npy"), np.load(ARTIFACTS / "groups.npy")
    Xc, Xq, Xctl = f["classical"], f["quantum"], f["control"]
    scans = np.unique(groups)

    fixed_sets = {"quantum (256)": Xq, "classical control: raw pixels (256)": Xctl,
                  "classical control: cos(pixel), no entanglement (256)": f["cos_control"],
                  "MobileNetV2, all features (1280)": Xc}
    names = list(fixed_sets) + ["MobileNetV2, 20 random features"] + \
        [f"MobileNetV2 + {a} selection (20)" for a in ALGORITHMS] + \
        ["Hybrid MLP: quantum + best selection", "Same MLP without quantum"]
    probs = {n: np.zeros(len(y)) for n in names}
    convergence = {a: [] for a in ALGORITHMS}
    picked = {a: [] for a in ALGORITHMS}
    rng = np.random.default_rng(SEED)
    best_per_fold = []
    seed_probs = {"Hybrid MLP: quantum + best selection": [], "Same MLP without quantum": []}

    for s in scans:
        tr, te = groups != s, groups == s
        print(f"\n--- held-out scan {s} ({te.sum()} images, label={y[te][0]}) ---", flush=True)
        for n, X in fixed_sets.items():
            probs[n][te] = logreg().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        rand = [rng.permutation(Xc.shape[1])[:K] for _ in range(10)]
        probs["MobileNetV2, 20 random features"][te] = np.mean(
            [logreg().fit(Xc[tr][:, m], y[tr]).predict_proba(Xc[te][:, m])[:, 1] for m in rand], axis=0)

        best_algo, best_fit = None, -1
        for a in ALGORITHMS:
            mask, fit, hist, evals = run_selector(a, Xc[tr], y[tr], groups[tr], k=K, budget=BUDGET, seed=SEED)
            convergence[a].append(hist)
            picked[a].append(np.where(mask)[0].tolist())
            probs[f"MobileNetV2 + {a} selection (20)"][te] = \
                logreg().fit(Xc[tr][:, mask], y[tr]).predict_proba(Xc[te][:, mask])[:, 1]
            if fit > best_fit:
                best_algo, best_fit, best_mask = a, fit, mask
            print(f"  {a:4s} cv_fitness={fit:.3f} evals={evals}", flush=True)
        best_per_fold.append(best_algo)
        for name, Xq_in in [("Hybrid MLP: quantum + best selection", Xq),
                            ("Same MLP without quantum", np.zeros((len(y), 0)))]:
            runs = [mlp_predict(Xq_in, Xc[:, best_mask], y, tr, te, seed=sd) for sd in MLP_SEEDS]
            seed_probs[name].append(runs)
            probs[name][te] = np.mean(runs, axis=0)   # average of 5 seeds
        print(f"  best selector on training CV: {best_algo}", flush=True)
        for n in names:
            print(f"  {n:40s} acc on unseen scan = {np.mean((probs[n][te] >= .5) == y[te]):.3f}", flush=True)

    n_feats = {n: X.shape[1] for n, X in fixed_sets.items()}
    results = {n: summarise(y, probs[n], groups, n_feats.get(n, K)) for n in names}
    for n, folds in seed_probs.items():   # spread across individual seeds
        accs = []
        for i in range(len(MLP_SEEDS)):
            p = np.zeros(len(y))
            for s, runs in zip(scans, folds):
                p[groups == s] = runs[i]
            accs.append(accuracy_score(y, p >= .5))
        results[n]["image_accuracy_per_seed"] = [round(float(a), 4) for a in accs]
    results["_best_selector_per_fold"] = best_per_fold
    random_cv = cross_val_predict(logreg(), Xc, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                                  method="predict_proba")[:, 1]
    results["(leaky) random image-level 5-fold CV, all features"] = summarise(y, random_cv, groups)
    random_cv_q = cross_val_predict(logreg(), Xq, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                                    method="predict_proba")[:, 1]
    results["(leaky) random image-level 5-fold CV, quantum"] = summarise(y, random_cv_q, groups)

    (RESULTS / "loso_results.json").write_text(json.dumps(results, indent=2))
    with open(RESULTS / "loso_per_scan.csv", "w") as fh:
        fh.write("model," + ",".join(f"scan_{s}" for s in scans) + ",image_accuracy,scans_correct\n")
        for n, r in results.items():
            if n.startswith("_"):
                continue
            fh.write(f"\"{n}\"," + ",".join(str(r["per_scan_accuracy"][int(s)]) for s in scans)
                     + f",{r['image_accuracy']},{r['scans_correct']}\n")
    (RESULTS / "selected_features_per_fold.json").write_text(json.dumps(picked))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6.5, 4))
    for a, hs in convergence.items():
        L = min(len(h) for h in hs)
        plt.plot(np.mean([h[:L] for h in hs], axis=0), label=a, lw=2)
    plt.xlabel("Iteration")
    plt.ylabel("Best inner-CV balanced accuracy")
    plt.title(f"Feature-selection convergence (mean of 8 folds, {BUDGET} evals each)")
    plt.legend()
    plt.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(RESULTS / "convergence.png", dpi=150)

    print("\n=== SUMMARY (leave-one-scan-out) ===")
    for n, r in results.items():
        if n.startswith("_"):
            continue
        print(f"{n:52s} img_acc={r['image_accuracy']:.3f} bal={r['image_balanced_accuracy']:.3f} "
              f"auc={r['image_roc_auc']:.3f} scans={r['scans_correct']}")


if __name__ == "__main__":
    main()
