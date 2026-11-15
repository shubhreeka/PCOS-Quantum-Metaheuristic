"""Wrapper feature selection with five population-based metaheuristics.

Each agent is a continuous vector with one score per feature; the `k` highest-scoring features
form the selected subset. Fitness = mean balanced accuracy of a logistic-regression classifier in
group-aware 3-fold cross-validation on the TRAINING split only (validation and test never seen).

Algorithms (all maximise fitness):
  PSO  Particle Swarm Optimisation, linearly decreasing inertia      (Kennedy & Eberhart, 1995)
  GWO  Grey Wolf Optimizer, alpha / beta / delta leaders            (Mirjalili et al., 2014)
  WOA  Whale Optimization Algorithm, encircling + bubble-net spiral (Mirjalili & Lewis, 2016)
  GA   Genetic Algorithm, tournament selection, uniform crossover,
       Gaussian mutation, elitism                                   (Holland, 1975)
  FA   Firefly Algorithm, attraction to every brighter firefly      (Yang, 2008)
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LOW, HIGH = -1.0, 1.0


def top_k_mask(vec, k):
    mask = np.zeros(len(vec), dtype=bool)
    mask[np.argsort(vec)[-k:]] = True
    return mask


class BudgetExhausted(Exception):
    pass


class Fitness:
    """mode="group3":    mean balanced accuracy, group-aware 3-fold CV.
    mode="loso_prob":  inner leave-one-scan-out; mean over held-out scans of the predicted probability of
                       the correct class. Mirrors the real test (a whole unseen scan) and is smooth."""

    def __init__(self, X, y, groups, k, seed=42, budget=None, mode="group3"):
        self.X, self.y, self.k, self.budget, self.mode = X, y, k, budget, mode
        if mode == "group3":
            cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
            self.folds = list(cv.split(X, y, groups))
        else:
            self.folds = [(np.where(groups != g)[0], np.where(groups == g)[0]) for g in np.unique(groups)]
        self.cache, self.evals = {}, 0

    def __call__(self, vec):
        mask = top_k_mask(vec, self.k)
        key = np.packbits(mask).tobytes()
        if key not in self.cache:
            if self.budget is not None and self.evals >= self.budget:
                raise BudgetExhausted
            self.evals += 1
            scores = []
            for tr, va in self.folds:
                clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=500))
                clf.fit(self.X[tr][:, mask], self.y[tr])
                if self.mode == "group3":
                    scores.append(balanced_accuracy_score(self.y[va], clf.predict(self.X[va][:, mask])))
                else:
                    p = clf.predict_proba(self.X[va][:, mask])[:, 1]
                    scores.append(float(np.mean(np.where(self.y[va] == 1, p, 1 - p))))
            self.cache[key] = float(np.mean(scores))
        return self.cache[key]


def run_selector(name, X, y, groups, k=20, pop_size=16, iters=None, budget=300, seed=42, mode="group3"):
    """Returns (mask, best CV fitness, best-so-far history per iteration, evaluations used).
    Every algorithm gets the same budget of unique fitness evaluations."""
    try:
        return _run(name, X, y, groups, k, pop_size, iters, budget, seed, mode)
    except BudgetExhausted as stop:
        return stop.args[0]


def _run(name, X, y, groups, k, pop_size, iters, budget, seed, mode):
    iters = iters or max(budget // pop_size - 1, 1)   # so decay schedules finish within the budget
    rng = np.random.default_rng(seed)
    f = Fitness(X, y, groups, k, seed, budget, mode)
    d = X.shape[1]
    pop = rng.uniform(LOW, HIGH, (pop_size, d))
    fit = np.array([f(p) for p in pop])  # initial population always fits in the budget
    best_i = int(fit.argmax())
    best, best_fit = pop[best_i].copy(), fit[best_i]
    history = [best_fit]

    def consider(i):
        nonlocal best, best_fit
        pop[i] = np.clip(pop[i], LOW, HIGH)
        try:
            fit[i] = f(pop[i])
        except BudgetExhausted:
            history.append(best_fit)
            raise BudgetExhausted((top_k_mask(best, k), best_fit, history, f.evals))
        if fit[i] > best_fit:
            best, best_fit = pop[i].copy(), fit[i]

    if name == "PSO":
        vel = rng.uniform(-0.2, 0.2, (pop_size, d))
        pbest, pbest_fit = pop.copy(), fit.copy()
        for t in range(iters):
            w = 0.9 - 0.5 * t / max(iters - 1, 1)
            for i in range(pop_size):
                r1, r2 = rng.random(d), rng.random(d)
                vel[i] = w * vel[i] + 1.5 * r1 * (pbest[i] - pop[i]) + 1.5 * r2 * (best - pop[i])
                vel[i] = np.clip(vel[i], -0.5, 0.5)
                pop[i] += vel[i]
                consider(i)
                if fit[i] > pbest_fit[i]:
                    pbest[i], pbest_fit[i] = pop[i].copy(), fit[i]
            history.append(best_fit)

    elif name == "GWO":
        for t in range(iters):
            a = 2 - 2 * t / iters
            leaders = pop[np.argsort(-fit)[:3]].copy()
            for i in range(pop_size):
                x = np.zeros(d)
                for L in leaders:
                    A = 2 * a * rng.random(d) - a
                    C = 2 * rng.random(d)
                    x += L - A * np.abs(C * L - pop[i])
                pop[i] = x / 3
                consider(i)
            history.append(best_fit)

    elif name == "WOA":
        for t in range(iters):
            a = 2 - 2 * t / iters
            for i in range(pop_size):
                A = 2 * a * rng.random() - a
                C = 2 * rng.random()
                if rng.random() < 0.5:
                    ref = best if abs(A) < 1 else pop[rng.integers(pop_size)]
                    pop[i] = ref - A * np.abs(C * ref - pop[i])
                else:
                    l = rng.uniform(-1, 1)
                    pop[i] = np.abs(best - pop[i]) * np.exp(l) * np.cos(2 * np.pi * l) + best
                consider(i)
            history.append(best_fit)

    elif name == "GA":
        for t in range(iters):
            def tournament():
                a, b = rng.integers(pop_size, size=2)
                return pop[a] if fit[a] >= fit[b] else pop[b]
            elite = pop[np.argsort(-fit)[:2]].copy()
            children = []
            while len(children) < pop_size - 2:
                p1, p2 = tournament(), tournament()
                cross = rng.random(d) < 0.5
                for c in (np.where(cross, p1, p2), np.where(cross, p2, p1)):
                    c = c + rng.normal(0, 0.3, d) * (rng.random(d) < 0.05)
                    children.append(c)
            pop = np.vstack([elite, children[:pop_size - 2]])
            for i in range(pop_size):
                consider(i)
            history.append(best_fit)

    elif name == "FA":
        beta0, gamma = 1.0, 1.0 / d
        for t in range(iters):
            alpha = 0.2 * (0.97 ** t)
            for i in range(pop_size):
                for j in range(pop_size):
                    if fit[j] > fit[i]:
                        r2 = np.sum((pop[i] - pop[j]) ** 2)
                        pop[i] += beta0 * np.exp(-gamma * r2) * (pop[j] - pop[i]) + alpha * (rng.random(d) - 0.5)
                        consider(i)
            history.append(best_fit)
    else:
        raise ValueError(name)

    return top_k_mask(best, k), best_fit, history, f.evals


ALGORITHMS = ["PSO", "GWO", "WOA", "GA", "FA"]
