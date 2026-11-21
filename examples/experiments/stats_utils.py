"""Statistics without scipy or sklearn (both are unreliable on some hosts): the tests a paper reports.

Kruskal-Wallis (ties corrected) with eta^2, chi-square tail, exact McNemar, bootstrap intervals, a linear
discriminant classifier with a permutation test, and the group-aware stratified fold assignment used everywhere.
"""

from __future__ import annotations

import math

import numpy as np


def chi2_sf(x: float, df: int) -> float:
    """P(chi-square with df degrees of freedom > x): the regularised upper incomplete gamma function Q(df/2, x/2)."""
    a, x = df / 2.0, x / 2.0
    if x <= 0:
        return 1.0
    if x < a + 1:
        term = total = 1.0 / a
        for n in range(1, 500):
            term *= x / (a + n)
            total += term
            if abs(term) < abs(total) * 1e-14:
                break
        return max(0.0, 1.0 - total * math.exp(-x + a * math.log(x) - math.lgamma(a)))
    b, c, d = x + 1 - a, 1e300, 1.0 / (x + 1 - a)
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        h *= d * c
        if abs(d * c - 1) < 1e-14:
            break
    return min(1.0, h * math.exp(-x + a * math.log(x) - math.lgamma(a)))


def kruskal(values: np.ndarray, y: np.ndarray) -> dict:
    """Kruskal-Wallis H test across the classes and eta^2, the share of rank variance the class explains."""
    groups = [values[y == c] for c in np.unique(y)]
    groups = [g[~np.isnan(g)] for g in groups]
    allv = np.concatenate(groups)
    n, k = len(allv), len(groups)
    order = np.argsort(allv, kind="mergesort")
    ranks, sorted_v, i = np.empty(n), allv[order], 0
    while i < n:
        j = i
        while j + 1 < n and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    h, start = 0.0, 0
    for g in groups:
        r = ranks[start:start + len(g)]
        h += r.sum() ** 2 / len(g)
        start += len(g)
    h = 12.0 / (n * (n + 1)) * h - 3 * (n + 1)
    _, counts = np.unique(allv, return_counts=True)
    tie = 1 - (counts ** 3 - counts).sum() / (n ** 3 - n)
    h = h / tie if tie > 0 else 0.0
    return {"H": round(float(h), 3), "p": chi2_sf(h, k - 1), "eta_squared": round(float(max(0.0, (h - k + 1) / (n - k))), 4)}


def mcnemar(a_correct: np.ndarray, b_correct: np.ndarray) -> dict:
    """Exact McNemar test of two paired sets of correct/incorrect outcomes."""
    only_a, only_b = int((a_correct & ~b_correct).sum()), int((~a_correct & b_correct).sum())
    n = only_a + only_b
    if n == 0:
        return {"only_first_correct": 0, "only_second_correct": 0, "p": 1.0}
    k = min(only_a, only_b)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"only_first_correct": only_a, "only_second_correct": only_b, "p": round(p, 5)}


def boot_ci(values: np.ndarray, rng: np.random.Generator, resamples: int = 1000) -> tuple[float, float]:
    """95% bootstrap interval of the mean of per-image values."""
    idx = rng.integers(0, len(values), (resamples, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def group_stratified_folds(y: np.ndarray, groups: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Fold index of every sample: images of one group (a capture session) stay together, classes are spread evenly.

    A local search: start from a random assignment of groups to folds, then move or swap groups while that evens out the
    per-fold class counts; a class missing from a fold is penalised heavily.  Every class must appear in every fold, or
    the function raises (the caller then decides whether to drop the grouping).
    """
    classes = np.unique(y)
    rng = np.random.default_rng(seed)
    ug = list(np.unique(groups))
    gcount = np.array([[(y[groups == g] == c).sum() for c in classes] for g in ug], float)  # groups x classes
    target = np.array([(y == c).sum() / k for c in classes])

    def cost(assign: np.ndarray) -> float:
        counts = np.stack([gcount[assign == f].sum(axis=0) for f in range(k)])
        return float((((counts - target) ** 2) / np.maximum(target, 1)).sum() + 100.0 * (counts == 0).sum())

    best_assign, best_cost = None, float("inf")
    for restart in range(30):
        assign = rng.integers(0, k, len(ug))
        current = cost(assign)
        improved = True
        while improved:
            improved = False
            for i in rng.permutation(len(ug)):
                for f in range(k):  # move group i to fold f
                    if f != assign[i]:
                        trial = assign.copy()
                        trial[i] = f
                        c = cost(trial)
                        if c < current - 1e-9:
                            assign, current, improved = trial, c, True
            for _ in range(len(ug) * 3):  # swap two groups
                i, j = rng.integers(0, len(ug), 2)
                if assign[i] != assign[j]:
                    trial = assign.copy()
                    trial[i], trial[j] = assign[j], assign[i]
                    c = cost(trial)
                    if c < current - 1e-9:
                        assign, current, improved = trial, c, True
        if current < best_cost:
            best_assign, best_cost = assign.copy(), current
    index = {g: i for i, g in enumerate(ug)}
    fold_of = np.array([best_assign[index[g]] for g in groups])
    if any(not (y[fold_of == f] == c).any() for f in range(k) for c in classes):
        raise ValueError("no group-aware fold assignment keeps every class in every fold")
    return fold_of


def lda_cv_accuracy(X: np.ndarray, y: np.ndarray, fold_of: np.ndarray) -> float:
    """Cross-validated accuracy of a linear discriminant (class means, pooled covariance, class priors)."""
    classes = np.unique(y)
    correct = 0
    for f in np.unique(fold_of):
        test, train = np.where(fold_of == f)[0], np.where(fold_of != f)[0]
        mu, sd = X[train].mean(0), X[train].std(0) + 1e-9
        Xt, Xs = (X[train] - mu) / sd, (X[test] - mu) / sd
        means = np.stack([Xt[y[train] == c].mean(0) if (y[train] == c).any() else np.zeros(X.shape[1]) for c in classes])
        resid = Xt - means[np.searchsorted(classes, y[train])]
        inv = np.linalg.inv(resid.T @ resid / max(1, len(Xt) - len(classes)) + 1e-2 * np.eye(X.shape[1]))
        prior = np.log(np.array([max((y[train] == c).mean(), 1e-6) for c in classes]))
        score = Xs @ inv @ means.T - 0.5 * np.einsum("kd,de,ke->k", means, inv, means) + prior
        correct += int((classes[score.argmax(1)] == y[test]).sum())
    return correct / len(y)


def classifier_report(X: np.ndarray, y: np.ndarray, fold_of: np.ndarray, permutations: int, seed: int) -> dict:
    """Cross-validated linear-discriminant accuracy against the majority baseline and a label-permutation null."""
    rng = np.random.default_rng(seed)
    acc = lda_cv_accuracy(X, y, fold_of)
    null = np.array([lda_cv_accuracy(X, rng.permutation(y), fold_of) for _ in range(permutations)])
    return {"cv_accuracy": round(acc, 4), "majority_baseline": round(float(np.bincount(y).max() / len(y)), 4),
            "permutation_null_mean": round(float(null.mean()), 4),
            "permutation_p": round(float((1 + (null >= acc).sum()) / (1 + permutations)), 4), "permutations": permutations}
