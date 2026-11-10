"""Ridge baseline with polynomial features, tuned by 5-fold cross-validation grouped by trajectory.

Every step that learns from data (polynomial expansion has no parameters, but the scaler does, and so does ridge)
sits inside a scikit-learn Pipeline that is refit on the training folds only, so nothing from a validation fold
leaks into the fit. Predictions are clipped to the torque limit before any error is computed, exactly as the
controller would clip them.
"""
from __future__ import annotations

from itertools import product

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from .sim import U_MAX

# Eight candidate settings, fixed before any result was seen (the proposal's "eight settings per method").
DEGREES = (1, 2, 3, 5)
ALPHAS = (1e-2, 1e1)
GRID = [{"degree": d, "alpha": a} for d, a in product(DEGREES, ALPHAS)]
N_FOLDS = 5


def make_model(degree, alpha):
    return make_pipeline(PolynomialFeatures(degree, include_bias=False), StandardScaler(), Ridge(alpha=alpha))


def clipped_mse(model, X, y):
    return float(np.mean((np.clip(model.predict(X), -U_MAX, U_MAX) - y) ** 2))


def trajectory_folds(groups, rng, k=N_FOLDS):
    """Assign whole trajectories to k folds at random (sizes differ by at most one trajectory)."""
    ids = np.unique(groups)
    perm = rng.permutation(ids)
    fold_of = {traj: i % k for i, traj in enumerate(perm)}
    return np.array([fold_of[g] for g in groups])


def cross_validate(X, y, groups, rng, grid=GRID, k=N_FOLDS):
    """Return one row per setting: mean CV MSE, its standard error over the k folds, and the per-fold values."""
    fold = trajectory_folds(groups, rng, k)
    rows = []
    for setting in grid:
        errs = []
        for f in range(k):
            tr, va = fold != f, fold == f
            m = make_model(**setting).fit(X[tr], y[tr])
            errs.append(clipped_mse(m, X[va], y[va]))
        errs = np.array(errs)
        rows.append({**setting, "cv_mse": float(errs.mean()), "cv_se": float(errs.std(ddof=1) / np.sqrt(k)),
                     "fold_mse": errs.tolist()})
    return rows


def pick(rows, degree=None):
    """Setting with the lowest mean CV MSE (optionally restricted to one degree)."""
    cand = [r for r in rows if degree is None or r["degree"] == degree]
    return min(cand, key=lambda r: r["cv_mse"])


def as_policy(model):
    return lambda x: np.clip(model.predict(np.asarray(x)), -U_MAX, U_MAX)
