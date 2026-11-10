"""Error bars (course Sections 4B and 4C): standard errors and the percentile bootstrap over training seeds."""
from __future__ import annotations

import numpy as np


def mean_se(values):
    v = np.asarray(values, dtype=float)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))


def binomial_se(p, n):
    """Standard error of a success rate p measured on n independent starts: sqrt(p(1-p)/n) (<= 0.5/sqrt(n))."""
    return float(np.sqrt(p * (1 - p) / n))


def zero_success_upper95(n):
    """When 0 of n trials succeed, sqrt(p(1-p)/n) = 0 is useless as an error bar. The exact one-sided 95% upper
    bound solves (1 - p)^n = 0.05, i.e. p = 1 - 0.05^(1/n) ~ 3/n (the 'rule of three')."""
    return float(1 - 0.05 ** (1 / n))


def bootstrap_ci(values, rng, b=10_000, level=0.95):
    """Percentile bootstrap interval for the mean, resampling whole training runs (seeds) with replacement."""
    v = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(v), size=(b, len(v)))
    means = v[idx].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def paired_bootstrap_ci(a, b_vals, rng, b=10_000, level=0.95):
    """Interval for mean(a - b) when a and b come from the same training runs: resample runs, keep pairs."""
    d = np.asarray(a, dtype=float) - np.asarray(b_vals, dtype=float)
    return bootstrap_ci(d, rng, b, level)
