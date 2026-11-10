"""Seeds, starting states and datasets.

One master seed is split with numpy's SeedSequence into independent streams, so the training data, the test
trajectories and the closed-loop test starts never share random numbers:
  stream 0: Gymnasium check and the expert-coverage estimate
  stream 1: the 100 test trajectories (torque MSE)
  stream 2: the 500 closed-loop test starts
  stream 3: training data, split again into one child per training seed
"""
from __future__ import annotations

import numpy as np

from .sim import T_STEPS, W_MAX, expert, rollout, success

MASTER_SEED = 20261109


def streams(master=MASTER_SEED):
    check, test_traj, test_starts, train = np.random.SeedSequence(master).spawn(4)
    return {"check": check, "test_traj": test_traj, "test_starts": test_starts, "train": train}


def uniform_starts(rng, n):
    return np.column_stack([rng.uniform(-np.pi, np.pi, n), rng.uniform(-W_MAX, W_MAX, n)])


def expert_starts(rng, n, batch=4000):
    """Rejection sampling: draw uniform starts, keep those the expert stabilizes, until n are kept.
    Returns the kept starts and the number of draws it took (so the acceptance rate can be reported)."""
    kept, draws = [], 0
    while sum(len(k) for k in kept) < n:
        x0 = uniform_starts(rng, batch)
        draws += batch
        states, _ = rollout(expert, x0)
        kept.append(x0[success(states)])
    accepted = int(sum(len(k) for k in kept))
    kept = np.concatenate(kept)
    return kept[:n], draws, accepted


def expert_dataset(x0):
    """Expert trajectories from the given starts: inputs X (n*200, 2) = (theta, omega) at t = 0..199,
    targets y = expert torque at those states, groups = trajectory index (for grouped folds)."""
    states, torques = rollout(expert, x0, T_STEPS)
    n = len(x0)
    X = states[:, :-1, :].reshape(-1, 2)
    y = torques.reshape(-1)
    groups = np.repeat(np.arange(n), T_STEPS)
    return X, y, groups
