"""Pendulum simulator, LQR expert, batched rollouts and the success test.

Everything here is deterministic: randomness enters only through the starting states that callers pass in.
Angles are measured from upright (theta = 0 is the inverted position), wrapped to [-pi, pi).
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_continuous_are

# Gymnasium Pendulum-v1 constants
G, M, L = 10.0, 1.0, 1.0
DT = 0.05            # s per step
U_MAX = 2.0          # torque limit
W_MAX = 8.0          # angular-velocity limit (rad/s)
T_STEPS = 200        # 10 s horizon
TAIL = 20            # success is judged on the last 20 sampled states (1 s)
TH_TOL, W_TOL = 0.1, 0.5

A_GAIN = 3 * G / (2 * L)     # 15: gravity term of theta_ddot
B_GAIN = 3.0 / (M * L ** 2)  # 3: torque term of theta_ddot


def wrap(th):
    """Map angles to [-pi, pi)."""
    return (th + np.pi) % (2 * np.pi) - np.pi


def step(x, u):
    """One Gymnasium Pendulum-v1 step (semi-implicit Euler) for a batch of states x[..., (theta, omega)]."""
    th, w = x[..., 0], x[..., 1]
    u = np.clip(u, -U_MAX, U_MAX)
    w_new = np.clip(w + (A_GAIN * np.sin(th) + B_GAIN * u) * DT, -W_MAX, W_MAX)
    th_new = wrap(th + w_new * DT)
    return np.stack([th_new, w_new], axis=-1)


def lqr_gain(q=(10.0, 1.0), r=1.0):
    """LQR on the continuous-time linearization about upright: theta_ddot = 15 theta + 3 u."""
    A = np.array([[0.0, 1.0], [A_GAIN, 0.0]])
    B = np.array([[0.0], [B_GAIN]])
    Q, R = np.diag(q), np.array([[r]])
    P = solve_continuous_are(A, B, Q, R)
    return (np.linalg.solve(R, B.T @ P)).ravel()  # u = -K x


K_LQR = lqr_gain()


def expert(x):
    """Clipped LQR torque for a batch of states (the 'answer key' the learners imitate)."""
    x = np.atleast_2d(x)
    return np.clip(-(np.column_stack([wrap(x[:, 0]), x[:, 1]]) @ K_LQR), -U_MAX, U_MAX)


def rollout(policy, x0, steps=T_STEPS):
    """Run many starts at once. policy maps an (n, 2) state batch to n torques (one call per time step).
    Returns states (n, steps + 1, 2) and the torques actually applied (n, steps), already clipped."""
    x = np.asarray(x0, dtype=float)
    xs, us = [x], []
    for _ in range(steps):
        u = np.clip(np.asarray(policy(x), dtype=float).reshape(-1), -U_MAX, U_MAX)
        us.append(u)
        x = step(x, u)
        xs.append(x)
    return np.stack(xs, axis=1), np.stack(us, axis=1)


def success(states):
    """True where |theta| < 0.1 rad and |omega| < 0.5 rad/s at every one of the last 20 sampled states."""
    tail = states[:, -TAIL:, :]
    return np.all((np.abs(tail[..., 0]) < TH_TOL) & (np.abs(tail[..., 1]) < W_TOL), axis=1)
