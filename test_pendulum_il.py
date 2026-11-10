"""Correctness tests for the simulator, the expert, the folds and the error bars.  Run:  python -m pytest -q"""
import numpy as np
import pytest

from pendulum_il.baseline import cross_validate, trajectory_folds
from pendulum_il.data import expert_dataset, expert_starts, streams
from pendulum_il.sim import K_LQR, U_MAX, W_MAX, expert, rollout, step, success, wrap
from pendulum_il.stats import binomial_se, bootstrap_ci, zero_success_upper95


def test_wrap_range():
    th = np.linspace(-20, 20, 10001)
    w = wrap(th)
    assert np.all(w >= -np.pi) and np.all(w < np.pi)
    assert np.allclose(np.sin(w), np.sin(th)) and np.allclose(np.cos(w), np.cos(th))


def test_upright_is_equilibrium_and_unstable():
    x = np.array([[0.0, 0.0]])
    assert np.allclose(step(x, np.zeros(1)), 0.0)
    x = np.array([[1e-3, 0.0]])
    for _ in range(40):
        x = step(x, np.zeros(1))
    assert abs(x[0, 0]) > 1e-2  # falls away without control


def test_limits():
    x = step(np.array([[0.5, 7.9]]), np.array([100.0]))
    assert abs(x[0, 1]) <= W_MAX
    assert np.all(np.abs(expert(np.random.default_rng(0).uniform(-4, 4, (1000, 2)))) <= U_MAX)


def test_lqr_gain_solves_riccati_and_stabilizes_linear_model():
    A = np.array([[0.0, 1.0], [15.0, 0.0]])
    B = np.array([[0.0], [3.0]])
    eig = np.linalg.eigvals(A - B @ K_LQR[None, :])
    assert np.all(eig.real < 0)
    assert K_LQR[0] * 3 > 15  # angle gain strong enough to beat gravity


def test_expert_succeeds_from_near_upright():
    # Small box only: with |u| <= 2 the torque term (3u <= 6) beats gravity (15 sin theta) only for
    # |theta| < ~0.41 rad, and less when the pendulum is already falling outward. A start such as
    # (-0.26 rad, -0.84 rad/s) is genuinely not catchable, so a wider box would test the wrong thing.
    x0 = np.random.default_rng(1).uniform([-0.1, -0.3], [0.1, 0.3], (200, 2))
    states, _ = rollout(expert, x0)
    assert success(states).all()


def test_rollout_applies_clipped_torque():
    _, u = rollout(lambda x: np.full(len(x), 50.0), np.zeros((3, 2)), steps=5)
    assert np.all(u == U_MAX)


def test_streams_are_independent_and_reproducible():
    a = np.random.default_rng(streams()["train"]).random(5)
    b = np.random.default_rng(streams()["train"]).random(5)
    c = np.random.default_rng(streams()["test_starts"]).random(5)
    assert np.array_equal(a, b) and not np.allclose(a, c)


def test_dataset_shapes_and_groups():
    x0, draws, acc = expert_starts(np.random.default_rng(3), 7, batch=50)
    assert len(x0) == 7 and acc <= draws
    X, y, g = expert_dataset(x0)
    assert X.shape == (1400, 2) and y.shape == (1400,) and np.array_equal(np.unique(g), np.arange(7))


def test_folds_keep_whole_trajectories_together():
    groups = np.repeat(np.arange(40), 200)
    fold = trajectory_folds(groups, np.random.default_rng(0))
    for t in range(40):
        assert len(np.unique(fold[groups == t])) == 1
    sizes = np.bincount(fold) // 200
    assert sizes.max() - sizes.min() <= 1 and sizes.sum() == 40


def test_cv_recovers_exact_linear_target():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1000, 2))
    y = 0.3 * X[:, 0] - 0.2 * X[:, 1]
    rows = cross_validate(X, y, np.repeat(np.arange(10), 100), rng, grid=[{"degree": 1, "alpha": 1e-6}])
    assert rows[0]["cv_mse"] < 1e-8


def test_error_bars():
    assert binomial_se(0.5, 500) == pytest.approx(0.02236, abs=1e-5)  # the feedback's "about 0.022"
    assert zero_success_upper95(5000) == pytest.approx(3 / 5000, rel=0.01)
    lo, hi = bootstrap_ci(np.arange(10.0), np.random.default_rng(0), b=4000)
    assert lo < 4.5 < hi
