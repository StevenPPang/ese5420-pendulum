"""Step 1: check our simulator against Gymnasium's Pendulum-v1, and measure how many uniform starts the expert
stabilizes.  Run:  python check_gym.py   (writes results/check_gym.json)"""
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from pendulum_il.data import expert_starts, streams, uniform_starts
from pendulum_il.sim import K_LQR, T_STEPS, U_MAX, expert, rollout, step, success, wrap
from pendulum_il.stats import binomial_se

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)


def compare_with_gym(rng, n_runs=6, n_agree=200):
    """(a) One-step check: from Gymnasium's own state, does our step() give Gymnasium's next state?
    (b) Outcome check: from the same starts, do the two simulators agree on whether the expert succeeds?
    Long multi-step comparisons are not meaningful when the pendulum swings chaotically: float rounding
    (1e-16) grows until the two copies separate, even though every individual step agrees."""
    env = gym.make("Pendulum-v1", g=10.0)
    worst, runs = 0.0, []
    for i in range(n_runs):
        env.reset(seed=int(rng.integers(1 << 31)))
        env.unwrapped.state = uniform_starts(rng, 1)[0].copy()
        use_expert = i % 2 == 0
        run_max = 0.0
        for _ in range(T_STEPS):
            before = np.array(env.unwrapped.state, dtype=float)
            u = float(expert(before)[0]) if use_expert else float(rng.uniform(-U_MAX, U_MAX))
            u = float(np.float32(u))  # Gymnasium stores the action as float32: give both the same value
            ours = step(before, u)
            env.step(np.array([u], dtype=np.float32))
            th_g, w_g = env.unwrapped.state
            run_max = max(run_max, float(abs(wrap(th_g - ours[0]))), float(abs(w_g - ours[1])))
        worst = max(worst, run_max)
        runs.append({"torques": "expert" if use_expert else "random", "one_step_max_abs_diff": run_max})
    # (b) outcome agreement over many starts
    x0 = uniform_starts(rng, n_agree)
    ours_ok = success(rollout(expert, x0)[0])
    gym_ok, max_div = [], []
    for x in x0:
        env.reset(seed=0)
        env.unwrapped.state = x.copy()
        traj = [x.copy()]
        for _ in range(T_STEPS):
            u = float(np.float32(float(expert(np.array(env.unwrapped.state))[0])))
            env.step(np.array([u], dtype=np.float32))
            traj.append(np.array(env.unwrapped.state, dtype=float))
        traj = np.array(traj)
        traj[:, 0] = wrap(traj[:, 0])
        gym_ok.append(bool(success(traj[None])[0]))
        ours_traj = rollout(expert, x[None])[0][0]
        max_div.append(float(np.max(np.abs(np.column_stack([wrap(traj[:, 0] - ours_traj[:, 0]),
                                                            traj[:, 1] - ours_traj[:, 1]])))))
    env.close()
    agree = float(np.mean(ours_ok == np.array(gym_ok)))
    max_div = np.array(max_div)
    return worst, runs, {"starts": n_agree, "agreement": agree, "ours_success": float(ours_ok.mean()),
                         "gym_success": float(np.mean(gym_ok)),
                         "whole_run_max_state_diff_median": float(np.median(max_div)),
                         "whole_run_max_state_diff_max": float(max_div.max()),
                         "whole_runs_diverging_above_0.1": int(np.sum(max_div > 0.1))}


def main():
    s = streams()
    rng = np.random.default_rng(s["check"])
    worst, runs, agree = compare_with_gym(rng)
    _, draws, accepted = expert_starts(rng, 4000, batch=4000)
    frac = accepted / draws
    res = {
        "lqr_gain_K": K_LQR.tolist(),
        "gym_one_step_max_abs_diff": worst,
        "gym_runs": runs,
        "gym_outcome_agreement": agree,
        "coverage_draws": draws,
        "coverage_accepted": accepted,
        "expert_success_fraction_uniform_starts": frac,
        "expert_success_fraction_se": binomial_se(frac, draws),
    }
    (OUT / "check_gym.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
