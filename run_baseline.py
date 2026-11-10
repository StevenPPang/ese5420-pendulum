"""Steps 2-4: fixed test sets, 10 training seeds x 40 expert trajectories, ridge tuned by grouped 5-fold CV,
then test torque MSE and closed-loop success with error bars.

Run:  python run_baseline.py
Writes results/results.json, results/per_seed.csv, results/cv_table.csv, results/run_log.txt and baseline.png.
The test sets are generated first from their own seed streams and are only *used* after every model choice
(the CV pick) has been made for that training seed.
"""
import csv
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn

from pendulum_il.baseline import GRID, as_policy, clipped_mse, cross_validate, make_model, pick
from pendulum_il.data import MASTER_SEED, expert_dataset, expert_starts, streams
from pendulum_il.sim import K_LQR, U_MAX, expert, rollout, success
from pendulum_il.stats import binomial_se, bootstrap_ci, mean_se, paired_bootstrap_ci, zero_success_upper95

HERE = Path(__file__).parent
OUT = HERE / "results"
N_TRAIN_TRAJ = 40
N_SEEDS = 10
N_TEST_TRAJ = 100
N_TEST_STARTS = 500
MODELS = ("ridge_degree1", "ridge_cv")


def main():
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    log = []
    say = lambda *a: (print(*a), log.append(" ".join(str(x) for x in a)))  # noqa: E731
    s = streams()

    # --- fixed test sets (separate streams; built once, used once per model at the very end)
    x0_test_traj, d1, a1 = expert_starts(np.random.default_rng(s["test_traj"]), N_TEST_TRAJ)
    X_test, y_test, _ = expert_dataset(x0_test_traj)
    x0_test_cl, d2, a2 = expert_starts(np.random.default_rng(s["test_starts"]), N_TEST_STARTS)
    exp_states, _ = rollout(expert, x0_test_cl)
    say(f"test MSE set: {N_TEST_TRAJ} expert trajectories -> {len(y_test)} pairs; "
        f"closed-loop set: {N_TEST_STARTS} starts; expert success on them = {success(exp_states).mean():.3f}")

    train_children = s["train"].spawn(N_SEEDS)
    per_seed, cv_rows_all = [], []
    draws_total = acc_total = 0
    for i, child in enumerate(train_children):
        rng = np.random.default_rng(child)
        x0, draws, acc = expert_starts(rng, N_TRAIN_TRAJ)
        draws_total += draws
        acc_total += acc
        X, y, groups = expert_dataset(x0)
        rows = cross_validate(X, y, groups, rng)
        for r in rows:
            cv_rows_all.append({"seed": i, **{k: v for k, v in r.items() if k != "fold_mse"}})
        chosen = {"ridge_degree1": pick(rows, degree=1), "ridge_cv": pick(rows)}
        rec = {"seed": i, "n_train_pairs": len(y), "frac_train_clipped": float(np.mean(np.abs(y) >= U_MAX - 1e-9))}
        # Diagnostic only (not a baseline, not tuned): degree-1 ridge fit on the unclipped pairs alone.
        un = np.abs(y) < U_MAX - 1e-9
        diag = make_model(1, 0.01).fit(X[un], y[un])
        coef = diag[-1].coef_ / diag[1].scale_      # undo the scaler -> torque per rad, per rad/s
        rec["diag_unclipped_coef"] = coef.tolist()
        rec["diag_unclipped_success"] = float(success(rollout(as_policy(diag), x0_test_cl)[0]).mean())
        for name, row in chosen.items():
            model = make_model(row["degree"], row["alpha"]).fit(X, y)          # refit on all 40 trajectories
            mse = clipped_mse(model, X_test, y_test)                          # test set touched here only
            states, _ = rollout(as_policy(model), x0_test_cl)                 # one batched predict per step
            p = float(success(states).mean())
            e = 1e-4  # local gain of the learned controller at upright (finite difference; diagnostic only)
            gain_th = float((model.predict([[e, 0.0]])[0] - model.predict([[-e, 0.0]])[0]) / (2 * e))
            gain_w = float((model.predict([[0.0, e]])[0] - model.predict([[0.0, -e]])[0]) / (2 * e))
            rec.update({f"{name}_degree": row["degree"], f"{name}_alpha": row["alpha"],
                        f"{name}_cv_mse": row["cv_mse"], f"{name}_cv_se": row["cv_se"],
                        f"{name}_test_mse": mse, f"{name}_success": p,
                        f"{name}_success_binom_se": binomial_se(p, N_TEST_STARTS),
                        f"{name}_gain_theta": gain_th, f"{name}_gain_omega": gain_w})
        per_seed.append(rec)
        say(f"seed {i}: CV picked degree {rec['ridge_cv_degree']}, alpha {rec['ridge_cv_alpha']:g} | "
            f"deg1: MSE {rec['ridge_degree1_test_mse']:.4f}, success {rec['ridge_degree1_success']:.3f} | "
            f"CV: MSE {rec['ridge_cv_test_mse']:.4f}, success {rec['ridge_cv_success']:.3f}")

    boot = np.random.default_rng(np.random.SeedSequence(MASTER_SEED).spawn(5)[4])
    summary = {}
    for name in MODELS:
        mse = [r[f"{name}_test_mse"] for r in per_seed]
        suc = [r[f"{name}_success"] for r in per_seed]
        m_mse, se_mse = mean_se(mse)
        m_suc, se_suc = mean_se(suc)
        summary[name] = {
            "test_mse_mean": m_mse, "test_mse_se_over_seeds": se_mse, "test_mse_boot95": bootstrap_ci(mse, boot),
            "success_mean": m_suc, "success_se_over_seeds": se_suc, "success_boot95": bootstrap_ci(suc, boot),
            "success_binom_se_single_run": binomial_se(m_suc, N_TEST_STARTS),
            "success_upper95_if_zero_pooled": (zero_success_upper95(N_TEST_STARTS * N_SEEDS)
                                               if max(suc) == 0 else None),
            "gain_theta_mean": mean_se([r[f"{name}_gain_theta"] for r in per_seed])[0],
            "gain_omega_mean": mean_se([r[f"{name}_gain_omega"] for r in per_seed])[0],
            "degrees_chosen": [r[f"{name}_degree"] for r in per_seed],
            "alphas_chosen": [r[f"{name}_alpha"] for r in per_seed],
        }
    summary["diagnostics"] = {
        "frac_train_clipped_mean": mean_se([r["frac_train_clipped"] for r in per_seed])[0],
        "frac_train_clipped_range": [min(r["frac_train_clipped"] for r in per_seed),
                                     max(r["frac_train_clipped"] for r in per_seed)],
        "unclipped_refit_coef_mean": np.mean([r["diag_unclipped_coef"] for r in per_seed], axis=0).tolist(),
        "unclipped_refit_max_rel_err_vs_K": float(max(np.max(np.abs((np.array(r["diag_unclipped_coef"]) + K_LQR) / K_LQR))
                                                    for r in per_seed)),
        "unclipped_refit_success_per_seed": [r["diag_unclipped_success"] for r in per_seed],
    }
    summary["difference_cv_minus_degree1"] = {
        "test_mse_paired_boot95": paired_bootstrap_ci([r["ridge_cv_test_mse"] for r in per_seed],
                                                      [r["ridge_degree1_test_mse"] for r in per_seed], boot),
        "success_paired_boot95": paired_bootstrap_ci([r["ridge_cv_success"] for r in per_seed],
                                                     [r["ridge_degree1_success"] for r in per_seed], boot),
    }
    results = {
        "settings": {"master_seed": MASTER_SEED, "n_train_traj": N_TRAIN_TRAJ, "n_seeds": N_SEEDS,
                     "n_test_traj": N_TEST_TRAJ, "n_test_starts": N_TEST_STARTS, "grid": GRID,
                     "versions": {"numpy": np.__version__, "sklearn": sklearn.__version__}},
        "expert_acceptance": {"train_draws": draws_total, "train_accepted": acc_total,
                              "test_traj_draws": d1, "test_traj_accepted": a1,
                              "test_starts_draws": d2, "test_starts_accepted": a2},
        "summary": summary, "per_seed": per_seed, "runtime_s": round(time.time() - t0, 1),
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    with open(OUT / "per_seed.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_seed[0]))
        w.writeheader()
        w.writerows(per_seed)
    with open(OUT / "cv_table.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cv_rows_all[0]))
        w.writeheader()
        w.writerows(cv_rows_all)
    for name in MODELS:
        v = summary[name]
        say(f"{name}: test MSE {v['test_mse_mean']:.4f} ± {v['test_mse_se_over_seeds']:.4f} (SE over seeds), "
            f"95% boot {v['test_mse_boot95'][0]:.4f}-{v['test_mse_boot95'][1]:.4f} | success "
            f"{v['success_mean']:.3f} ± {v['success_se_over_seeds']:.3f} (seeds), binomial SE of one run "
            f"{v['success_binom_se_single_run']:.3f}, 95% boot {v['success_boot95'][0]:.3f}-{v['success_boot95'][1]:.3f}"
            + (f", one-sided 95% upper bound (0 of {N_TEST_STARTS * N_SEEDS} rollouts) "
               f"{v['success_upper95_if_zero_pooled']:.4f}" if v["success_upper95_if_zero_pooled"] else "")
            + f" | gain at upright d(u)/d(theta) {v['gain_theta_mean']:.2f}, d(u)/d(omega) {v['gain_omega_mean']:.2f}")
    say(f"expert gain at upright: d(u)/d(theta) {-K_LQR[0]:.2f}, d(u)/d(omega) {-K_LQR[1]:.2f}; "
        f"holding upright needs d(u)/d(theta) < -5 (3*K_theta > 15)")
    d = summary["diagnostics"]
    say(f"diagnostics: clipped training pairs {100 * d['frac_train_clipped_mean']:.1f}% "
        f"(range {100 * d['frac_train_clipped_range'][0]:.1f}-{100 * d['frac_train_clipped_range'][1]:.1f}%); "
        f"unclipped-only degree-1 refit coef {np.round(d['unclipped_refit_coef_mean'], 3).tolist()} "
        f"(max rel. error vs -K {100 * d['unclipped_refit_max_rel_err_vs_K']:.3f}%), success per seed "
        f"{d['unclipped_refit_success_per_seed']}")
    say(f"runtime {results['runtime_s']} s")
    (OUT / "run_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    figure(per_seed, summary, cv_rows_all)


def figure(per_seed, summary, cv_rows_all):
    """Left: the project's key figure (test MSE vs closed-loop success), baseline only for now.
    Right: the 5-fold CV curve that chose the setting (mean over the 10 seeds, error bar = SE over seeds)."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.4))
    style = {"ridge_degree1": ("C0", "o", "Ridge, degree 1"), "ridge_cv": ("C3", "s", "Ridge, CV-chosen degree")}
    for name, (c, mk, label) in style.items():
        xs = [r[f"{name}_test_mse"] for r in per_seed]
        ys = [r[f"{name}_success"] for r in per_seed]
        ax.scatter(xs, ys, color=c, marker=mk, alpha=0.3, s=18)
        v = summary[name]
        ax.errorbar(v["test_mse_mean"], v["success_mean"], xerr=v["test_mse_se_over_seeds"],
                    yerr=v["success_se_over_seeds"], color=c, marker=mk, ms=8, capsize=4,
                    label=f"{label} (mean ± SE, {len(per_seed)} seeds)")
    ax.axhline(1.0, color="0.5", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.08)
    ax.text(ax.get_xlim()[0], 1.01, " expert: MSE 0, success 1.00", va="bottom", fontsize=8, color="0.4")
    ax.set_xlabel("test torque MSE, 100 held-out expert trajectories [(N·m)²]")
    ax.set_ylabel("closed-loop success rate (500 fixed starts)")
    ax.set_title("A. Imitation error vs closed-loop success (40 traj.)")
    ax.legend(fontsize=8, loc="center left")
    ax.grid(alpha=0.3, which="both")

    alphas = sorted({r["alpha"] for r in cv_rows_all})
    degrees = sorted({r["degree"] for r in cv_rows_all})
    for j, a in enumerate(alphas):
        means, ses = [], []
        for d in degrees:
            vals = [r["cv_mse"] for r in cv_rows_all if r["alpha"] == a and r["degree"] == d]
            means.append(np.mean(vals))
            ses.append(np.std(vals, ddof=1) / np.sqrt(len(vals)))
        bx.errorbar(np.array(degrees) + (j - 0.5) * 0.08, means, yerr=ses, marker="o", capsize=3,
                    label=f"penalty α = {a:g}")
    bx.set_yscale("log")
    bx.set_xticks(degrees)
    bx.set_xlabel("polynomial degree")
    bx.set_ylabel("5-fold CV MSE (folds grouped by trajectory)")
    bx.set_title("B. Cross-validation curve (mean ± SE over 10 seeds)")
    bx.legend(fontsize=8)
    bx.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(HERE / "baseline.png", dpi=150)


if __name__ == "__main__":
    main()
