# Low imitation error, unstable controller? A pendulum imitation-learning study

ESE 5420 final project (Topic 14), baseline checkpoint.

Question: when models imitate an LQR pendulum expert, does lower test torque MSE go with higher closed-loop success?

Baseline result (ridge on polynomial features, 10 training seeds, 40 trajectories each): the model chosen by
cross-validation has test torque MSE 0.0465 ± 0.0016, the degree-1 model 0.1408 ± 0.0008, and neither balances the
pendulum from any of the 500 test starts. `baseline.txt` has the full write-up and `baseline.png` the figure.

## Files
| File | What it is |
|---|---|
| `pendulum_il/sim.py` | Pendulum-v1 dynamics, LQR gain, clipped expert, batched rollout, success test |
| `pendulum_il/data.py` | One master seed split into independent streams; expert-stabilized starts (rejection sampling); datasets |
| `pendulum_il/baseline.py` | Ridge + polynomial pipeline, 8-setting grid, 5-fold CV grouped by trajectory |
| `pendulum_il/stats.py` | Standard error, binomial SE, bootstrap and paired bootstrap, zero-success upper bound |
| `check_gym.py` | Feedback step 1: one-step agreement with Gymnasium, outcome agreement, expert coverage |
| `run_baseline.py` | Feedback steps 2-3: test sets, 10 seeds x 40 trajectories, CV, test MSE, closed loop, figure |
| `test_pendulum_il.py` | 11 correctness tests |
| `compare_results.py` | Compares two `results.json` files with a relative tolerance |
| `results/` | The committed run: `check_gym.json`, `results.json`, `per_seed.csv`, `cv_table.csv`, `run_log.txt` |
| `baseline.txt`, `baseline.png` | The checkpoint write-up and its figure (A: MSE vs success; B: CV curve) |

## Run it
Tested with Python 3.14; package versions are pinned in `requirements.txt`.
```
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt     # Windows: .venv\Scripts\python
cp -r results results_committed                          # keep the committed numbers for the comparison
.venv/bin/python -m pytest -q                            # 11 passed
.venv/bin/python check_gym.py                            # writes results/check_gym.json
.venv/bin/python run_baseline.py                         # writes results/ and baseline.png, a few seconds
.venv/bin/python compare_results.py results_committed/results.json results/results.json
```
Every file in `results/` and `baseline.png` comes from one run of `check_gym.py` followed by `run_baseline.py` with
this code. The master seed is 20261109. With the same seed and package versions the last command prints
`0 field(s) differ` (raw floats may differ in the last digits; the comparison uses a 1e-9 relative tolerance).
