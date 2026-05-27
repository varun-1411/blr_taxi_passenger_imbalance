# Airport Taxi Queue Optimizer

Transient queueing control framework for optimizing taxi-passenger imbalance at Kempegowda International Airport (Terminal 1).

## Model

Two-zone system (reserve + pickup) modeled as a 2D continuous-time Markov chain (CTMC):
- **State:** `(s, n)` where `s` = taxis in reserve, `n` = signed pickup occupancy (+taxi/-passenger)
- **Controls:** taxi addition rate `μ⁺` and removal rate `μ⁻` per interval
- **Method:** Piecewise-constant approximation (PCA) solved via uniformization with autodiff (PyTorch)

## Quick Start

```bash
# Install
pip install torch numpy pandas matplotlib

# Run diagnostic
python experiments/debug_trace.py --n_intervals 20 --do_nothing

# Quick test (20 intervals)
python experiments/quick_test.py --n_intervals 20 --commit 5 --max_iter 100

# Full pipeline
bash experiments/run_pipeline.sh test    # 20 intervals, quick
bash experiments/run_pipeline.sh full    # 288 intervals, publication
```

## Repository Structure

```
├── config.py                    # QueueConfig (state space, costs, delays)
├── data.py                      # Load blr_demand_profile.csv (T1/T2)
├── optimizer_utils.py           # Shared functions (single source of truth)
│
├── model/                       # Core CTMC model
│   ├── generator.py             # Q matrix construction
│   ├── simulation.py            # Uniformization, RK4, expm solvers
│   ├── metrics.py               # Objective computation, run_simulation
│   └── steady_state.py          # Steady-state solver
│
├── optimizers/                  # Standalone optimizers
│   ├── adam_optimizer.py         # Full-day Adam (transient)
│   ├── brent_optimizer.py       # Steady-state Brent
│   ├── aimd_optimizer.py
│   ├── random_search.py
│   └── bayesian_optimization.py
│
├── experiments/                 # Experiment scripts
│   ├── find_initial_state.py    # π₀ calibration (cyclic + carryover)
│   ├── run_optimizers.py        # Full-day vs Greedy vs MPC comparison
│   ├── quick_test.py            # Verbose test + publication plots
│   ├── run_model_analysis.py    # Numerical methods + SS vs transient
│   ├── sensitivity_analysis.py  # 6 sensitivity analyses
│   ├── debug_trace.py           # Per-interval diagnostic trace
│   ├── verify_utils.py          # Verify optimizer_utils matches old code
│   └── run_pipeline.sh          # Run all experiments in order
│
├── Datasets/
│   └── blr_demand_profile.csv   # BLR airport demand data
└── results/                     # Output (gitignored)
```

## Experiment Pipeline

### Step 1: Initial State Calibration
```bash
python experiments/find_initial_state.py --n_refine 5 --max_iter 500 --sensitivity
```
Finds the periodic fixed point π* using cyclic wrapping. Saves `pi0.npy` + carry-over arrays.

### Step 2: Optimizer Comparison
```bash
python experiments/run_optimizers.py --n_samples 5 --commit 36 --sample_state \
    --initial_state_dir results/initial_state
```
Compares Full-day (open-loop) vs Greedy (myopic) vs MPC (receding-horizon). With `--sample_state`, Greedy/MPC sample the realized state at window boundaries.

### Step 3: Model Analysis
```bash
python experiments/run_model_analysis.py --analysis all --control_dir results/initial_state
```
Compares uniformization vs RK4 vs expm, and steady-state vs transient evaluation.

### Step 4: Sensitivity Analysis
```bash
python experiments/sensitivity_analysis.py --analysis all
```
Six analyses: delay, commit size, VoT, demand scaling, interval length, SS vs transient gap.

## Data Format

`Datasets/blr_demand_profile.csv`:
```csv
interval,time_start,lambda_T1,mu_d_T1,lambda_T2,mu_d_T2
1,00:00,11.2000,5.4000,7.0000,4.0000
2,00:05,18.4000,3.2000,7.4000,3.4000
...
```
- `lambda_T1`: passenger arrival rate at Terminal 1 (per 5-min interval)
- `mu_d_T1`: taxi drop-off rate at Terminal 1

## Color Scheme (Paper Figures)

All plots use the `paperfigs.sty` palette:
| Color | Hex | Role |
|-------|-----|------|
| Teal  | `#2A9D8F` | Full-day, transient-optimal |
| Rose  | `#E76F51` | Greedy, errors, gaps |
| Plum  | `#9B59B6` | MPC |
| Gray  | `#7F8C8D` | Do-nothing, baselines |
| Blue  | `#264653` | Demand (λ) overlay |
| Amber | `#E9C46A` | Supply (μ^d) overlay |
