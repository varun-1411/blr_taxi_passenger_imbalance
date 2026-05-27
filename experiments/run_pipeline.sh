#!/bin/bash
# ================================================================
# Complete Experiment Pipeline
#
# Usage:
#   bash experiments/run_pipeline.sh test      # 20 intervals, quick
#   bash experiments/run_pipeline.sh full      # 288 intervals, publication
# ================================================================

set -e
MODE=${1:-test}

if [ "$MODE" = "test" ]; then
    NI=20; COMMIT=5; NS=3; MI=100; EPS=0.1; NR=3
    echo "=== TEST MODE (20 intervals) ==="
elif [ "$MODE" = "full" ]; then
    NI=288; COMMIT=36; NS=5; MI=500; EPS=1.0; NR=5
    echo "=== FULL MODE (288 intervals) ==="
else
    echo "Usage: bash experiments/run_pipeline.sh [test|full]"; exit 1
fi

PI0_DIR="results/initial_state"
COMP_DIR="results/comparison"
SENS_DIR="results/sensitivity"

echo "  Intervals=$NI, Commit=$COMMIT, Samples=$NS, MaxIter=$MI"
echo ""

# ── Step 1: Initial State ──
echo "========================================"
echo "STEP 1: Initial State Calibration"
echo "========================================"
if [ -f "${PI0_DIR}/pi0.npy" ]; then
    echo "  Found existing pi0 at ${PI0_DIR}/pi0.npy — skipping"
else
    python experiments/find_initial_state.py \
        --n_intervals $NI --n_refine $NR --max_iter $MI \
        --epsilon $EPS --sensitivity --out_dir "$PI0_DIR"
fi

# ── Step 2: Quick Test (deterministic) ──
echo ""
echo "========================================"
echo "STEP 2: Quick Test (deterministic)"
echo "========================================"
python experiments/quick_test.py \
    --n_intervals $NI --commit $COMMIT --max_iter $MI \
    --epsilon $EPS --pi0 "${PI0_DIR}/pi0.npy" \
    --out_dir results/quick_test

# ── Step 3: Comparison (stochastic) ──
echo ""
echo "========================================"
echo "STEP 3: Full Comparison (stochastic)"
echo "========================================"
python experiments/run_optimizers.py \
    --n_intervals $NI --n_samples $NS --commit $COMMIT \
    --max_iter $MI --epsilon $EPS --sample_state \
    --initial_state_dir "$PI0_DIR" --out_dir "$COMP_DIR"

# ── Step 4: Model Analysis ──
echo ""
echo "========================================"
echo "STEP 4: Model Analysis (numerical + SS vs transient)"
echo "========================================"
python experiments/run_model_analysis.py \
    --n_intervals $NI --analysis all \
    --control_dir results/quick_test \
    --out_dir results/model_analysis

# ── Step 5: Optimizer Comparison (Brent, Adam, AIMD, RS) ──
echo ""
echo "========================================"
echo "STEP 5: Optimizer Comparison"
echo "========================================"
python experiments/run_optimizer_comparison.py \
    --methods brent adam aimd rs \
    --n_intervals $NI --max_iter $MI \
    --out_dir results/optimizer_comparison

# ── Step 6: Sensitivity ──
echo ""
echo "========================================"
echo "STEP 6: Sensitivity Analysis"
echo "========================================"
python experiments/sensitivity_analysis.py \
    --n_intervals $NI --max_iter $MI --epsilon $EPS \
    --analysis all --out_dir "$SENS_DIR"

echo ""
echo "========================================"
echo "DONE — Results:"
echo "  Initial state:      ${PI0_DIR}/"
echo "  Quick test:         results/quick_test/"
echo "  Comparison (GR/MPC):${COMP_DIR}/"
echo "  Optimizer compare:  results/optimizer_comparison/"
echo "  Model analysis:     results/model_analysis/"
echo "  Sensitivity:        ${SENS_DIR}/"
echo "========================================"
