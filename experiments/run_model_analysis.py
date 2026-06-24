"""
Model Analysis Experiments.

Three analyses:
  1. Numerical method comparison: Uniformization vs RK4 vs Expm
  2. Steady-state vs Transient (no delays) — clean comparison
  3. Four-way: (steady/transient) × (delay/no-delay)

These use run_simulation and run_steady_state_evaluation from model/metrics.py
which support multiple solvers.

Usage:
    python experiments/run_model_analysis.py --analysis all
    python experiments/run_model_analysis.py --analysis numerical
    python experiments/run_model_analysis.py --analysis steady_vs_transient
    python experiments/run_model_analysis.py --analysis four_way --control_dir results/adam_transient
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse, json, time
from dataclasses import replace
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import QueueConfig
from data import load_default_data
from model.metrics import run_simulation, run_steady_state_evaluation

# ══════════════════════════════════════════════════════════════
# PAPER COLOR SCHEME
# ══════════════════════════════════════════════════════════════

TEAL  = '#2A9D8F'
PLUM  = '#9B59B6'
AMBER = '#E9C46A'
GRAY  = '#7F8C8D'
BLUE  = '#264653'
ROSE  = '#E76F51'

# Method color assignments
C_UNIF = TEAL     # uniformization (reference)
C_RK4  = ROSE     # RK4
C_EXPM = PLUM     # matrix exponential
C_TRANSIENT = TEAL
C_STEADY    = ROSE
C_NODELAY   = AMBER
C_DONOTHING = GRAY

def setup_style():
    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 11,
        'axes.labelsize': 12, 'axes.titlesize': 13,
        'legend.fontsize': 9, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'axes.grid': True, 'grid.alpha': 0.3, 'lines.linewidth': 1.5,
    })

setup_style()


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def make_nodelay_config(config):
    """Zero all delays; inherit every other field from config."""
    return replace(config,
                   delay_non_reserved=0.0,
                   delay_ext_minutes=0.0)


def print_row(name, r):
    """Print one result row."""
    print(f"  {name:<30} {r['objective']:>12.2f} "
          f"{r['total_passenger_wait']:>10.2f} {r['total_taxi_idle_time']:>10.2f} "
          f"{r.get('total_passenger_block_time', 0):>10.4f}")


def print_header():
    print(f"  {'Method':<30} {'Objective':>12} {'Pax Wait':>10} "
          f"{'Taxi Idle':>10} {'Pax Block':>10}")
    print(f"  {'-'*74}")


def _save(data, out_dir, filename):
    os.makedirs(out_dir, exist_ok=True)
    def conv(o):
        if isinstance(o, (np.floating, np.integer)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, dict): return {str(k): conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [conv(v) for v in o]
        return o
    with open(os.path.join(out_dir, filename), 'w') as f:
        json.dump(conv(data), f, indent=2)


# ══════════════════════════════════════════════════════════════
# ANALYSIS 1: NUMERICAL METHOD COMPARISON
# ══════════════════════════════════════════════════════════════

def analysis_numerical(lambdas, mus_init, alpha1, alpha2, config,
                       mus_add, mus_removed, out_dir):
    """Compare uniformization, RK4, and expm on the same controls.
    All three are evaluated independently — pairwise differences show
    they converge to the same solution. Uniformization is then selected
    for its speed and autodiff compatibility."""
    print("\n" + "="*70)
    print("ANALYSIS: NUMERICAL METHOD COMPARISON")
    print("="*70)

    solvers = ['uniformization', 'rk4', 'expm']
    solver_labels = {'uniformization': 'Uniformization', 'rk4': 'RK4', 'expm': 'Expm (Krylov)'}
    colors = {'uniformization': C_UNIF, 'rk4': C_RK4, 'expm': C_EXPM}
    results = {}; timings = {}

    for solver in solvers:
        print(f"  Running {solver}...", end=' ', flush=True)
        t0 = time.time()
        r = run_simulation(
            lambdas=lambdas, mu_0=mus_init,
            alpha1=alpha1, alpha2=alpha2,
            mus_add=mus_add, mus_removed=mus_removed,
            config=config, solver=solver, verbose=False)
        elapsed = time.time() - t0
        results[solver] = r; timings[solver] = elapsed
        print(f"obj={r['objective']:.6f}, time={elapsed:.1f}s")

    # Results table
    print("\n")
    print_header()
    for s in solvers:
        print_row(f"{solver_labels[s]} ({timings[s]:.1f}s)", results[s])

    # Pairwise differences
    pairs = [('uniformization', 'rk4'), ('uniformization', 'expm'), ('rk4', 'expm')]
    print(f"\n  Pairwise differences (all three should agree):")
    print(f"  {'Pair':<25} {'|Δ Objective|':>14} {'|Δ Pax Wait|':>14} "
          f"{'|Δ Taxi Idle|':>14} {'Rel Obj Diff':>14}")
    print(f"  {'─'*83}")
    for sa, sb in pairs:
        ra, rb = results[sa], results[sb]
        d_obj = abs(ra['objective'] - rb['objective'])
        d_pax = abs(ra['total_passenger_wait'] - rb['total_passenger_wait'])
        d_taxi = abs(ra['total_taxi_idle_time'] - rb['total_taxi_idle_time'])
        avg_obj = (abs(ra['objective']) + abs(rb['objective'])) / 2
        rel = d_obj / avg_obj * 100 if avg_obj > 0 else 0
        label = f"{solver_labels[sa]} vs {solver_labels[sb]}"
        print(f"  {label:<25} {d_obj:>14.6f} {d_pax:>14.6f} {d_taxi:>14.6f} {rel:>13.6f}%")

    max_diff = max(abs(results[sa]['objective'] - results[sb]['objective'])
                   for sa, sb in pairs)
    agree_digits = max(0, int(-np.log10(max_diff / abs(results['uniformization']['objective']))))
    print(f"\n  All three methods agree to {agree_digits}+ significant digits.")
    print(f"  Uniformization selected: fastest ({timings['uniformization']:.1f}s) "
          f"and supports PyTorch autodiff.")

    # Speedup
    print(f"\n  Speedup relative to slowest:")
    slowest = max(timings.values())
    for s in solvers:
        speedup = slowest / timings[s] if timings[s] > 0 else float('inf')
        print(f"    {solver_labels[s]:<20}: {timings[s]:.1f}s ({speedup:.1f}×)")

    # Save
    save_data = {
        'results': {s: {
            'objective': results[s]['objective'],
            'pax_wait': results[s]['total_passenger_wait'],
            'taxi_idle': results[s]['total_taxi_idle_time'],
        } for s in solvers},
        'timings': timings,
        'pairwise_diffs': {
            f"{sa}_vs_{sb}": {
                'obj_diff': abs(results[sa]['objective'] - results[sb]['objective']),
                'pax_diff': abs(results[sa]['total_passenger_wait'] - results[sb]['total_passenger_wait']),
                'taxi_diff': abs(results[sa]['total_taxi_idle_time'] - results[sb]['total_taxi_idle_time']),
            } for sa, sb in pairs
        },
    }
    _save(save_data, out_dir, 'numerical_comparison.json')

    # ── Plot 1: Queue lengths overlay ──
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    titles = ['Passenger Queue', 'Taxi Queue (Pickup)', 'Staging Queue']
    keys = ['pax_queue_ts', 'taxi_queue_ts', 'resv_queue_ts']
    styles = {'uniformization': '-', 'rk4': '--', 'expm': '-.'}
    for ax, title, key in zip(axes, titles, keys):
        for s in solvers:
            ax.plot(results[s][key], label=solver_labels[s], color=colors[s],
                    lw=1.5, ls=styles[s], alpha=0.85)
        ax.set_ylabel('Expected Length'); ax.set_title(title); ax.legend()
    axes[-1].set_xlabel('Interval')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'numerical_queues.png'), dpi=300); plt.close()

    # ── Plot 2: Timing + autodiff support ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    bars = ax.bar(range(len(solvers)),
                  [timings[s] for s in solvers],
                  color=[colors[s] for s in solvers], alpha=0.85)
    ax.set_xticks(range(len(solvers)))
    ax.set_xticklabels([solver_labels[s] for s in solvers])
    for bar, t in zip(bars, [timings[s] for s in solvers]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                f'{t:.1f}s', ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('Time (s)'); ax.set_title('Computation Time')

    # Autodiff support indicator
    ax = axes[1]
    autodiff = {'uniformization': True, 'rk4': True, 'expm': False}
    bar_colors = [TEAL if autodiff[s] else GRAY for s in solvers]
    ax.bar(range(len(solvers)), [1 if autodiff[s] else 0 for s in solvers],
           color=bar_colors, alpha=0.85)
    ax.set_xticks(range(len(solvers)))
    ax.set_xticklabels([solver_labels[s] for s in solvers])
    ax.set_yticks([0, 1]); ax.set_yticklabels(['No', 'Yes'])
    ax.set_ylabel('Autodiff Support'); ax.set_title('Gradient Computation')
    ax.set_ylim(-0.1, 1.3)
    for i, s in enumerate(solvers):
        ax.text(i, 0.5, 'Yes' if autodiff[s] else 'No',
                ha='center', va='center', fontsize=11, fontweight='bold',
                color='white' if autodiff[s] else 'black')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'numerical_timing.png'), dpi=300); plt.close()

    # ── Plot 3: Pairwise differences per interval ──
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    pair_colors = {
        ('uniformization', 'rk4'): C_RK4,
        ('uniformization', 'expm'): C_EXPM,
        ('rk4', 'expm'): AMBER,
    }
    pair_labels = {
        ('uniformization', 'rk4'): 'Unif $-$ RK4',
        ('uniformization', 'expm'): 'Unif $-$ Expm',
        ('rk4', 'expm'): 'RK4 $-$ Expm',
    }
    for ax, title, key in zip(axes, titles, keys):
        for (sa, sb), pc in pair_colors.items():
            diff = np.array(results[sa][key]) - np.array(results[sb][key])
            ax.plot(diff, color=pc, lw=1.0, label=pair_labels[(sa, sb)], alpha=0.85)
        ax.axhline(0, color='black', lw=0.5)
        ax.set_ylabel('Difference')
        ax.set_title(f'{title} — Pairwise Differences')
        ax.legend(fontsize=8)
    axes[-1].set_xlabel('Interval')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'numerical_differences.png'), dpi=300); plt.close()

    # ── Plot 4: Summary figure for paper ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel 1: Objectives (should be nearly identical)
    ax = axes[0]
    objs = [results[s]['objective'] for s in solvers]
    bars = ax.bar(range(len(solvers)), objs,
                  color=[colors[s] for s in solvers], alpha=0.85)
    ax.set_xticks(range(len(solvers)))
    ax.set_xticklabels([solver_labels[s] for s in solvers], fontsize=9)
    ax.set_ylabel('Objective'); ax.set_title('Solution Quality')
    # Zoom y-axis to show differences
    if max(objs) > 0:
        margin = max(max(objs) - min(objs), max(objs) * 0.001)
        ax.set_ylim(min(objs) - margin * 2, max(objs) + margin * 2)
    for bar, obj in zip(bars, objs):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                f'{obj:.2f}', ha='center', va='bottom', fontsize=8)

    # Panel 2: Runtime
    ax = axes[1]
    bars = ax.bar(range(len(solvers)), [timings[s] for s in solvers],
                  color=[colors[s] for s in solvers], alpha=0.85)
    ax.set_xticks(range(len(solvers)))
    ax.set_xticklabels([solver_labels[s] for s in solvers], fontsize=9)
    ax.set_ylabel('Time (s)'); ax.set_title('Runtime')
    for bar, t in zip(bars, [timings[s] for s in solvers]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                f'{t:.1f}s', ha='center', va='bottom', fontsize=9)

    # Panel 3: Max pairwise queue difference
    ax = axes[2]
    pair_names = [pair_labels[p] for p in pairs]
    max_diffs = []
    for sa, sb in pairs:
        md = max(
            np.abs(np.array(results[sa][k]) - np.array(results[sb][k])).max()
            for k in keys if k in results[sa] and k in results[sb]
        )
        max_diffs.append(md)
    pc = [pair_colors[p] for p in pairs]
    bars = ax.bar(range(len(pairs)), max_diffs, color=pc, alpha=0.85)
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(pair_names, fontsize=9)
    ax.set_ylabel('Max |Difference|'); ax.set_title('Agreement (Queue Lengths)')
    for bar, d in zip(bars, max_diffs):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                f'{d:.2e}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'numerical_summary.png'), dpi=300); plt.close()

    print(f"  Plots saved to {out_dir}/")
    return results, timings


# ══════════════════════════════════════════════════════════════
# ANALYSIS 2: STEADY-STATE vs TRANSIENT (paper Table)
# ══════════════════════════════════════════════════════════════

def analysis_steady_vs_transient(lambdas, mus_init, alpha1, alpha2, config,
                                 mus_add, mus_removed, out_dir,
                                 max_iter=None):
    """
    Paper comparison: SS vs Transient.

    Part A — Zero delays (isolates stationarity error):
      1. Do-nothing (transient, d=0)
      2. Brent SS-optimal → SS-reported objective
      3. Brent SS-optimal → evaluate on transient → true cost
         Gap (2→3) = stationarity error (within-interval non-convergence)
      4. Adam transient-optimal → best achievable
         Gap (3→4) = additional gain from transient optimization

    Part B — With delays (full model):
      5. Do-nothing (transient, with delays)
      6. Brent SS controls → evaluate on transient WITH delays
      7. Adam transient-optimal with delays → best achievable
    """
    print("\n" + "="*70)
    print("ANALYSIS: STEADY-STATE vs TRANSIENT")
    print("="*70)

    cfg_nd = make_nodelay_config(config)
    n = len(lambdas)

    # Import optimizers
    from optimizers.brent_optimizer import run_brent_steady_state
    from optimizers.adam_optimizer import run_adam_transient
    adam_max = max_iter or config.adam_max_iter

    # ────────────────────────────────────────────────────────
    # PART A: Zero delays
    # ────────────────────────────────────────────────────────
    print(f"\n  PART A: Zero delays (d_d = d_e = 0)")
    print(f"  {'─'*60}")

    # 1. Do-nothing (transient, zero delay)
    print(f"  1. Do-nothing (transient, d=0)...", end=' ', flush=True)
    dn_nd = run_simulation(
        lambdas, mus_init, alpha1, alpha2,
        np.zeros(n), np.zeros(n), cfg_nd, solver='uniformization', verbose=False)
    dn_nd['mu_add'] = np.zeros(n); dn_nd['mu_remove'] = np.zeros(n)
    print(f"obj={dn_nd['objective']:.2f}")

    # 2. Brent SS-optimal
    brent_dir = os.path.join(out_dir, 'brent_ss')
    os.makedirs(brent_dir, exist_ok=True)
    print(f"  2. Brent SS optimizer...", end=' ', flush=True)
    t0 = time.time()
    brent = run_brent_steady_state(
        lambdas, mus_init, alpha1, alpha2, cfg_nd,
        delay_method='none', out_dir=brent_dir, verbose=False)
    brent_time = time.time() - t0
    print(f"SS-reported obj={brent['objective']:.2f} ({brent_time:.1f}s)")

    # 3. Evaluate Brent controls on transient model (zero delay)
    print(f"  3. Brent controls → transient eval...", end=' ', flush=True)
    brent_on_tr = run_simulation(
        lambdas, mus_init, alpha1, alpha2,
        brent['mu_add'], brent['mu_remove'],
        cfg_nd, solver='uniformization', verbose=False)
    print(f"true cost={brent_on_tr['objective']:.2f}")

    # Stationarity error
    ss_reported = brent['objective']
    ss_true = brent_on_tr['objective']
    stationarity_error = ss_true - ss_reported
    stationarity_pct = stationarity_error / ss_reported * 100

    # 4. Adam transient-optimal (zero delay)
    adam_nd_dir = os.path.join(out_dir, 'adam_nodelay')
    os.makedirs(adam_nd_dir, exist_ok=True)
    print(f"  4. Adam transient optimizer (d=0)...", end=' ', flush=True)
    t0 = time.time()
    adam_nd = run_adam_transient(
        lambdas, mus_init, alpha1, alpha2, cfg_nd,
        max_iterations=adam_max, epsilon=config.adam_epsilon,
        lr=config.adam_lr, out_dir=adam_nd_dir)
    adam_nd_time = time.time() - t0
    print(f"obj={adam_nd['objective']:.2f} ({adam_nd_time:.1f}s)")

    # Evaluate Adam on transient (should match, but explicit)
    adam_nd_eval = run_simulation(
        lambdas, mus_init, alpha1, alpha2,
        adam_nd['mu_add'], adam_nd['mu_remove'],
        cfg_nd, solver='uniformization', verbose=False)

    # Transient improvement over SS
    transient_gain = ss_true - adam_nd_eval['objective']
    transient_gain_pct = transient_gain / ss_true * 100

    # Part A summary
    print(f"\n  {'─'*70}")
    print(f"  PART A RESULTS (zero delays, d_d = d_e = 0)")
    print(f"  {'─'*70}")
    print(f"  {'Scenario':<40} {'Objective':>12} {'Pax Wait':>10} {'Taxi Idle':>10}")
    print(f"  {'─'*70}")
    print(f"  {'Do-nothing (transient)':<40} {dn_nd['objective']:>12.2f} "
          f"{dn_nd['total_passenger_wait']:>10.2f} {dn_nd['total_taxi_idle_time']:>10.2f}")
    print(f"  {'Brent SS-reported':<40} {ss_reported:>12.2f}")
    print(f"  {'Brent controls on transient':<40} {ss_true:>12.2f} "
          f"{brent_on_tr['total_passenger_wait']:>10.2f} {brent_on_tr['total_taxi_idle_time']:>10.2f}")
    print(f"  {'Adam transient-optimal':<40} {adam_nd_eval['objective']:>12.2f} "
          f"{adam_nd_eval['total_passenger_wait']:>10.2f} {adam_nd_eval['total_taxi_idle_time']:>10.2f}")
    print(f"\n  Stationarity error (Brent true - SS reported):")
    print(f"    {stationarity_error:+.2f} ({stationarity_pct:+.1f}%)")
    print(f"  Transient gain (Brent true - Adam transient):")
    print(f"    {transient_gain:+.2f} ({transient_gain_pct:+.1f}%)")
    total_gap = ss_true - adam_nd_eval['objective']
    total_gap_from_dn = dn_nd['objective'] - adam_nd_eval['objective']
    print(f"  Total improvement (DN - Adam):")
    print(f"    {total_gap_from_dn:+.2f} ({total_gap_from_dn/dn_nd['objective']*100:+.1f}%)")

    # ────────────────────────────────────────────────────────
    # PART B: With delays
    # ────────────────────────────────────────────────────────
    print(f"\n  PART B: With delays (d_d={config.delay_non_reserved}, d_e={config.delay_ext_minutes})")
    print(f"  {'─'*60}")

    # 5. Do-nothing (with delays)
    print(f"  5. Do-nothing (with delays)...", end=' ', flush=True)
    dn_d = run_simulation(
        lambdas, mus_init, alpha1, alpha2,
        np.zeros(n), np.zeros(n), config, solver='uniformization', verbose=False)
    print(f"obj={dn_d['objective']:.2f}")

    # 6. Brent SS controls → transient WITH delays
    print(f"  6. Brent controls → transient with delays...", end=' ', flush=True)
    brent_on_delay = run_simulation(
        lambdas, mus_init, alpha1, alpha2,
        brent['mu_add'], brent['mu_remove'],
        config, solver='uniformization', verbose=False)
    print(f"obj={brent_on_delay['objective']:.2f}")

    # 7. Adam transient-optimal WITH delays
    adam_d_dir = os.path.join(out_dir, 'adam_delay')
    os.makedirs(adam_d_dir, exist_ok=True)
    print(f"  7. Adam transient optimizer (with delays)...", end=' ', flush=True)
    t0 = time.time()
    adam_d = run_adam_transient(
        lambdas, mus_init, alpha1, alpha2, config,
        max_iterations=adam_max, epsilon=config.adam_epsilon,
        lr=config.adam_lr, out_dir=adam_d_dir)
    adam_d_time = time.time() - t0
    print(f"obj={adam_d['objective']:.2f} ({adam_d_time:.1f}s)")

    adam_d_eval = run_simulation(
        lambdas, mus_init, alpha1, alpha2,
        adam_d['mu_add'], adam_d['mu_remove'],
        config, solver='uniformization', verbose=False)

    # Part B summary
    delay_cost_brent = brent_on_delay['objective'] - ss_true
    delay_cost_adam = adam_d_eval['objective'] - adam_nd_eval['objective']

    print(f"\n  {'─'*70}")
    print(f"  PART B RESULTS (with delays)")
    print(f"  {'─'*70}")
    print(f"  {'Scenario':<40} {'Objective':>12} {'Pax Wait':>10} {'Taxi Idle':>10}")
    print(f"  {'─'*70}")
    print(f"  {'Do-nothing (with delays)':<40} {dn_d['objective']:>12.2f} "
          f"{dn_d['total_passenger_wait']:>10.2f} {dn_d['total_taxi_idle_time']:>10.2f}")
    print(f"  {'Brent SS controls + delays':<40} {brent_on_delay['objective']:>12.2f} "
          f"{brent_on_delay['total_passenger_wait']:>10.2f} {brent_on_delay['total_taxi_idle_time']:>10.2f}")
    print(f"  {'Adam transient + delays':<40} {adam_d_eval['objective']:>12.2f} "
          f"{adam_d_eval['total_passenger_wait']:>10.2f} {adam_d_eval['total_taxi_idle_time']:>10.2f}")
    print(f"\n  Cost of delays:")
    print(f"    On Brent controls: {delay_cost_brent:+.2f} ({delay_cost_brent/ss_true*100:+.1f}%)")
    print(f"    On Adam controls:  {delay_cost_adam:+.2f} ({delay_cost_adam/adam_nd_eval['objective']*100:+.1f}%)")

    # ────────────────────────────────────────────────────────
    # SAVE EVERYTHING
    # ────────────────────────────────────────────────────────
    all_results = {
        'part_a': {
            'do_nothing': dn_nd, 'brent_ss_reported': ss_reported,
            'brent_on_transient': brent_on_tr, 'adam_transient': adam_nd_eval,
        },
        'part_b': {
            'do_nothing_delay': dn_d,
            'brent_on_delay': brent_on_delay, 'adam_delay': adam_d_eval,
        },
        'controls': {
            'brent': {'mu_add': brent['mu_add'], 'mu_remove': brent['mu_remove']},
            'adam_nodelay': {'mu_add': adam_nd['mu_add'], 'mu_remove': adam_nd['mu_remove']},
            'adam_delay': {'mu_add': adam_d['mu_add'], 'mu_remove': adam_d['mu_remove']},
        },
    }

    # Save controls
    for name, ctrl in all_results['controls'].items():
        np.save(os.path.join(out_dir, f'{name}_mu_add.npy'), ctrl['mu_add'])
        np.save(os.path.join(out_dir, f'{name}_mu_remove.npy'), ctrl['mu_remove'])

    # Save queue time series
    for name, r in [('dn_nodelay', dn_nd), ('brent_on_tr', brent_on_tr),
                    ('adam_nd_eval', adam_nd_eval), ('dn_delay', dn_d),
                    ('brent_on_delay', brent_on_delay), ('adam_d_eval', adam_d_eval)]:
        for key in ['pax_queue_ts', 'taxi_queue_ts', 'resv_queue_ts']:
            if key in r:
                np.save(os.path.join(out_dir, f'{name}_{key}.npy'), np.array(r[key]))

    _save({
        'part_a': {
            'do_nothing': float(dn_nd['objective']),
            'brent_ss_reported': float(ss_reported),
            'brent_on_transient': float(ss_true),
            'adam_transient': float(adam_nd_eval['objective']),
            'stationarity_error': float(stationarity_error),
            'stationarity_pct': float(stationarity_pct),
            'transient_gain': float(transient_gain),
            'transient_gain_pct': float(transient_gain_pct),
        },
        'part_b': {
            'do_nothing_delay': float(dn_d['objective']),
            'brent_on_delay': float(brent_on_delay['objective']),
            'adam_delay': float(adam_d_eval['objective']),
            'delay_cost_brent': float(delay_cost_brent),
            'delay_cost_adam': float(delay_cost_adam),
        },
        'timings': {
            'brent': float(brent_time),
            'adam_nodelay': float(adam_nd_time),
            'adam_delay': float(adam_d_time),
        },
    }, out_dir, 'ss_vs_transient.json')

    # ────────────────────────────────────────────────────────
    # PLOTS
    # ────────────────────────────────────────────────────────
    t_axis = np.arange(n) * config.interval_length
    titles = ['Passenger Queue', 'Taxi Queue (Pickup)', 'Staging Queue']
    keys = ['pax_queue_ts', 'taxi_queue_ts', 'resv_queue_ts']

    # Plot 1: Part A — queue lengths (SS controls vs transient controls, zero delay)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for ax, title, key in zip(axes, titles, keys):
        ax.plot(brent_on_tr[key], color=AMBER, lw=1.5, ls='--',
                label='Brent SS controls (on transient)')
        ax.plot(adam_nd_eval[key], color=TEAL, lw=1.8,
                label='Adam transient-optimal')
        ax.plot(dn_nd[key], color=GRAY, lw=1.0, ls=':',
                label='Do-nothing', alpha=0.7)
        ax.set_ylabel('Expected Length'); ax.set_title(title); ax.legend(fontsize=8)
    axes[-1].set_xlabel('Interval')
    fig.suptitle('Zero Delays: SS vs Transient Controls', fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'ss_vs_tr_queues.png'), dpi=300); plt.close()

    # Plot 2: Control profiles comparison
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    ax = axes[0]
    ax2 = ax.twinx()
    ax2.fill_between(t_axis, lambdas, alpha=0.08, color=BLUE)
    ax2.set_ylabel(r'$\lambda$', color=BLUE)
    ax.plot(t_axis, brent['mu_add'], color=AMBER, lw=1.5, ls='--', label='Brent SS')
    ax.plot(t_axis, adam_nd['mu_add'], color=TEAL, lw=1.8, label='Adam (d=0)')
    ax.plot(t_axis, adam_d['mu_add'], color=ROSE, lw=1.5, ls='-.', label='Adam (with delay)')
    ax.set_ylabel(r'$\mu^+$'); ax.set_title('Dispatch Rate'); ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(t_axis, brent['mu_remove'], color=AMBER, lw=1.5, ls='--', label='Brent SS')
    ax.plot(t_axis, adam_nd['mu_remove'], color=TEAL, lw=1.8, label='Adam (d=0)')
    ax.plot(t_axis, adam_d['mu_remove'], color=ROSE, lw=1.5, ls='-.', label='Adam (with delay)')
    ax.set_xlabel('Time (min)'); ax.set_ylabel(r'$\mu^-$')
    ax.set_title('Removal Rate'); ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'ss_vs_tr_controls.png'), dpi=300); plt.close()

    # Plot 3: Paper summary figure — bar chart with gap annotations
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Part A bars
    ax = axes[0]
    labels_a = ['Do-Nothing', 'Brent\n(SS reported)', 'Brent on\nTransient', 'Adam\nTransient']
    objs_a = [dn_nd['objective'], ss_reported, ss_true, adam_nd_eval['objective']]
    colors_a = [GRAY, AMBER, AMBER, TEAL]
    alphas_a = [0.6, 0.5, 0.85, 0.85]
    edges_a = ['none', 'none', AMBER, TEAL]
    lws_a = [0, 0, 2, 2]
    for i in range(4):
        ax.bar(i, objs_a[i], color=colors_a[i], alpha=alphas_a[i],
               edgecolor=edges_a[i], linewidth=lws_a[i])
    for i in range(4):
        ax.text(i, objs_a[i], f'{objs_a[i]:,.0f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(range(4)); ax.set_xticklabels(labels_a, fontsize=8)
    ax.set_ylabel('Objective'); ax.set_title('Part A: Zero Delays ($d_d = d_e = 0$)')

    # Annotate stationarity error
    ax.annotate('', xy=(2, ss_true), xytext=(1, ss_reported),
                arrowprops=dict(arrowstyle='<->', color=ROSE, lw=1.5))
    mid_y = (ss_reported + ss_true) / 2
    ax.text(1.5, mid_y, f'Stationarity\nerror: {stationarity_pct:+.1f}%',
            ha='center', va='center', fontsize=7, color=ROSE,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))

    # Annotate transient gain
    ax.annotate('', xy=(3, adam_nd_eval['objective']), xytext=(2, ss_true),
                arrowprops=dict(arrowstyle='<->', color=TEAL, lw=1.5))
    mid_y2 = (ss_true + adam_nd_eval['objective']) / 2
    ax.text(2.5, mid_y2, f'Transient\ngain: {transient_gain_pct:+.1f}%',
            ha='center', va='center', fontsize=7, color=TEAL,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))

    # Part B bars
    ax = axes[1]
    labels_b = ['Do-Nothing\n(with delay)', 'Brent SS\n+ delays', 'Adam\n+ delays']
    objs_b = [dn_d['objective'], brent_on_delay['objective'], adam_d_eval['objective']]
    colors_b = [GRAY, AMBER, TEAL]
    bars = ax.bar(range(3), objs_b, color=colors_b, alpha=0.85)
    for bar, obj in zip(bars, objs_b):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                f'{obj:,.0f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(range(3)); ax.set_xticklabels(labels_b, fontsize=8)
    ax.set_ylabel('Objective')
    ax.set_title(f'Part B: With Delays ($d_d={config.delay_non_reserved:.0f}$, '
                 f'$d_e={config.delay_ext_minutes:.0f}$)')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'ss_vs_tr_summary.png'), dpi=300); plt.close()

    # Plot 4: Per-interval difference (Brent vs Adam queue lengths, zero delay)
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    for ax, title, key in zip(axes, titles, keys):
        diff = np.array(brent_on_tr[key]) - np.array(adam_nd_eval[key])
        ax.plot(diff, color=PLUM, lw=1.2)
        ax.axhline(0, color='black', lw=0.5)
        ax.fill_between(range(len(diff)), diff, alpha=0.15, color=PLUM)
        ax.set_ylabel('Brent $-$ Adam')
        ax.set_title(f'{title} — Stationarity Gap (per interval)')
    axes[-1].set_xlabel('Interval')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'ss_vs_tr_gap.png'), dpi=300); plt.close()

    print(f"\n  Plots saved to {out_dir}/")
    return all_results


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Analysis')
    parser.add_argument('--analysis', type=str, default='all',
                        choices=['all', 'numerical', 'steady_vs_transient'])
    parser.add_argument('--control_dir', type=str, default=None,
                        help='Directory with mu_add.npy and mu_remove.npy '
                             '(for numerical comparison; SS analysis runs its own optimizers)')
    parser.add_argument('--n_intervals', type=int, default=None)
    parser.add_argument('--max_iter', type=int, default=None,
                        help='Max Adam iterations for SS vs transient (default: config)')
    parser.add_argument('--out_dir', type=str, default='results/model_analysis')
    args = parser.parse_args()

    config = QueueConfig()
    lambdas, mus_init = load_default_data(config)
    if args.n_intervals:
        lambdas = lambdas[:args.n_intervals]
        mus_init = mus_init[:args.n_intervals]
    alpha1, alpha2 = config.get_alpha_arrays(size=len(lambdas))

    # Load controls for numerical comparison (or use zeros)
    n = len(lambdas)
    if args.control_dir and os.path.exists(args.control_dir):
        mus_add = np.load(os.path.join(args.control_dir, 'mu_add.npy'))[:n]
        mus_removed = np.load(os.path.join(args.control_dir, 'mu_remove.npy'))[:n]
        print(f"  Loaded controls from {args.control_dir}")
    else:
        mus_add = np.zeros(n)
        mus_removed = np.zeros(n)
        print(f"  Using zero controls for numerical comparison")

    os.makedirs(args.out_dir, exist_ok=True)

    print("="*70)
    print("MODEL ANALYSIS")
    print("="*70)
    print(f"  Intervals: {n}")
    print(f"  States:    {(config.K_S+1)*(config.K_P+config.M+1)}")
    print(f"  Delays:    d_d={config.delay_non_reserved}, d_e={config.delay_ext_minutes}")
    print(f"  Analysis:  {args.analysis}")
    print("="*70)

    t0 = time.time()

    if args.analysis in ('all', 'numerical'):
        analysis_numerical(lambdas, mus_init, alpha1, alpha2, config,
                          mus_add, mus_removed, args.out_dir)

    if args.analysis in ('all', 'steady_vs_transient'):
        analysis_steady_vs_transient(lambdas, mus_init, alpha1, alpha2, config,
                                    mus_add, mus_removed, args.out_dir,
                                    max_iter=args.max_iter)

    print(f"\nTotal: {time.time()-t0:.1f}s. Results in {args.out_dir}/")