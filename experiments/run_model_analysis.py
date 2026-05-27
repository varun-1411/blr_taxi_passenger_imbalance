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
    """Create config copy with zero delays."""
    return QueueConfig(
        K_S=config.K_S, K_P=config.K_P, M=config.M,
        tau=config.tau,
        time_horizon=config.time_horizon,
        interval_length=config.interval_length,
        group_size=config.group_size,
        delay_non_reserved=0.0,
        delay_extra=0.0,
    )


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
    """Compare uniformization, RK4, and expm on the same controls."""
    print("\n" + "="*70)
    print("ANALYSIS: NUMERICAL METHOD COMPARISON")
    print("="*70)

    solvers = ['uniformization', 'rk4', 'expm']
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
        print(f"obj={r['objective']:.4f}, time={elapsed:.1f}s")

    # Comparison table
    print("\n")
    print_header()
    for s in solvers:
        print_row(f"{s} ({timings[s]:.1f}s)", results[s])

    # Error relative to uniformization
    ref = results['uniformization']
    print(f"\n  Error relative to uniformization:")
    print(f"  {'Method':<15} {'Obj Error':>12} {'Pax Error':>12} {'Taxi Error':>12} {'Rel Obj %':>12}")
    print(f"  {'-'*55}")
    for s in ['rk4', 'expm']:
        r = results[s]
        oe = abs(r['objective'] - ref['objective'])
        pe = abs(r['total_passenger_wait'] - ref['total_passenger_wait'])
        te = abs(r['total_taxi_idle_time'] - ref['total_taxi_idle_time'])
        re = oe / abs(ref['objective']) * 100
        print(f"  {s:<15} {oe:>12.6f} {pe:>12.6f} {te:>12.6f} {re:>12.6f}%")

    _save({'results': {s: {'objective': results[s]['objective'],
                           'pax_wait': results[s]['total_passenger_wait'],
                           'taxi_idle': results[s]['total_taxi_idle_time']}
                       for s in solvers},
           'timings': timings}, out_dir, 'numerical_comparison.json')

    # Plot 1: Queue lengths overlay
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    titles = ['Passenger Queue', 'Taxi Queue (Pickup)', 'Staging Queue']
    keys = ['pax_queue_ts', 'taxi_queue_ts', 'resv_queue_ts']
    for ax, title, key in zip(axes, titles, keys):
        for s in solvers:
            ax.plot(results[s][key], label=s, color=colors[s], lw=1.5,
                    ls='--' if s != 'uniformization' else '-', alpha=0.85)
        ax.set_ylabel('Expected Length'); ax.set_title(title); ax.legend()
    axes[-1].set_xlabel('Interval')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'numerical_queues.png'), dpi=300); plt.close()

    # Plot 2: Timing bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(solvers, [timings[s] for s in solvers],
                  color=[colors[s] for s in solvers], alpha=0.85)
    for bar, t in zip(bars, [timings[s] for s in solvers]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                f'{t:.1f}s', ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('Time (s)'); ax.set_title('Computation Time by Method')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'numerical_timing.png'), dpi=300); plt.close()

    # Plot 3: Per-interval error (RK4 and expm vs uniformization)
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    for ax, title, key in zip(axes, titles, keys):
        ref_arr = np.array(ref[key])
        for s in ['rk4', 'expm']:
            err = np.array(results[s][key]) - ref_arr
            ax.plot(err, color=colors[s], lw=1.2, label=f'{s} $-$ unif')
        ax.axhline(0, color='black', lw=0.5)
        ax.set_ylabel('Error'); ax.set_title(f'{title} — Error vs Uniformization')
        ax.legend()
    axes[-1].set_xlabel('Interval')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'numerical_error.png'), dpi=300); plt.close()

    print(f"  Plots saved to {out_dir}/")
    return results, timings


# ══════════════════════════════════════════════════════════════
# ANALYSIS 2: STEADY-STATE vs TRANSIENT
# ══════════════════════════════════════════════════════════════

def analysis_steady_vs_transient(lambdas, mus_init, alpha1, alpha2, config,
                                 mus_add, mus_removed, out_dir):
    """
    Compare steady-state vs transient with NO delays.
    Clean comparison: same rates, no delay confounding.
    """
    print("\n" + "="*70)
    print("ANALYSIS: STEADY-STATE vs TRANSIENT (No Delays)")
    print("="*70)

    cfg_nd = make_nodelay_config(config)

    # Transient (no delay)
    print(f"  Running transient...", end=' ', flush=True)
    t0 = time.time()
    transient = run_simulation(
        lambdas=lambdas, mu_0=mus_init,
        alpha1=alpha1, alpha2=alpha2,
        mus_add=mus_add, mus_removed=mus_removed,
        config=cfg_nd, solver='uniformization', verbose=False)
    t_tr = time.time() - t0
    print(f"obj={transient['objective']:.2f} ({t_tr:.1f}s)")

    # Steady-state (no delay)
    print(f"  Running steady-state...", end=' ', flush=True)
    t0 = time.time()
    steady = run_steady_state_evaluation(
        lambdas=lambdas, mu_0=mus_init,
        alpha1=alpha1, alpha2=alpha2,
        mus_add=mus_add, mus_removed=mus_removed,
        config=cfg_nd, verbose=False)
    t_ss = time.time() - t0
    print(f"obj={steady['objective']:.2f} ({t_ss:.1f}s)")

    # Table
    print("\n")
    print_header()
    print_row(f"Transient ({t_tr:.1f}s)", transient)
    print_row(f"Steady-State ({t_ss:.1f}s)", steady)

    # Gap
    gap = steady['objective'] - transient['objective']
    gap_pct = gap / transient['objective'] * 100
    print(f"\n  Gap (SS - Transient): {gap:+.2f} ({gap_pct:+.3f}%)")

    _save({
        'transient': {'objective': transient['objective'],
                      'pax_wait': transient['total_passenger_wait'],
                      'taxi_idle': transient['total_taxi_idle_time']},
        'steady': {'objective': steady['objective'],
                   'pax_wait': steady['total_passenger_wait'],
                   'taxi_idle': steady['total_taxi_idle_time']},
        'gap': gap, 'gap_pct': gap_pct,
    }, out_dir, 'steady_vs_transient.json')

    # Plot 1: Queue lengths overlay
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    titles = ['Passenger Queue', 'Taxi Queue (Pickup)', 'Staging Queue']
    keys = ['pax_queue_ts', 'taxi_queue_ts', 'resv_queue_ts']
    for ax, title, key in zip(axes, titles, keys):
        ax.plot(transient[key], color=C_TRANSIENT, lw=1.8, label='Transient')
        ax.plot(steady[key], color=C_STEADY, lw=1.5, ls='--', label='Steady-State')
        ax.set_ylabel('Expected Length'); ax.set_title(title); ax.legend()
    axes[-1].set_xlabel('Interval')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'ss_vs_tr_queues.png'), dpi=300); plt.close()

    # Plot 2: Per-interval error
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    for ax, title, key in zip(axes, titles, keys):
        t_arr = np.array(transient[key]); s_arr = np.array(steady[key])
        ax.plot(s_arr - t_arr, color=PLUM, lw=1.2)
        ax.axhline(0, color='black', lw=0.5)
        ax.set_ylabel('Error (SS $-$ TR)'); ax.set_title(title)
    axes[-1].set_xlabel('Interval')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'ss_vs_tr_error.png'), dpi=300); plt.close()

    # Plot 3: Relative error
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    for ax, title, key in zip(axes, titles, keys):
        t_arr = np.array(transient[key]); s_arr = np.array(steady[key])
        with np.errstate(divide='ignore', invalid='ignore'):
            rel = np.where(t_arr != 0, (s_arr - t_arr) / np.abs(t_arr) * 100, 0)
        ax.plot(rel, color=PLUM, lw=1.2)
        ax.axhline(0, color='black', lw=0.5)
        ax.set_ylabel('Relative Error (%)'); ax.set_title(title)
    axes[-1].set_xlabel('Interval')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'ss_vs_tr_relerr.png'), dpi=300); plt.close()

    print(f"  Plots saved to {out_dir}/")
    return {'transient': transient, 'steady': steady}


# ══════════════════════════════════════════════════════════════
# ANALYSIS 3: FOUR-WAY COMPARISON
# ══════════════════════════════════════════════════════════════

def analysis_four_way(lambdas, mus_init, alpha1, alpha2, config,
                      mus_add, mus_removed, out_dir):
    """
    4-way comparison: (steady/transient) × (delay/no-delay).
    Evaluates the same controls in all four model variants.
    """
    print("\n" + "="*70)
    print("ANALYSIS: FOUR-WAY COMPARISON")
    print(f"  Delays: d_d={config.delay_non_reserved}, d_e={config.delay_extra}")
    print("="*70)

    cfg_nd = make_nodelay_config(config)

    cases = {}
    labels = {
        'transient_delay':   ('Transient + Delay', TEAL, '-'),
        'transient_nodelay': ('Transient (no delay)', TEAL, '--'),
        'steady_delay':      ('Steady + Delay', ROSE, '-'),
        'steady_nodelay':    ('Steady (no delay)', ROSE, '--'),
    }

    print(f"  1/4: Transient with delay...", end=' ', flush=True)
    cases['transient_delay'] = run_simulation(
        lambdas, mus_init, alpha1, alpha2, mus_add, mus_removed,
        config, solver='uniformization')
    print(f"obj={cases['transient_delay']['objective']:.2f}")

    print(f"  2/4: Transient no delay...", end=' ', flush=True)
    cases['transient_nodelay'] = run_simulation(
        lambdas, mus_init, alpha1, alpha2, mus_add, mus_removed,
        cfg_nd, solver='uniformization')
    print(f"obj={cases['transient_nodelay']['objective']:.2f}")

    print(f"  3/4: Steady with delay...", end=' ', flush=True)
    cases['steady_delay'] = run_steady_state_evaluation(
        lambdas, mus_init, alpha1, alpha2, mus_add, mus_removed, config)
    print(f"obj={cases['steady_delay']['objective']:.2f}")

    print(f"  4/4: Steady no delay...", end=' ', flush=True)
    cases['steady_nodelay'] = run_steady_state_evaluation(
        lambdas, mus_init, alpha1, alpha2, mus_add, mus_removed, cfg_nd)
    print(f"obj={cases['steady_nodelay']['objective']:.2f}")

    # Do-nothing baseline
    donothing = run_simulation(
        lambdas, mus_init, alpha1, alpha2,
        np.zeros_like(mus_add), np.zeros_like(mus_removed),
        config, solver='uniformization')

    # Table
    print("\n")
    print_header()
    print_row("Do-Nothing", donothing)
    for key in cases:
        label = labels[key][0]
        print_row(label, cases[key])

    # Gap analysis
    ref = cases['transient_delay']['objective']
    print(f"\n  Gaps from Transient+Delay (reference):")
    for key in cases:
        g = cases[key]['objective'] - ref
        gp = g / ref * 100
        print(f"    {labels[key][0]:<30}: {g:+.2f} ({gp:+.3f}%)")

    _save({key: {'objective': cases[key]['objective'],
                  'pax_wait': cases[key]['total_passenger_wait'],
                  'taxi_idle': cases[key]['total_taxi_idle_time']}
           for key in cases}, out_dir, 'four_way.json')

    # Plot 1: Objective bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    names = ['Do-Nothing'] + [labels[k][0] for k in cases]
    objs = [donothing['objective']] + [cases[k]['objective'] for k in cases]
    bar_colors = [C_DONOTHING] + [labels[k][1] for k in cases]
    alphas = [0.6] + [0.85 if labels[k][2]=='-' else 0.5 for k in cases]

    bars = ax.bar(range(len(names)), objs, color=bar_colors, alpha=0.85)
    for bar, obj in zip(bars, objs):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                f'{obj:.0f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('Objective'); ax.set_title('Four-Way Comparison')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'four_way_objectives.png'), dpi=300); plt.close()

    # Plot 2: Queue lengths all four
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    titles = ['Passenger Queue', 'Taxi Queue', 'Staging Queue']
    keys_ts = ['pax_queue_ts', 'taxi_queue_ts', 'resv_queue_ts']
    for ax, title, key in zip(axes, titles, keys_ts):
        for case_key in cases:
            label, color, ls = labels[case_key]
            ax.plot(cases[case_key][key], label=label, color=color,
                    ls=ls, lw=1.5, alpha=0.85)
        ax.set_ylabel('Expected Length'); ax.set_title(title); ax.legend(fontsize=8)
    axes[-1].set_xlabel('Interval')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'four_way_queues.png'), dpi=300); plt.close()

    # Plot 3: 2×2 gap heatmap
    fig, ax = plt.subplots(figsize=(7, 5))
    mat = np.array([
        [cases['transient_nodelay']['objective'], cases['transient_delay']['objective']],
        [cases['steady_nodelay']['objective'], cases['steady_delay']['objective']],
    ])
    im = ax.imshow(mat, cmap='YlOrRd', aspect='auto')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['No Delay', 'With Delay'])
    ax.set_yticks([0, 1]); ax.set_yticklabels(['Transient', 'Steady-State'])
    ax.set_title('Objective: Model × Delay')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{mat[i,j]:.0f}', ha='center', va='center', fontsize=12)
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'four_way_heatmap.png'), dpi=300); plt.close()

    print(f"  Plots saved to {out_dir}/")
    return cases


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Analysis')
    parser.add_argument('--analysis', type=str, default='all',
                        choices=['all', 'numerical', 'steady_vs_transient', 'four_way'])
    parser.add_argument('--control_dir', type=str, default=None,
                        help='Directory with mu_add.npy and mu_remove.npy '
                             '(default: zero controls)')
    parser.add_argument('--n_intervals', type=int, default=None)
    parser.add_argument('--out_dir', type=str, default='results/model_analysis')
    args = parser.parse_args()

    config = QueueConfig()
    lambdas, mus_init = load_default_data(config)
    if args.n_intervals:
        lambdas = lambdas[:args.n_intervals]
        mus_init = mus_init[:args.n_intervals]
    alpha1, alpha2 = config.get_alpha_arrays(size=len(lambdas))

    # Load controls (or use zeros)
    n = len(lambdas)
    if args.control_dir and os.path.exists(args.control_dir):
        mus_add = np.load(os.path.join(args.control_dir, 'mu_add.npy'))[:n]
        mus_removed = np.load(os.path.join(args.control_dir, 'mu_remove.npy'))[:n]
        print(f"  Loaded controls from {args.control_dir}")
    else:
        mus_add = np.zeros(n)
        mus_removed = np.zeros(n)
        print(f"  Using zero controls (do-nothing baseline)")

    os.makedirs(args.out_dir, exist_ok=True)

    print("="*70)
    print("MODEL ANALYSIS")
    print("="*70)
    print(f"  Intervals: {n}")
    print(f"  States:    {(config.K_S+1)*(config.K_P+config.M+1)}")
    print(f"  Delays:    d_d={config.delay_non_reserved}, d_e={config.delay_extra}")
    print(f"  Controls:  sum(mu+)={mus_add.sum():.4f}, sum(mu-)={mus_removed.sum():.4f}")
    print(f"  Analysis:  {args.analysis}")
    print("="*70)

    t0 = time.time()

    if args.analysis in ('all', 'numerical'):
        analysis_numerical(lambdas, mus_init, alpha1, alpha2, config,
                          mus_add, mus_removed, args.out_dir)

    if args.analysis in ('all', 'steady_vs_transient'):
        analysis_steady_vs_transient(lambdas, mus_init, alpha1, alpha2, config,
                                    mus_add, mus_removed, args.out_dir)

    if args.analysis in ('all', 'four_way'):
        analysis_four_way(lambdas, mus_init, alpha1, alpha2, config,
                         mus_add, mus_removed, args.out_dir)

    print(f"\nTotal: {time.time()-t0:.1f}s. Results in {args.out_dir}/")
