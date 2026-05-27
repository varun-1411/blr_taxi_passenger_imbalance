"""
Optimizer and Model Comparison.

Experiment 1: Optimizer comparison (zero delays)
  - Do-nothing baseline
  - Brent (steady-state optimal)
  - Adam, AIMD, RS, BO (transient optimal)
  All evaluated on the transient model → shows:
    (a) Which optimizer is best
    (b) The SS-vs-transient gap (Brent vs Adam)

Experiment 2: Effect of delays
  - Take Adam's zero-delay solution → evaluate WITH delays
  - Run Adam WITH delays → new optimal
  - Shows the cost of delays

All raw data saved for plot regeneration.

Usage:
    python experiments/run_optimizer_comparison.py --experiment all
    python experiments/run_optimizer_comparison.py --experiment optimizers --methods brent adam
    python experiments/run_optimizer_comparison.py --experiment delays
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse, json, time, copy
import numpy as np
import torch
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

METHOD_COLORS = {
    'do_nothing': GRAY,
    'brent':      AMBER,
    'adam':        TEAL,
    'aimd':       ROSE,
    'rs':         PLUM,
    'bo':         BLUE,
    'adam_delay':  TEAL,
    'brent_on_transient': AMBER,
}

METHOD_LABELS = {
    'do_nothing': 'Do-Nothing',
    'brent':      'Brent (SS)',
    'adam':        'Adam (Transient)',
    'aimd':       'AIMD',
    'rs':         'Random Search',
    'bo':         'Bayesian Opt',
    'adam_delay':  'Adam (Transient + Delay)',
    'brent_on_transient': 'Brent SS on Transient',
}

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
    """Config copy with zero delays."""
    return QueueConfig(
        K_S=config.K_S, K_P=config.K_P, M=config.M,
        tau=config.tau, time_horizon=config.time_horizon,
        interval_length=config.interval_length,
        group_size=config.group_size,
        delay_non_reserved=0.0, delay_extra=0.0,
    )


def eval_transient(mu_add, mu_remove, lambdas, mus_init, alpha1, alpha2,
                   config, solver='uniformization'):
    """Evaluate controls on transient model. Returns full result dict."""
    return run_simulation(
        lambdas=lambdas, mu_0=mus_init,
        alpha1=alpha1, alpha2=alpha2,
        mus_add=mu_add, mus_removed=mu_remove,
        config=config, solver=solver, verbose=False)


def eval_steady(mu_add, mu_remove, lambdas, mus_init, alpha1, alpha2, config):
    """Evaluate controls on steady-state model."""
    return run_steady_state_evaluation(
        lambdas=lambdas, mu_0=mus_init,
        alpha1=alpha1, alpha2=alpha2,
        mus_add=mu_add, mus_removed=mu_remove,
        config=config, verbose=False)


def save_all(data, out_dir, prefix):
    """Save everything: controls, eval results, time series."""
    os.makedirs(out_dir, exist_ok=True)
    for key, val in data.items():
        if isinstance(val, np.ndarray):
            np.save(os.path.join(out_dir, f'{prefix}_{key}.npy'), val)
        elif isinstance(val, list) and len(val) > 0 and isinstance(val[0], (int, float)):
            np.save(os.path.join(out_dir, f'{prefix}_{key}.npy'), np.array(val))


def save_json(data, path):
    """Save JSON-serializable data."""
    def conv(o):
        if isinstance(o, (np.floating, np.integer)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, dict): return {str(k): conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [conv(v) for v in o]
        return o
    with open(path, 'w') as f:
        json.dump(conv(data), f, indent=2)


def print_row(name, r, baseline_obj=None):
    """Print one result row."""
    obj = r['objective']
    impr = (baseline_obj - obj) / baseline_obj * 100 if baseline_obj else 0
    t = r.get('time', 0)
    print(f"  {name:<25} {obj:>12.2f} "
          f"{r['total_passenger_wait']:>10.2f} "
          f"{r['total_taxi_idle_time']:>10.2f} "
          f"{r.get('total_additional_cost', 0):>10.2f} "
          f"{t:>7.1f}s {impr:>+7.2f}%")


# ══════════════════════════════════════════════════════════════
# OPTIMIZER RUNNERS (each returns dict with mu_add, mu_remove, etc.)
# ══════════════════════════════════════════════════════════════

def run_brent(lambdas, mus_init, alpha1, alpha2, config, out_dir):
    from optimizers.brent_optimizer import run_brent_steady_state
    print(f"    Brent (steady-state)...", end=' ', flush=True)
    t0 = time.time()
    r = run_brent_steady_state(
        lambdas, mus_init, alpha1, alpha2, config,
        delay_method='none', out_dir=out_dir, verbose=False)
    r['time'] = time.time() - t0
    print(f"obj_ss={r['objective']:.2f} ({r['time']:.1f}s)")
    return r


def run_adam(lambdas, mus_init, alpha1, alpha2, config, out_dir,
             init_add=None, init_rem=None, max_iter=None, max_time=None):
    from optimizers.adam_optimizer import run_adam_transient
    max_iter = max_iter or config.adam_max_iter
    print(f"    Adam (transient, max_iter={max_iter})...", end=' ', flush=True)
    t0 = time.time()
    r = run_adam_transient(
        lambdas, mus_init, alpha1, alpha2, config,
        init_mu_add=init_add, init_mu_remove=init_rem,
        max_iterations=max_iter, epsilon=config.adam_epsilon,
        lr=config.adam_lr, max_time=max_time, out_dir=out_dir)
    r['time'] = time.time() - t0
    print(f"obj={r['objective']:.2f} ({r['time']:.1f}s)")
    return r


def run_aimd(lambdas, mus_init, alpha1, alpha2, config, out_dir,
             max_iter=None, max_time=None):
    from optimizers.aimd_optimizer import run_aimd
    max_iter = max_iter or config.aimd_max_iter
    print(f"    AIMD (max_iter={max_iter})...", end=' ', flush=True)
    t0 = time.time()
    r = run_aimd(
        lambdas, mus_init, alpha1, alpha2, config,
        inc=config.aimd_inc, dec=config.aimd_dec,
        max_iters=max_iter, tol=config.aimd_tol,
        max_time=max_time, N_JOBS=-1, out_dir=out_dir)
    r['time'] = time.time() - t0
    print(f"obj={r['objective']:.2f} ({r['time']:.1f}s)")
    return r


def run_rs(lambdas, mus_init, alpha1, alpha2, config, out_dir,
           max_iter=None, max_time=None):
    from optimizers.random_search import run_random_search
    n_samples = max_iter or config.rs_n_samples
    print(f"    Random Search (n={n_samples})...", end=' ', flush=True)
    t0 = time.time()
    r = run_random_search(
        lambdas, mus_init, alpha1, alpha2, config,
        n_samples=n_samples, batch_size=config.rs_batch_size,
        n_refine=config.rs_n_refine,
        max_time=max_time, seed=config.optimizer_seed,
        N_JOBS=-1, out_dir=out_dir)
    r['time'] = time.time() - t0
    print(f"obj={r['objective']:.2f} ({r['time']:.1f}s)")
    return r


def run_bo(lambdas, mus_init, alpha1, alpha2, config, out_dir,
           init_add=None, init_rem=None, max_iter=None, max_time=None):
    from optimizers.bayesian_optimization import run_bayesopt
    num_iter = max_iter or config.bo_num_iter
    print(f"    Bayesian Opt (n={num_iter})...", end=' ', flush=True)
    lt = torch.tensor(lambdas, dtype=torch.float32)
    mt = torch.tensor(mus_init, dtype=torch.float32)
    a1t = torch.tensor(alpha1, dtype=torch.float32)
    a2t = torch.tensor(alpha2, dtype=torch.float32)
    ia = torch.tensor(init_add, dtype=torch.float32) if init_add is not None else None
    ir = torch.tensor(init_rem, dtype=torch.float32) if init_rem is not None else None
    t0 = time.time()
    r = run_bayesopt(lt, mt, a1t, a2t, config,
        init_mu_add=ia, init_mu_remove=ir,
        NUM_ITER=num_iter, max_time=max_time,
        SEED=config.optimizer_seed, out_dir=out_dir)
    r['time'] = time.time() - t0
    print(f"obj={r['objective']:.2f} ({r['time']:.1f}s)")
    return r


RUNNERS = {'brent': run_brent, 'adam': run_adam, 'aimd': run_aimd,
           'rs': run_rs, 'bo': run_bo}


# ══════════════════════════════════════════════════════════════
# EXPERIMENT 1: OPTIMIZER COMPARISON (ZERO DELAY)
# ══════════════════════════════════════════════════════════════

def experiment_optimizers(lambdas, mus_init, alpha1, alpha2, config,
                          methods, max_iter=None, max_time=None,
                          out_dir='results/optimizer_comparison'):
    """
    Compare optimizers with zero delays.
    Brent = SS optimal. Others = transient optimal.
    All evaluated on transient model → SS-vs-transient gap.
    """
    os.makedirs(out_dir, exist_ok=True)
    cfg_nd = make_nodelay_config(config)
    n = len(lambdas)

    print(f"\n{'='*70}")
    print(f"EXPERIMENT 1: OPTIMIZER COMPARISON (zero delays)")
    print(f"{'='*70}")

    # ── Run optimizers ──
    print(f"\n  Running optimizers:")
    opt_results = {}

    for method in methods:
        method_dir = os.path.join(out_dir, method)
        os.makedirs(method_dir, exist_ok=True)

        # Warm-start from Brent if available
        kw = {'max_iter': max_iter, 'max_time': max_time}
        if method in ('adam', 'bo') and 'brent' in opt_results:
            kw['init_add'] = opt_results['brent']['mu_add']
            kw['init_rem'] = opt_results['brent']['mu_remove']

        try:
            opt_results[method] = RUNNERS[method](
                lambdas, mus_init, alpha1, alpha2, cfg_nd, method_dir, **kw)
        except Exception as e:
            print(f"    {method} FAILED: {e}")

    # ── Evaluate ALL on transient model (zero delay) ──
    print(f"\n  Evaluating all on transient model (zero delay):")
    eval_results = {}

    # Do-nothing
    print(f"    Do-nothing...", end=' ', flush=True)
    dn = eval_transient(np.zeros(n), np.zeros(n), lambdas, mus_init,
                        alpha1, alpha2, cfg_nd)
    dn['time'] = 0; dn['mu_add'] = np.zeros(n); dn['mu_remove'] = np.zeros(n)
    eval_results['do_nothing'] = dn
    print(f"obj={dn['objective']:.2f}")

    for method, r in opt_results.items():
        label = f"{method} (transient eval)"
        print(f"    {label}...", end=' ', flush=True)
        er = eval_transient(r['mu_add'], r['mu_remove'], lambdas, mus_init,
                           alpha1, alpha2, cfg_nd)
        er['time'] = r.get('time', 0)
        er['mu_add'] = r['mu_add']
        er['mu_remove'] = r['mu_remove']
        eval_results[method] = er
        print(f"obj={er['objective']:.2f}")

    # Also evaluate Brent on steady-state model for comparison
    if 'brent' in opt_results:
        print(f"    Brent (SS eval)...", end=' ', flush=True)
        ss_eval = eval_steady(opt_results['brent']['mu_add'],
                              opt_results['brent']['mu_remove'],
                              lambdas, mus_init, alpha1, alpha2, cfg_nd)
        ss_eval['time'] = opt_results['brent'].get('time', 0)
        print(f"obj_ss={ss_eval['objective']:.2f}")

    # ── Summary table ──
    print(f"\n{'─'*80}")
    print(f"  {'Method':<25} {'Transient Obj':>12} {'Pax Wait':>10} "
          f"{'Taxi Idle':>10} {'Add Cost':>10} {'Time':>8} {'Impr':>8}")
    print(f"  {'─'*80}")
    dn_obj = eval_results['do_nothing']['objective']
    for name in ['do_nothing'] + methods:
        if name in eval_results:
            print_row(METHOD_LABELS.get(name, name), eval_results[name], dn_obj)

    # SS vs transient gap
    if 'brent' in eval_results and 'adam' in eval_results:
        brent_tr = eval_results['brent']['objective']
        adam_tr = eval_results['adam']['objective']
        gap = brent_tr - adam_tr
        gap_pct = gap / adam_tr * 100
        print(f"\n  SS-vs-Transient gap (Brent - Adam on transient):")
        print(f"    Absolute: {gap:+.2f}")
        print(f"    Relative: {gap_pct:+.3f}%")
        if 'brent' in opt_results:
            print(f"    Brent SS objective:        {opt_results['brent']['objective']:.2f}")
            print(f"    Brent on transient:        {brent_tr:.2f}")
            print(f"    Adam on transient:         {adam_tr:.2f}")

    # ── Save everything ──
    for name, r in eval_results.items():
        save_all(r, os.path.join(out_dir, 'eval'), name)
        np.save(os.path.join(out_dir, 'eval', f'{name}_mu_add.npy'),
                r.get('mu_add', np.zeros(n)))
        np.save(os.path.join(out_dir, 'eval', f'{name}_mu_remove.npy'),
                r.get('mu_remove', np.zeros(n)))

    # Save convergence histories
    for name, r in opt_results.items():
        for hist_key in ['history', 'objective_history']:
            if hist_key in r:
                np.save(os.path.join(out_dir, name, f'{hist_key}.npy'),
                        np.array(r[hist_key]))

    summary = {name: {
        'objective': float(r['objective']),
        'pax_wait': float(r['total_passenger_wait']),
        'taxi_idle': float(r['total_taxi_idle_time']),
        'add_cost': float(r.get('total_additional_cost', 0)),
        'remove_cost': float(r.get('total_removal_cost', 0)),
        'time': float(r.get('time', 0)),
    } for name, r in eval_results.items()}
    save_json(summary, os.path.join(out_dir, 'summary.json'))

    # ── Plots ──
    _plot_optimizers(eval_results, opt_results, lambdas, config, out_dir)

    return eval_results, opt_results


# ══════════════════════════════════════════════════════════════
# EXPERIMENT 2: EFFECT OF DELAYS
# ══════════════════════════════════════════════════════════════

def experiment_delays(lambdas, mus_init, alpha1, alpha2, config,
                      max_iter=None, max_time=None,
                      out_dir='results/delay_comparison'):
    """
    Compare zero-delay vs with-delay:
      1. Adam zero-delay → evaluate with delay (suboptimal)
      2. Adam with-delay → true optimum
      3. Gap = cost of ignoring delays
    """
    os.makedirs(out_dir, exist_ok=True)
    cfg_nd = make_nodelay_config(config)
    n = len(lambdas)

    print(f"\n{'='*70}")
    print(f"EXPERIMENT 2: EFFECT OF DELAYS")
    print(f"  d_d={config.delay_non_reserved}, d_e_extra={config.delay_extra}")
    print(f"{'='*70}")

    # Do-nothing (with delays)
    print(f"\n  Do-nothing (with delays)...", end=' ', flush=True)
    dn_delay = eval_transient(np.zeros(n), np.zeros(n), lambdas, mus_init,
                              alpha1, alpha2, config)
    print(f"obj={dn_delay['objective']:.2f}")

    # Do-nothing (no delays)
    print(f"  Do-nothing (no delays)...", end=' ', flush=True)
    dn_nodelay = eval_transient(np.zeros(n), np.zeros(n), lambdas, mus_init,
                                alpha1, alpha2, cfg_nd)
    print(f"obj={dn_nodelay['objective']:.2f}")

    # Adam zero-delay optimal
    print(f"\n  Adam (zero delay):")
    adam_nd_dir = os.path.join(out_dir, 'adam_nodelay')
    os.makedirs(adam_nd_dir, exist_ok=True)
    adam_nd = run_adam(lambdas, mus_init, alpha1, alpha2, cfg_nd, adam_nd_dir,
                      max_iter=max_iter, max_time=max_time)

    # Evaluate zero-delay solution ON delayed model
    print(f"    → Evaluating on delayed model...", end=' ', flush=True)
    adam_nd_on_delay = eval_transient(adam_nd['mu_add'], adam_nd['mu_remove'],
                                     lambdas, mus_init, alpha1, alpha2, config)
    print(f"obj={adam_nd_on_delay['objective']:.2f}")

    # Adam with-delay optimal
    print(f"\n  Adam (with delay):")
    adam_d_dir = os.path.join(out_dir, 'adam_delay')
    os.makedirs(adam_d_dir, exist_ok=True)
    adam_d = run_adam(lambdas, mus_init, alpha1, alpha2, config, adam_d_dir,
                     max_iter=max_iter, max_time=max_time)

    # Evaluate delay-optimal on delayed model
    print(f"    → Evaluating on delayed model...", end=' ', flush=True)
    adam_d_eval = eval_transient(adam_d['mu_add'], adam_d['mu_remove'],
                                lambdas, mus_init, alpha1, alpha2, config)
    print(f"obj={adam_d_eval['objective']:.2f}")

    # ── Summary ──
    print(f"\n{'─'*70}")
    print(f"  DELAY COMPARISON (all evaluated on transient WITH delays):")
    print(f"  {'Scenario':<35} {'Objective':>12} {'Δ from best':>12} {'%':>8}")
    print(f"  {'─'*70}")

    best = adam_d_eval['objective']
    rows = [
        ('Do-nothing (with delay)', dn_delay['objective']),
        ('Adam zero-delay on delay model', adam_nd_on_delay['objective']),
        ('Adam with-delay (true optimal)', adam_d_eval['objective']),
    ]
    for name, obj in rows:
        gap = obj - best
        pct = gap / best * 100 if best else 0
        print(f"  {name:<35} {obj:>12.2f} {gap:>+12.2f} {pct:>+7.2f}%")

    delay_cost = adam_nd_on_delay['objective'] - adam_d_eval['objective']
    delay_pct = delay_cost / adam_d_eval['objective'] * 100
    print(f"\n  Cost of ignoring delays: {delay_cost:+.2f} ({delay_pct:+.3f}%)")

    # ── Save ──
    results = {
        'dn_delay': dn_delay, 'dn_nodelay': dn_nodelay,
        'adam_nodelay': adam_nd, 'adam_nodelay_on_delay': adam_nd_on_delay,
        'adam_delay': adam_d, 'adam_delay_eval': adam_d_eval,
    }
    for name, r in results.items():
        save_all(r, os.path.join(out_dir, 'eval'), name)
        if 'mu_add' in r:
            np.save(os.path.join(out_dir, 'eval', f'{name}_mu_add.npy'), r['mu_add'])
            np.save(os.path.join(out_dir, 'eval', f'{name}_mu_remove.npy'), r['mu_remove'])

    save_json({
        'dn_delay': float(dn_delay['objective']),
        'dn_nodelay': float(dn_nodelay['objective']),
        'adam_nodelay_obj': float(adam_nd['objective']),
        'adam_nodelay_on_delay': float(adam_nd_on_delay['objective']),
        'adam_delay_obj': float(adam_d_eval['objective']),
        'delay_cost': float(delay_cost),
        'delay_cost_pct': float(delay_pct),
    }, os.path.join(out_dir, 'summary.json'))

    _plot_delays(dn_delay, dn_nodelay, adam_nd, adam_nd_on_delay,
                 adam_d, adam_d_eval, lambdas, config, out_dir)

    return results


# ══════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════

def _plot_optimizers(eval_results, opt_results, lambdas, config, out_dir):
    """Publication plots for optimizer comparison."""
    n = len(lambdas)
    t = np.arange(n) * config.interval_length
    names = list(eval_results.keys())

    # 1. Objective bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = [METHOD_LABELS.get(n, n) for n in names]
    objs = [eval_results[n]['objective'] for n in names]
    colors = [METHOD_COLORS.get(n, GRAY) for n in names]
    bars = ax.bar(range(len(names)), objs, color=colors, alpha=0.85)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_ylabel('Objective (Transient Evaluation)')
    ax.set_title('Optimizer Comparison — All on Transient Model (Zero Delay)')
    for bar, obj in zip(bars, objs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{obj:.0f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'objectives.png'), dpi=300); plt.close()

    # 2. Controls overlay
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    ax = axes[0]
    ax2 = ax.twinx()
    ax2.fill_between(t, lambdas, alpha=0.08, color=BLUE)
    ax2.set_ylabel(r'$\lambda$', color=BLUE)
    for name in names:
        if name == 'do_nothing':
            continue
        r = eval_results[name]
        c = METHOD_COLORS.get(name, GRAY)
        lbl = METHOD_LABELS.get(name, name)
        ax.plot(t, r.get('mu_add', np.zeros(n)), color=c, lw=1.5,
                label=lbl, alpha=0.85)
    ax.set_ylabel(r'$\mu^+$'); ax.set_title('Addition Rate'); ax.legend(fontsize=8)

    ax = axes[1]
    for name in names:
        if name == 'do_nothing':
            continue
        r = eval_results[name]
        c = METHOD_COLORS.get(name, GRAY)
        lbl = METHOD_LABELS.get(name, name)
        ax.plot(t, r.get('mu_remove', np.zeros(n)), color=c, lw=1.5,
                label=lbl, alpha=0.85)
    ax.set_xlabel('Time (min)'); ax.set_ylabel(r'$\mu^-$')
    ax.set_title('Removal Rate'); ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'controls.png'), dpi=300); plt.close()

    # 3. Queue lengths
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    keys_ts = ['pax_queue_ts', 'taxi_queue_ts', 'resv_queue_ts']
    titles = ['Passenger Queue', 'Taxi Queue (Pickup)', 'Staging Queue']
    for ax, title, key in zip(axes, titles, keys_ts):
        for name in names:
            r = eval_results[name]
            if key in r:
                c = METHOD_COLORS.get(name, GRAY)
                lbl = METHOD_LABELS.get(name, name)
                ls = '--' if name in ('do_nothing', 'brent') else '-'
                ax.plot(r[key], color=c, lw=1.5, label=lbl, ls=ls, alpha=0.85)
        ax.set_ylabel('Expected Length'); ax.set_title(title); ax.legend(fontsize=8)
    axes[-1].set_xlabel('Interval')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'queues.png'), dpi=300); plt.close()

    # 4. Timing
    opt_names = [n for n in names if n != 'do_nothing' and 'time' in eval_results[n]]
    if opt_names:
        fig, ax = plt.subplots(figsize=(8, 5))
        times = [eval_results[n].get('time', 0) for n in opt_names]
        labels_t = [METHOD_LABELS.get(n, n) for n in opt_names]
        colors_t = [METHOD_COLORS.get(n, GRAY) for n in opt_names]
        bars = ax.bar(range(len(opt_names)), times, color=colors_t, alpha=0.85)
        ax.set_xticks(range(len(opt_names)))
        ax.set_xticklabels(labels_t, rotation=20, ha='right')
        ax.set_ylabel('Time (s)'); ax.set_title('Optimizer Runtime')
        for bar, ti in zip(bars, times):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                    f'{ti:.1f}s', ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'timing.png'), dpi=300); plt.close()

    # 5. Convergence
    fig, ax = plt.subplots(figsize=(10, 6))
    has_hist = False
    for name, r in opt_results.items():
        hist = r.get('history', r.get('objective_history', None))
        if hist is not None and len(hist) > 1:
            c = METHOD_COLORS.get(name, GRAY)
            ax.plot(hist, color=c, lw=1.5, label=METHOD_LABELS.get(name, name))
            has_hist = True
    if has_hist:
        ax.set_xlabel('Iteration'); ax.set_ylabel('Objective')
        ax.set_title('Convergence'); ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'convergence.png'), dpi=300)
    plt.close()

    print(f"  Plots saved to {out_dir}/")


def _plot_delays(dn_d, dn_nd, adam_nd, adam_nd_on_d, adam_d, adam_d_eval,
                 lambdas, config, out_dir):
    """Plots for delay experiment."""
    n = len(lambdas)
    t = np.arange(n) * config.interval_length

    # 1. Objective comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    names = ['Do-Nothing\n(no delay)', 'Do-Nothing\n(with delay)',
             'Adam zero-delay\n(eval on delay)', 'Adam\n(with delay)']
    objs = [dn_nd['objective'], dn_d['objective'],
            adam_nd_on_d['objective'], adam_d_eval['objective']]
    colors = [GRAY, GRAY, AMBER, TEAL]
    alphas = [0.5, 0.8, 0.8, 0.85]
    bars = ax.bar(range(len(names)), objs, color=colors, alpha=alphas)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel('Objective'); ax.set_title('Effect of Delays')
    for bar, obj in zip(bars, objs):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                f'{obj:.0f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'delay_objectives.png'), dpi=300); plt.close()

    # 2. Control profiles: zero-delay vs with-delay optimal
    fig, ax = plt.subplots(figsize=(14, 5))
    ax2 = ax.twinx()
    ax2.fill_between(t, lambdas, alpha=0.08, color=BLUE)
    ax2.set_ylabel(r'$\lambda$', color=BLUE)
    ax.plot(t, adam_nd['mu_add'], color=AMBER, lw=1.5,
            label='Adam (zero-delay optimal)', ls='--')
    ax.plot(t, adam_d['mu_add'], color=TEAL, lw=1.8,
            label='Adam (with-delay optimal)')
    ax.set_xlabel('Time (min)'); ax.set_ylabel(r'$\mu^+$')
    ax.set_title('Control Profiles: Zero-Delay vs With-Delay Optimal')
    ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'delay_controls.png'), dpi=300); plt.close()

    print(f"  Plots saved to {out_dir}/")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimizer & Model Comparison')
    parser.add_argument('--experiment', type=str, default='all',
                        choices=['all', 'optimizers', 'delays'])
    parser.add_argument('--methods', nargs='+',
                        default=['brent', 'adam', 'aimd', 'rs'],
                        choices=['brent', 'adam', 'aimd', 'rs', 'bo', 'all'])
    parser.add_argument('--max_iter', type=int, default=None)
    parser.add_argument('--max_time', type=float, default=None)
    parser.add_argument('--n_intervals', type=int, default=None)
    parser.add_argument('--out_dir', type=str, default='results/optimizer_comparison')
    args = parser.parse_args()

    if 'all' in args.methods:
        args.methods = ['brent', 'adam', 'aimd', 'rs', 'bo']

    config = QueueConfig()
    lambdas, mus_init = load_default_data(config)
    if args.n_intervals:
        lambdas = lambdas[:args.n_intervals]
        mus_init = mus_init[:args.n_intervals]
    alpha1, alpha2 = config.get_alpha_arrays(size=len(lambdas))

    print("=" * 70)
    print("OPTIMIZER & MODEL COMPARISON")
    print("=" * 70)
    print(f"  Intervals: {len(lambdas)}")
    print(f"  States:    {(config.K_S+1)*(config.K_P+config.M+1)}")
    print(f"  Delays:    d_d={config.delay_non_reserved}, d_e={config.delay_extra}")
    print(f"  Methods:   {args.methods}")
    print(f"  Experiment:{args.experiment}")
    print("=" * 70)

    t0 = time.time()

    if args.experiment in ('all', 'optimizers'):
        experiment_optimizers(
            lambdas, mus_init, alpha1, alpha2, config,
            methods=args.methods,
            max_iter=args.max_iter, max_time=args.max_time,
            out_dir=os.path.join(args.out_dir, 'zero_delay'))

    if args.experiment in ('all', 'delays'):
        experiment_delays(
            lambdas, mus_init, alpha1, alpha2, config,
            max_iter=args.max_iter, max_time=args.max_time,
            out_dir=os.path.join(args.out_dir, 'delay_effect'))

    print(f"\nTotal: {time.time()-t0:.1f}s")
    print(f"Results in {args.out_dir}/")
