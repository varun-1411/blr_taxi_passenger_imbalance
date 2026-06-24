"""
Optimizer Comparison (with delays).

Compare transient optimizers — Adam, AIMD, RS, BO — on the delayed model:
  - Do-nothing baseline
  - Each optimizer cold-started, trained and evaluated WITH delays
  - Shows which optimizer is best under the real (delayed) operating model

All raw data saved for plot regeneration.

Usage:
    python experiments/run_optimizer_comparison.py
    python experiments/run_optimizer_comparison.py --methods adam aimd
    python experiments/run_optimizer_comparison.py --max_time 600
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse, json, time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import QueueConfig
from data import load_default_data
from model.metrics import run_simulation

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
    'adam':       TEAL,
    'aimd':       ROSE,
    'rs':         PLUM,
    'bo':         BLUE,
}

METHOD_LABELS = {
    'do_nothing': 'Do-Nothing',
    'adam':       'Adam (Transient)',
    'aimd':       'AIMD',
    'rs':         'Random Search',
    'bo':         'Bayesian Opt',
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

def eval_transient(mu_add, mu_remove, lambdas, mus_init, alpha1, alpha2,
                   config, solver='uniformization'):
    """Evaluate controls on transient model. Returns full result dict."""
    return run_simulation(
        lambdas=lambdas, mu_0=mus_init,
        alpha1=alpha1, alpha2=alpha2,
        mus_add=mu_add, mus_removed=mu_remove,
        config=config, solver=solver, verbose=False)


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


RUNNERS = {'adam': run_adam, 'aimd': run_aimd, 'rs': run_rs, 'bo': run_bo}


# ══════════════════════════════════════════════════════════════
# OPTIMIZER COMPARISON (WITH DELAYS)
# ══════════════════════════════════════════════════════════════

def experiment_optimizers(lambdas, mus_init, alpha1, alpha2, config,
                          methods, max_iter=None, max_time=None,
                          out_dir='results/optimizer_comparison'):
    """
    Compare transient optimizers (Adam, AIMD, RS, BO) WITH delays.
    All cold-started, trained and evaluated on the delayed transient model.
    """
    os.makedirs(out_dir, exist_ok=True)
    n = len(lambdas)

    print(f"\n{'='*70}")
    print(f"OPTIMIZER COMPARISON (with delays)")
    print(f"{'='*70}")

    # -- Run optimizers (cold start, with delays) --
    print(f"\n  Running optimizers:")
    opt_results = {}

    for method in methods:
        method_dir = os.path.join(out_dir, method)
        os.makedirs(method_dir, exist_ok=True)

        kw = {'max_iter': max_iter, 'max_time': max_time}
        try:
            opt_results[method] = RUNNERS[method](
                lambdas, mus_init, alpha1, alpha2, config, method_dir, **kw)
        except Exception as e:
            print(f"    {method} FAILED: {e}")

    # -- Evaluate ALL on transient model (with delays) --
    print(f"\n  Evaluating all on transient model (with delays):")
    eval_results = {}

    # Do-nothing
    print(f"    Do-nothing...", end=' ', flush=True)
    dn = eval_transient(np.zeros(n), np.zeros(n), lambdas, mus_init,
                        alpha1, alpha2, config)
    dn['time'] = 0; dn['mu_add'] = np.zeros(n); dn['mu_remove'] = np.zeros(n)
    eval_results['do_nothing'] = dn
    print(f"obj={dn['objective']:.2f}")

    for method, r in opt_results.items():
        label = f"{method} (transient eval)"
        print(f"    {label}...", end=' ', flush=True)
        er = eval_transient(r['mu_add'], r['mu_remove'], lambdas, mus_init,
                           alpha1, alpha2, config)
        er['time'] = r.get('time', 0)
        er['mu_add'] = r['mu_add']
        er['mu_remove'] = r['mu_remove']
        eval_results[method] = er
        print(f"obj={er['objective']:.2f}")

    # -- Summary table --
    print(f"\n{'-'*80}")
    print(f"  {'Method':<25} {'Transient Obj':>12} {'Pax Wait':>10} "
          f"{'Taxi Idle':>10} {'Add Cost':>10} {'Time':>8} {'Impr':>8}")
    print(f"  {'-'*80}")
    dn_obj = eval_results['do_nothing']['objective']
    for name in ['do_nothing'] + methods:
        if name in eval_results:
            print_row(METHOD_LABELS.get(name, name), eval_results[name], dn_obj)

    # Best optimizer
    opt_only = {k: v for k, v in eval_results.items() if k != 'do_nothing'}
    if opt_only:
        best_name = min(opt_only, key=lambda k: opt_only[k]['objective'])
        print(f"\n  Best optimizer: {METHOD_LABELS.get(best_name, best_name)} "
              f"(obj={opt_only[best_name]['objective']:.2f})")

    # -- Save everything --
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

    # -- Plots --
    _plot_optimizers(eval_results, opt_results, lambdas, config, out_dir)

    return eval_results, opt_results


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
    labels = [METHOD_LABELS.get(nm, nm) for nm in names]
    objs = [eval_results[nm]['objective'] for nm in names]
    colors = [METHOD_COLORS.get(nm, GRAY) for nm in names]
    bars = ax.bar(range(len(names)), objs, color=colors, alpha=0.85)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_ylabel('Objective (Transient Evaluation)')
    ax.set_title('Optimizer Comparison - All on Transient Model (With Delays)')
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
                ls = '--' if name == 'do_nothing' else '-'
                ax.plot(r[key], color=c, lw=1.5, label=lbl, ls=ls, alpha=0.85)
        ax.set_ylabel('Expected Length'); ax.set_title(title); ax.legend(fontsize=8)
    axes[-1].set_xlabel('Interval')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'queues.png'), dpi=300); plt.close()

    # 4. Timing
    opt_names = [nm for nm in names if nm != 'do_nothing' and 'time' in eval_results[nm]]
    if opt_names:
        fig, ax = plt.subplots(figsize=(8, 5))
        times = [eval_results[nm].get('time', 0) for nm in opt_names]
        labels_t = [METHOD_LABELS.get(nm, nm) for nm in opt_names]
        colors_t = [METHOD_COLORS.get(nm, GRAY) for nm in opt_names]
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


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimizer Comparison (with delays)')
    parser.add_argument('--methods', nargs='+',
                        default=['bo', 'aimd', 'rs', 'adam'],
                        choices=['adam', 'aimd', 'rs', 'bo', 'all'])
    parser.add_argument('--max_iter', type=int, default=None)
    parser.add_argument('--max_time', type=float, default=None,
                        help='Per-optimizer wall-clock time limit (seconds)')
    parser.add_argument('--n_intervals', type=int, default=None)
    parser.add_argument('--out_dir', type=str, default='results/optimizer_comparison')
    args = parser.parse_args()

    if 'all' in args.methods:
        args.methods = ['bo', 'aimd', 'rs', 'adam']

    config = QueueConfig()
    lambdas, mus_init = load_default_data(config)
    if args.n_intervals:
        lambdas = lambdas[:args.n_intervals]
        mus_init = mus_init[:args.n_intervals]
    alpha1, alpha2 = config.get_alpha_arrays(size=len(lambdas))

    print("=" * 70)
    print("OPTIMIZER COMPARISON (WITH DELAYS)")
    print("=" * 70)
    print(f"  Intervals: {len(lambdas)}")
    print(f"  States:    {(config.K_S+1)*(config.K_P+config.M+1)}")
    print(f"  Delays:    d_d={config.delay_non_reserved}, d_e={config.delay_ext_minutes}")
    print(f"  Methods:   {args.methods}")
    print(f"  Max time:  {args.max_time if args.max_time else 'none'} s/optimizer")
    print("=" * 70)

    t0 = time.time()

    experiment_optimizers(
        lambdas, mus_init, alpha1, alpha2, config,
        methods=args.methods,
        max_iter=args.max_iter, max_time=args.max_time,
        out_dir=os.path.join(args.out_dir, 'with_delay'))

    print(f"\nTotal: {time.time()-t0:.1f}s")
    print(f"Results in {args.out_dir}/")