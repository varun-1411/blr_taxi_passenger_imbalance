"""
plot_results.py — read per-method/per-seed results from disk, print statistics
(objectives AND wall-clock time averaged across seeds/machines), and plot.

Greedy/MPC seeds are globbed, so you can accumulate seeds across separate runs and
machines into one --out_dir and this summarizes over the union. Full-day/do-nothing
are read from the new layout, or fall back to the legacy fd_*.npy + summary.json.

Usage:
    python experiments/plot_results.py --out_dir results/comparison
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse, json, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import QueueConfig
from data import load_default_data
from optimizer_utils import evaluate_per_block

COLORS = {'full_day': '#2E86AB', 'greedy': '#E8475F', 'mpc': '#F5A623'}


# ──────────────────────────────────────────────────────────────
# LOADING
# ──────────────────────────────────────────────────────────────

def load_run_config(out_dir):
    path = os.path.join(out_dir, 'run_config.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _load_run(d):
    with open(os.path.join(d, 'meta.json')) as f:
        meta = json.load(f)
    return {
        'objective': float(meta['objective']),
        'time': float(meta.get('time_seconds', 0.0)),
        'mu_add': np.load(os.path.join(d, 'mu_add.npy')),
        'mu_remove': np.load(os.path.join(d, 'mu_remove.npy')),
        'sampled_states': meta.get('sampled_states', []),
        'hostname': meta.get('hostname', '?'),
        'seed': meta.get('seed'),
    }


def load_single(out_dir, method, n):
    """full_day / do_nothing: new layout, else legacy fallback."""
    d = os.path.join(out_dir, method)
    if os.path.exists(os.path.join(d, 'meta.json')):
        return _load_run(d)

    sj = os.path.join(out_dir, 'summary.json')
    if method == 'full_day':
        fa = os.path.join(out_dir, 'fd_mu_add.npy')
        fr = os.path.join(out_dir, 'fd_mu_remove.npy')
        if os.path.exists(fa) and os.path.exists(sj):
            with open(sj) as f: s = json.load(f)
            print("  (full_day: using legacy fd_*.npy + summary.json)")
            return {'objective': float(s['full_day']), 'time': 0.0,
                    'mu_add': np.load(fa), 'mu_remove': np.load(fr),
                    'sampled_states': [], 'hostname': 'legacy', 'seed': None}
    if method == 'do_nothing' and os.path.exists(sj):
        with open(sj) as f: s = json.load(f)
        print("  (do_nothing: using legacy summary.json)")
        return {'objective': float(s['do_nothing']), 'time': 0.0,
                'mu_add': np.zeros(n), 'mu_remove': np.zeros(n),
                'sampled_states': [], 'hostname': 'legacy', 'seed': None}
    return None


def load_seeds(out_dir, method):
    runs = []
    for d in sorted(glob.glob(os.path.join(out_dir, method, 'seed_*'))):
        if os.path.exists(os.path.join(d, 'meta.json')):
            runs.append(_load_run(d))
    return runs


# ──────────────────────────────────────────────────────────────
# STATISTICS  (moved here from the runner)
# ──────────────────────────────────────────────────────────────

def print_statistics(results):
    fd = results['full_day']; dn = results['do_nothing']
    gr = results['greedy']; mpc = results['mpc']
    co = results.get('carryover_cost', 0.0)

    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*70}")

    if fd is None:
        print("  [!] No full_day result found — it is the benchmark; cannot rank without it.")
    else:
        fd_obj = fd['objective']
        print(f"\n  BENCHMARK (zero-pad objective):")
        print(f"    Full-Day:   {fd_obj:>12.2f}")
        if dn is not None:
            dn_obj = dn['objective']
            print(f"    Do-Nothing: {dn_obj:>12.2f}")
            print(f"    Improvement: {(dn_obj-fd_obj)/dn_obj*100:.2f}%")

        if co > 0:
            print(f"\n  TRUE DAILY COST (zero-pad + carryover {co:.2f}):")
            print(f"    Full-Day:   {fd_obj + co:>12.2f}")
            if gr:  print(f"    Greedy mean:{np.mean([r['objective'] for r in gr]) + co:>12.2f}")
            if mpc: print(f"    MPC mean:   {np.mean([r['objective'] for r in mpc]) + co:>12.2f}")

        # Objective table
        print(f"\n  {'Method':<10} {'Runs':>5} {'Mean':>12} {'Std':>12} "
              f"{'Min':>12} {'Max':>12} {'Gap %':>10}")
        print(f"  {'─'*70}")
        for label, runs in [('Greedy', gr), ('MPC', mpc)]:
            if runs:
                o = np.array([r['objective'] for r in runs])
                print(f"  {label:<10} {len(o):>5} {o.mean():>12.2f} {o.std():>12.2f} "
                      f"{o.min():>12.2f} {o.max():>12.2f} "
                      f"{(o.mean()-fd_obj)/fd_obj*100:>+10.2f}%")

        # Per-run objectives
        if gr or mpc:
            print(f"\n  Per-run objectives (gap vs Full-Day):")
            print(f"  {'Seed':>6} {'Greedy':>12} {'GR-FD':>10}   {'MPC':>12} {'MPC-FD':>10}")
            print(f"  {'─'*58}")
            gr_by = {r['seed']: r for r in gr}
            mpc_by = {r['seed']: r for r in mpc}
            for seed in sorted(set(gr_by) | set(mpc_by), key=lambda x: (x is None, x)):
                g = gr_by.get(seed); m = mpc_by.get(seed)
                gs  = f"{g['objective']:>12.2f}" if g else f"{'—':>12}"
                gg  = f"{g['objective']-fd_obj:>+10.2f}" if g else f"{'—':>10}"
                ms  = f"{m['objective']:>12.2f}" if m else f"{'—':>12}"
                mg  = f"{m['objective']-fd_obj:>+10.2f}" if m else f"{'—':>10}"
                print(f"  {str(seed):>6} {gs} {gg}   {ms} {mg}")

    # ── WALL-CLOCK TIME: averaged across seeds / machines ──
    if gr or mpc:
        print(f"\n  WALL-CLOCK TIME (averaged across seeds / machines):")
        print(f"  {'Method':<10} {'Runs':>5} {'Mean(s)':>10} {'Std':>10} "
              f"{'Min':>10} {'Max':>10}")
        print(f"  {'─'*56}")
        for label, runs in [('Greedy', gr), ('MPC', mpc)]:
            if runs:
                t = np.array([r['time'] for r in runs])
                print(f"  {label:<10} {len(t):>5} {t.mean():>10.1f} {t.std():>10.1f} "
                      f"{t.min():>10.1f} {t.max():>10.1f}")
        for label, runs in [('Greedy', gr), ('MPC', mpc)]:
            if runs:
                print(f"\n  {label} per-seed time / host:")
                for r in runs:
                    print(f"    seed={r['seed']}: {r['time']:>8.1f}s   on {r['hostname']}")


# ──────────────────────────────────────────────────────────────
# PLOTTING  (tolerates missing greedy/mpc)
# ──────────────────────────────────────────────────────────────

def plot_all(results, lambdas, mus_init, alpha1, alpha2, config, commit_size, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    n = len(lambdas)
    t = np.arange(n) * config.interval_length

    fd = results['full_day']; dn = results['do_nothing']
    gr = results['greedy']; mpc = results['mpc']
    if fd is None:
        print("  [!] No full_day — skipping plots (need the benchmark).")
        return

    fd_obj = fd['objective']
    present = [(k, runs) for k, runs in [('greedy', gr), ('mpc', mpc)] if runs]

    # ── 1. Objectives boxplot (only methods present) ──
    if present:
        fig, ax = plt.subplots(figsize=(8, 5))
        data = [np.array([r['objective'] for r in runs]) for _, runs in present]
        labels = [k.upper() for k, _ in present]
        cols = [COLORS[k] for k, _ in present]
        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5)
        for patch, c in zip(bp['boxes'], cols):
            patch.set_facecolor(c); patch.set_alpha(0.6)
        for i, (d, c) in enumerate(zip(data, cols)):
            ax.scatter(np.random.normal(i+1, 0.04, len(d)), d,
                       color=c, alpha=0.7, s=30, zorder=3)
        ax.axhline(y=fd_obj, color=COLORS['full_day'], lw=2,
                   label=f'Full-Day ({fd_obj:.0f})')
        if dn is not None:
            ax.axhline(y=dn['objective'], color='gray', ls='--',
                       label=f'Do-Nothing ({dn["objective"]:.0f})')
        ax.set_ylabel('Objective'); ax.set_title('Objective Distribution')
        ax.legend(); ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'objectives.png'), dpi=150); plt.close()

    # ── 2. Control profiles ──
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    for ax_i, (ctrl_key, base, base_lbl, ylabel, title) in enumerate([
            ('mu_add', lambdas, 'λ', 'μ⁺', 'Taxi Addition Rate'),
            ('mu_remove', mus_init, 'μ^d', 'μ⁻', 'Taxi Removal Rate')]):
        ax = axes[ax_i]; ax2 = ax.twinx()
        ax2.fill_between(t, base, alpha=0.08, color='gray')
        ax2.set_ylabel(base_lbl, color='gray')
        ax.plot(t, fd[ctrl_key], color=COLORS['full_day'], lw=1.5, label='Full-Day')
        for k, runs in present:
            arr = np.array([r[ctrl_key] for r in runs])
            mean = arr.mean(axis=0); std = arr.std(axis=0)
            ax.plot(t, mean, color=COLORS[k], lw=1.5, label=k.upper())
            ax.fill_between(t, mean-std, mean+std, color=COLORS[k], alpha=0.15)
        for b in range(0, n, commit_size):
            ax.axvline(x=b*config.interval_length, color='gray', ls='--', alpha=0.2)
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(loc='upper left'); ax.grid(True, alpha=0.3)
    axes[1].set_xlabel('Time (min)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'controls.png'), dpi=150); plt.close()

    # ── 3. Per-block costs (best run of each present method vs full-day) ──
    fd_blocks, _, _ = evaluate_per_block(
        fd['mu_add'], fd['mu_remove'], lambdas, mus_init,
        alpha1, alpha2, config, commit_size)
    n_blocks = len(fd_blocks)
    x = np.arange(n_blocks)
    bl = [f'{b*commit_size*config.interval_length/60:.0f}-'
          f'{min((b+1)*commit_size,n)*config.interval_length/60:.0f}h'
          for b in range(n_blocks)]

    block_data = {'full_day': fd_blocks}
    for k, runs in present:
        best = runs[int(np.argmin([r['objective'] for r in runs]))]
        block_data[k], _, _ = evaluate_per_block(
            best['mu_add'], best['mu_remove'], lambdas, mus_init,
            alpha1, alpha2, config, commit_size)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    series = ['full_day'] + [k for k, _ in present]
    w = 0.8 / len(series)
    for i, k in enumerate(series):
        axes[0].bar(x + (i - (len(series)-1)/2)*w, block_data[k], w,
                    color=COLORS[k], alpha=0.8, label=k.replace('_', '-').title())
    axes[0].set_xticks(x); axes[0].set_xticklabels(bl, rotation=45, ha='right')
    axes[0].set_ylabel('Block Cost'); axes[0].set_title('Per-Block Cost')
    axes[0].legend(); axes[0].grid(True, alpha=0.3, axis='y')

    gap_series = [k for k, _ in present]
    if gap_series:
        wg = 0.8 / len(gap_series)
        for i, k in enumerate(gap_series):
            gap = np.array(block_data[k]) - np.array(fd_blocks)
            axes[1].bar(x + (i - (len(gap_series)-1)/2)*wg, gap, wg,
                        color=COLORS[k], alpha=0.7, label=f'{k.upper()} − FD')
    axes[1].axhline(y=0, color='black', lw=0.5)
    axes[1].set_xticks(x); axes[1].set_xticklabels(bl, rotation=45, ha='right')
    axes[1].set_ylabel('Δ Cost'); axes[1].set_title('Gap from Full-Day per Block')
    axes[1].legend(); axes[1].grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'per_block.png'), dpi=150); plt.close()

    print(f"  Plots saved to {out_dir}/")


def save_summary(results, out_dir):
    fd = results['full_day']; dn = results['do_nothing']
    gr = results['greedy']; mpc = results['mpc']
    co = results.get('carryover_cost', 0.0)

    def stats(runs, key):
        if not runs: return None
        v = np.array([r[key] for r in runs])
        return {'mean': float(v.mean()), 'std': float(v.std()),
                'min': float(v.min()), 'max': float(v.max()), 'n': int(len(v))}

    summary = {
        'full_day': fd['objective'] if fd else None,
        'do_nothing': dn['objective'] if dn else None,
        'carryover_cost': co,
        'greedy_obj': stats(gr, 'objective'),
        'mpc_obj': stats(mpc, 'objective'),
        'greedy_time': stats(gr, 'time'),   # averaged across seeds/machines
        'mpc_time': stats(mpc, 'time'),
        'greedy_seeds': [r['seed'] for r in gr],
        'mpc_seeds': [r['seed'] for r in mpc],
    }
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved to {os.path.join(out_dir, 'summary.json')}")


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Read per-method results, print stats, plot.')
    p.add_argument('--out_dir', type=str, default='results/comparison')
    p.add_argument('--commit', type=int, default=None,
                   help='Override commit size (default: from run_config.json)')
    p.add_argument('--n_intervals', type=int, default=None,
                   help='Override n_intervals (default: from run_config.json)')
    args = p.parse_args()

    rc = load_run_config(args.out_dir)
    commit = args.commit or rc.get('commit', 36)
    n_intervals = args.n_intervals if args.n_intervals is not None else rc.get('n_intervals')
    co_cost = float(rc.get('carryover_cost', 0.0) or 0.0)

    config = QueueConfig()
    lambdas, mus_init = load_default_data(config)
    if n_intervals is not None:
        lambdas = lambdas[:n_intervals]
        mus_init = mus_init[:n_intervals]
    alpha1, alpha2 = config.get_alpha_arrays(size=len(lambdas))
    n = len(lambdas)

    results = {
        'full_day':  load_single(args.out_dir, 'full_day', n),
        'do_nothing': load_single(args.out_dir, 'do_nothing', n),
        'greedy':    load_seeds(args.out_dir, 'greedy'),
        'mpc':       load_seeds(args.out_dir, 'mpc'),
        'carryover_cost': co_cost,
    }

    print("=" * 60)
    print("PLOT / SUMMARIZE RESULTS")
    print(f"  out_dir:   {args.out_dir}")
    print(f"  intervals: {n}  commit: {commit}  carryover_cost: {co_cost:.2f}")
    print(f"  found:     full_day={results['full_day'] is not None}  "
          f"do_nothing={results['do_nothing'] is not None}  "
          f"greedy={len(results['greedy'])} seeds  mpc={len(results['mpc'])} seeds")
    print("=" * 60)

    print_statistics(results)
    plot_all(results, lambdas, mus_init, alpha1, alpha2, config, commit, args.out_dir)
    save_summary(results, args.out_dir)