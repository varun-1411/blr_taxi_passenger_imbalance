"""
Regenerate plots from saved results without rerunning optimization.

Usage:
    python experiments/replot_model_analysis.py --results_dir results/model_analysis
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import QueueConfig
from data import load_default_data

# Colors
TEAL='#2A9D8F'; PLUM='#9B59B6'; AMBER='#E9C46A'
GRAY='#7F8C8D'; BLUE='#264653'; ROSE='#E76F51'

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.labelsize': 12, 'axes.titlesize': 13,
    'legend.fontsize': 9, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.grid': True, 'grid.alpha': 0.3, 'lines.linewidth': 1.5,
})


def load_ts(results_dir, prefix, key):
    """Load a queue time series .npy file."""
    path = os.path.join(results_dir, f'{prefix}_{key}.npy')
    if os.path.exists(path):
        return np.load(path)
    return None


def replot_ss_vs_transient(results_dir, config, lambdas):
    """Regenerate SS vs Transient plots from saved data."""
    n = len(lambdas)
    t_axis = np.arange(n) * config.interval_length

    # Load summary
    summary_path = os.path.join(results_dir, 'ss_vs_transient.json')
    if not os.path.exists(summary_path):
        print(f"  No ss_vs_transient.json found in {results_dir}")
        return
    with open(summary_path) as f:
        summary = json.load(f)

    pa = summary['part_a']
    pb = summary['part_b']

    # Load controls
    controls = {}
    for name in ['brent', 'adam_nodelay', 'adam_delay']:
        ma_path = os.path.join(results_dir, f'{name}_mu_add.npy')
        mr_path = os.path.join(results_dir, f'{name}_mu_remove.npy')
        if os.path.exists(ma_path):
            controls[name] = {
                'mu_add': np.load(ma_path),
                'mu_remove': np.load(mr_path),
            }

    # Load queue time series
    ts_data = {}
    for prefix in ['dn_nodelay', 'brent_on_tr', 'adam_nd_eval',
                    'dn_delay', 'brent_on_delay', 'adam_d_eval']:
        ts_data[prefix] = {}
        for key in ['pax_queue_ts', 'taxi_queue_ts', 'resv_queue_ts']:
            arr = load_ts(results_dir, prefix, key)
            if arr is not None:
                ts_data[prefix][key] = arr

    titles = ['Passenger Queue', 'Taxi Queue (Pickup)', 'Staging Queue']
    keys = ['pax_queue_ts', 'taxi_queue_ts', 'resv_queue_ts']

    # ── Plot 1: Queue lengths (Part A) ──
    if all(k in ts_data.get('brent_on_tr', {}) for k in keys):
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        for ax, title, key in zip(axes, titles, keys):
            if key in ts_data.get('brent_on_tr', {}):
                ax.plot(ts_data['brent_on_tr'][key], color=AMBER, lw=1.5, ls='--',
                        label='Brent SS controls (on transient)')
            if key in ts_data.get('adam_nd_eval', {}):
                ax.plot(ts_data['adam_nd_eval'][key], color=TEAL, lw=1.8,
                        label='Adam transient-optimal')
            if key in ts_data.get('dn_nodelay', {}):
                ax.plot(ts_data['dn_nodelay'][key], color=GRAY, lw=1.0, ls=':',
                        label='Do-nothing', alpha=0.7)
            ax.set_ylabel('Expected Length'); ax.set_title(title); ax.legend(fontsize=8)
        axes[-1].set_xlabel('Interval')
        fig.suptitle('Zero Delays: SS vs Transient Controls', fontsize=12, y=1.01)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'ss_vs_tr_queues.png'), dpi=300); plt.close()
        print("  Saved ss_vs_tr_queues.png")

    # ── Plot 2: Control profiles ──
    if controls:
        fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
        ax = axes[0]
        ax2 = ax.twinx()
        ax2.fill_between(t_axis, lambdas, alpha=0.08, color=BLUE)
        ax2.set_ylabel(r'$\lambda$', color=BLUE)
        if 'brent' in controls:
            ax.plot(t_axis, controls['brent']['mu_add'], color=AMBER, lw=1.5, ls='--', label='Brent SS')
        if 'adam_nodelay' in controls:
            ax.plot(t_axis, controls['adam_nodelay']['mu_add'], color=TEAL, lw=1.8, label='Adam (d=0)')
        if 'adam_delay' in controls:
            ax.plot(t_axis, controls['adam_delay']['mu_add'], color=ROSE, lw=1.5, ls='-.', label='Adam (with delay)')
        ax.set_ylabel(r'$\mu^+$'); ax.set_title('Dispatch Rate'); ax.legend(fontsize=8)

        ax = axes[1]
        if 'brent' in controls:
            ax.plot(t_axis, controls['brent']['mu_remove'], color=AMBER, lw=1.5, ls='--', label='Brent SS')
        if 'adam_nodelay' in controls:
            ax.plot(t_axis, controls['adam_nodelay']['mu_remove'], color=TEAL, lw=1.8, label='Adam (d=0)')
        if 'adam_delay' in controls:
            ax.plot(t_axis, controls['adam_delay']['mu_remove'], color=ROSE, lw=1.5, ls='-.', label='Adam (with delay)')
        ax.set_xlabel('Time (min)'); ax.set_ylabel(r'$\mu^-$')
        ax.set_title('Removal Rate'); ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'ss_vs_tr_controls.png'), dpi=300); plt.close()
        print("  Saved ss_vs_tr_controls.png")

    # ── Plot 3: Paper summary figure ──
    dn_obj = pa['do_nothing']
    ss_reported = pa['brent_ss_reported']
    ss_true = pa['brent_on_transient']
    adam_obj = pa['adam_transient']
    stationarity_pct = pa['stationarity_pct']
    transient_gain_pct = pa['transient_gain_pct']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Part A bars
    ax = axes[0]
    labels_a = ['Do-Nothing', 'Brent\n(SS reported)', 'Brent on\nTransient', 'Adam\nTransient']
    objs_a = [dn_obj, ss_reported, ss_true, adam_obj]
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
    ax.annotate('', xy=(3, adam_obj), xytext=(2, ss_true),
                arrowprops=dict(arrowstyle='<->', color=TEAL, lw=1.5))
    mid_y2 = (ss_true + adam_obj) / 2
    ax.text(2.5, mid_y2, f'Transient\ngain: {transient_gain_pct:+.1f}%',
            ha='center', va='center', fontsize=7, color=TEAL,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))

    # Part B bars
    ax = axes[1]
    labels_b = ['Do-Nothing\n(with delay)', 'Brent SS\n+ delays', 'Adam\n+ delays']
    objs_b = [pb['do_nothing_delay'], pb['brent_on_delay'], pb['adam_delay']]
    colors_b = [GRAY, AMBER, TEAL]
    bars = ax.bar(range(3), objs_b, color=colors_b, alpha=0.85)
    for bar, obj in zip(bars, objs_b):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                f'{obj:,.0f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(range(3)); ax.set_xticklabels(labels_b, fontsize=8)
    ax.set_ylabel('Objective')
    ax.set_title(f'Part B: With Delays ($d_d={config.delay_non_reserved:.0f}$, '
                 f'$d_e={config.delay_extra:.0f}$)')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'ss_vs_tr_summary.png'), dpi=300); plt.close()
    print("  Saved ss_vs_tr_summary.png")

    # ── Plot 4: Per-interval gap ──
    if all(k in ts_data.get('brent_on_tr', {}) and k in ts_data.get('adam_nd_eval', {})
           for k in keys):
        fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
        for ax, title, key in zip(axes, titles, keys):
            diff = ts_data['brent_on_tr'][key] - ts_data['adam_nd_eval'][key]
            ax.plot(diff, color=PLUM, lw=1.2)
            ax.axhline(0, color='black', lw=0.5)
            ax.fill_between(range(len(diff)), diff, alpha=0.15, color=PLUM)
            ax.set_ylabel('Brent $-$ Adam')
            ax.set_title(f'{title} — Stationarity Gap (per interval)')
        axes[-1].set_xlabel('Interval')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'ss_vs_tr_gap.png'), dpi=300); plt.close()
        print("  Saved ss_vs_tr_gap.png")

    # Print summary
    print(f"\n  RESULTS SUMMARY:")
    print(f"  Part A (zero delays):")
    print(f"    Do-nothing:          {dn_obj:,.0f}")
    print(f"    Brent SS reported:   {ss_reported:,.0f}")
    print(f"    Brent on transient:  {ss_true:,.0f} (stationarity error: {stationarity_pct:+.1f}%)")
    print(f"    Adam transient:      {adam_obj:,.0f} (transient gain: {transient_gain_pct:+.1f}%)")
    print(f"  Part B (with delays):")
    print(f"    Do-nothing:          {pb['do_nothing_delay']:,.0f}")
    print(f"    Brent SS + delays:   {pb['brent_on_delay']:,.0f}")
    print(f"    Adam + delays:       {pb['adam_delay']:,.0f}")


def replot_numerical(results_dir):
    """Regenerate numerical comparison plots from saved JSON."""
    json_path = os.path.join(results_dir, 'numerical_comparison.json')
    if not os.path.exists(json_path):
        print(f"  No numerical_comparison.json found")
        return

    with open(json_path) as f:
        data = json.load(f)

    results = data['results']
    timings = data['timings']
    solvers = list(results.keys())
    solver_labels = {'uniformization': 'Uniformization', 'rk4': 'RK4', 'expm': 'Expm (Krylov)'}
    colors = {'uniformization': TEAL, 'rk4': ROSE, 'expm': PLUM}

    pairs = [('uniformization', 'rk4'), ('uniformization', 'expm'), ('rk4', 'expm')]
    pair_labels = {
        ('uniformization', 'rk4'): 'Unif $-$ RK4',
        ('uniformization', 'expm'): 'Unif $-$ Expm',
        ('rk4', 'expm'): 'RK4 $-$ Expm',
    }
    pair_colors = {
        ('uniformization', 'rk4'): ROSE,
        ('uniformization', 'expm'): PLUM,
        ('rk4', 'expm'): AMBER,
    }

    # Plot 1: Timing + autodiff
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    bars = ax.bar(range(len(solvers)),
                  [timings[s] for s in solvers],
                  color=[colors[s] for s in solvers], alpha=0.85)
    ax.set_xticks(range(len(solvers)))
    ax.set_xticklabels([solver_labels.get(s, s) for s in solvers])
    for bar, t in zip(bars, [timings[s] for s in solvers]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                f'{t:.1f}s', ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('Time (s)'); ax.set_title('Computation Time')

    ax = axes[1]
    autodiff = {'uniformization': True, 'rk4': True, 'expm': False}
    bar_colors = [TEAL if autodiff.get(s, False) else GRAY for s in solvers]
    ax.bar(range(len(solvers)), [1 if autodiff.get(s, False) else 0 for s in solvers],
           color=bar_colors, alpha=0.85)
    ax.set_xticks(range(len(solvers)))
    ax.set_xticklabels([solver_labels.get(s, s) for s in solvers])
    ax.set_yticks([0, 1]); ax.set_yticklabels(['No', 'Yes'])
    ax.set_ylabel('Autodiff Support'); ax.set_title('Gradient Computation')
    ax.set_ylim(-0.1, 1.3)
    for i, s in enumerate(solvers):
        ax.text(i, 0.5, 'Yes' if autodiff.get(s, False) else 'No',
                ha='center', va='center', fontsize=11, fontweight='bold',
                color='white' if autodiff.get(s, False) else 'black')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'numerical_timing.png'), dpi=300); plt.close()
    print("  Saved numerical_timing.png")

    # Plot 2: Summary
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Objectives
    ax = axes[0]
    objs = [results[s]['objective'] for s in solvers]
    bars = ax.bar(range(len(solvers)), objs,
                  color=[colors[s] for s in solvers], alpha=0.85)
    ax.set_xticks(range(len(solvers)))
    ax.set_xticklabels([solver_labels.get(s, s) for s in solvers], fontsize=9)
    ax.set_ylabel('Objective'); ax.set_title('Solution Quality')
    if max(objs) > 0:
        margin = max(max(objs) - min(objs), max(objs) * 0.001)
        ax.set_ylim(min(objs) - margin * 2, max(objs) + margin * 2)
    for bar, obj in zip(bars, objs):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                f'{obj:.2f}', ha='center', va='bottom', fontsize=8)

    # Runtime
    ax = axes[1]
    bars = ax.bar(range(len(solvers)), [timings[s] for s in solvers],
                  color=[colors[s] for s in solvers], alpha=0.85)
    ax.set_xticks(range(len(solvers)))
    ax.set_xticklabels([solver_labels.get(s, s) for s in solvers], fontsize=9)
    ax.set_ylabel('Time (s)'); ax.set_title('Runtime')
    for bar, t in zip(bars, [timings[s] for s in solvers]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                f'{t:.1f}s', ha='center', va='bottom', fontsize=9)

    # Pairwise diffs
    ax = axes[2]
    if 'pairwise_diffs' in data:
        pd_data = data['pairwise_diffs']
        pair_keys = list(pd_data.keys())
        diffs = [pd_data[k]['obj_diff'] for k in pair_keys]
        pn = [pair_labels.get(tuple(k.split('_vs_')), k) for k in pair_keys]
        pc = [pair_colors.get(tuple(k.split('_vs_')), GRAY) for k in pair_keys]
        bars = ax.bar(range(len(pair_keys)), diffs, color=pc, alpha=0.85)
        ax.set_xticks(range(len(pair_keys)))
        ax.set_xticklabels(pn, fontsize=9)
        ax.set_ylabel('|Δ Objective|'); ax.set_title('Pairwise Agreement')
        for bar, d in zip(bars, diffs):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                    f'{d:.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'numerical_summary.png'), dpi=300); plt.close()
    print("  Saved numerical_summary.png")

    # Print table
    print(f"\n  NUMERICAL COMPARISON:")
    print(f"  {'Method':<20} {'Objective':>14} {'Time':>8}")
    print(f"  {'─'*44}")
    for s in solvers:
        print(f"  {solver_labels.get(s,s):<20} {results[s]['objective']:>14.2f} {timings[s]:>7.1f}s")

    if 'pairwise_diffs' in data:
        print(f"\n  Pairwise |Δ objective|:")
        for k, v in data['pairwise_diffs'].items():
            print(f"    {k:<30}: {v['obj_diff']:.6f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Regenerate plots from saved results')
    parser.add_argument('--results_dir', type=str, default='results/model_analysis')
    parser.add_argument('--plot', type=str, default='all',
                        choices=['all', 'numerical', 'ss_vs_transient'])
    args = parser.parse_args()

    config = QueueConfig()
    lambdas, mus_init = load_default_data(config)

    print(f"Regenerating plots from {args.results_dir}/")

    if args.plot in ('all', 'numerical'):
        replot_numerical(args.results_dir)

    if args.plot in ('all', 'ss_vs_transient'):
        replot_ss_vs_transient(args.results_dir, config, lambdas)

    print(f"\nDone!")