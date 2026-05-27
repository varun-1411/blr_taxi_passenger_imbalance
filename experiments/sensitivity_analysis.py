"""
Sensitivity Analysis for Airport Taxi Queue Optimizer.

6 analyses:
  1. Delay sensitivity         — When do delays matter?
  2. Commit size / lookahead   — What planning horizon is needed?
  3. VoT & cost sensitivity    — How does the passenger-driver tradeoff work?
  4. Demand scaling            — Is the policy robust under congestion?
  5. Interval length delta     — Is PCA discretization accurate?
  6. SS vs Transient gap       — When does transient modeling become necessary?

Usage:
    python experiments/sensitivity_analysis.py --analysis all --n_intervals 20 --max_iter 100
    python experiments/sensitivity_analysis.py --analysis delay --n_intervals 288 --max_iter 500
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
from data import load_default_data, aggregate_passengers
from optimizer_utils import (
    build_eff_nr_zero_pad,
    compute_objective_detailed,
    propagate_pi,
    make_pi0,
    optimize_full_day,
    optimize_greedy,
    run_do_nothing,
    get_distribution_stats,
)

# ══════════════════════════════════════════════════════════════
# PAPER COLOR SCHEME (paperfigs.sty)
# ══════════════════════════════════════════════════════════════

TEAL  = '#2A9D8F'
PLUM  = '#9B59B6'
AMBER = '#E9C46A'
GRAY  = '#7F8C8D'
BLUE  = '#264653'
ROSE  = '#E76F51'

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

def run_multi(fn, n_seeds, base_seed, **kw):
    objs=[]; adds=[]; rems=[]; dets=[]
    for s in range(n_seeds):
        r = fn(seed=base_seed+s, **kw)
        objs.append(r['objective']); adds.append(r['mu_add']); rems.append(r['mu_remove'])
        if 'details' in r and r['details']: dets.append(r['details'])
    res = {'objectives': np.array(objs), 'mu_add': np.array(adds),
           'mu_remove': np.array(rems),
           'obj_mean': np.mean(objs), 'obj_std': np.std(objs)}
    if dets: res['details_mean'] = {k: np.mean([d[k] for d in dets]) for k in dets[0]}
    return res


# ══════════════════════════════════════════════════════════════
# ANALYSIS 1: DELAY
# ══════════════════════════════════════════════════════════════

def analysis_delay(lambdas, mus_init, alpha1, alpha2, base_config,
                   n_seeds=2, max_iter=200, lr=1.0, epsilon=0.1,
                   base_seed=42, out_dir='results/sensitivity'):
    print("\n" + "="*60 + "\nANALYSIS 1: DELAY\n" + "="*60)
    dnr = [0, 5, 10, 15, 20]; dex = [0, 5, 10, 15, 20]; results = {}
    for dr in dnr:
        for de in dex:
            cfg = copy.deepcopy(base_config)
            cfg.delay_non_reserved = float(dr); cfg.delay_extra = float(de)
            k = f"{dr}_{de}"
            print(f"  d_d={dr}, d_e={de}...", end=' ', flush=True)
            fd = run_multi(optimize_full_day, n_seeds, base_seed,
                lambdas=lambdas, mus_init=mus_init, alpha1=alpha1, alpha2=alpha2,
                config=cfg, max_iter=max_iter, lr=lr, epsilon=epsilon)
            dn = run_do_nothing(lambdas, mus_init, alpha1, alpha2, cfg)
            det = fd.get('details_mean', {})
            results[k] = {'fd': fd['obj_mean'], 'dn': dn['objective'],
                          'impr': (dn['objective']-fd['obj_mean'])/dn['objective']*100,
                          'pax_wait': det.get('pax_wait', 0), 'taxi_idle': det.get('taxi_idle', 0)}
            print(f"FD={fd['obj_mean']:.0f}, impr={results[k]['impr']:.1f}%")
    _save(results, out_dir, 'delay.json')

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for idx, (fld, ttl, cm) in enumerate([
        ('fd', 'Full-Day Cost', 'YlOrRd'), ('impr', 'Improvement (%)', 'YlGn'),
        ('pax_wait', 'Passenger Wait Cost', 'Reds'), ('taxi_idle', 'Taxi Idle Cost', 'Blues')]):
        ax = axes[idx//2][idx%2]; mat = np.zeros((len(dnr), len(dex)))
        for i, dr in enumerate(dnr):
            for j, de in enumerate(dex): mat[i,j] = results[f"{dr}_{de}"][fld]
        im = ax.imshow(mat, cmap=cm, aspect='auto')
        ax.set_xticks(range(len(dex))); ax.set_xticklabels(dex)
        ax.set_yticks(range(len(dnr))); ax.set_yticklabels(dnr)
        ax.set_xlabel(r'$\delta_e$ extra (min)'); ax.set_ylabel(r'$\delta_d$ (min)'); ax.set_title(ttl)
        for i in range(len(dnr)):
            for j in range(len(dex)):
                f = f'{mat[i,j]:.0f}' if 'Cost' in ttl else f'{mat[i,j]:.1f}%'
                ax.text(j, i, f, ha='center', va='center', fontsize=7)
        plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'delay.png'), dpi=300); plt.close()
    return results


# ══════════════════════════════════════════════════════════════
# ANALYSIS 2: COMMIT SIZE
# ══════════════════════════════════════════════════════════════

def analysis_commit(lambdas, mus_init, alpha1, alpha2, config,
                    n_seeds=2, max_iter=200, lr=1.0, epsilon=0.1,
                    base_seed=42, out_dir='results/sensitivity'):
    print("\n" + "="*60 + "\nANALYSIS 2: COMMIT SIZE\n" + "="*60)
    n = len(lambdas); ps = config.get_delay_blocks()[1]
    commits = sorted(set([c for c in [3, 6, 9, 12, 18, 24, 36, 48, 72, n] if c <= n]))

    fd = run_multi(optimize_full_day, n_seeds, base_seed,
        lambdas=lambdas, mus_init=mus_init, alpha1=alpha1, alpha2=alpha2,
        config=config, max_iter=max_iter, lr=lr, epsilon=epsilon)
    dn = run_do_nothing(lambdas, mus_init, alpha1, alpha2, config)
    print(f"  FD={fd['obj_mean']:.0f}, DN={dn['objective']:.0f}")

    results = {'fd': fd['obj_mean'], 'dn': dn['objective'], 'greedy': {}}
    for c in commits:
        buf = min(ps, n-c) if c < n else 0
        print(f"  commit={c}...", end=' ', flush=True)
        gr = run_multi(optimize_greedy, n_seeds, base_seed,
            lambdas=lambdas, mus_init=mus_init, alpha1=alpha1, alpha2=alpha2,
            config=config, commit_size=c, buffer_size=buf,
            max_iter=max_iter, lr=lr, epsilon=epsilon)
        gap = (gr['obj_mean']-fd['obj_mean'])/fd['obj_mean']*100
        results['greedy'][c] = {'mean': gr['obj_mean'], 'std': gr['obj_std'], 'gap': gap}
        print(f"obj={gr['obj_mean']:.0f} (gap={gap:+.2f}%)")
    _save(results, out_dir, 'commit.json')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ch = [c*config.interval_length/60 for c in commits]
    ms = [results['greedy'][c]['mean'] for c in commits]
    ss = [results['greedy'][c]['std'] for c in commits]
    gs = [results['greedy'][c]['gap'] for c in commits]

    axes[0].errorbar(ch, ms, yerr=ss, fmt='o-', color=ROSE, capsize=4, label='Greedy')
    axes[0].axhline(y=fd['obj_mean'], color=TEAL, lw=2, label='Full-Day')
    axes[0].axhline(y=dn['objective'], color=GRAY, ls='--', label='Do-Nothing')
    axes[0].set_xlabel('Commit (hours)'); axes[0].set_ylabel('Objective')
    axes[0].set_title('Cost vs Planning Horizon'); axes[0].legend()

    axes[1].bar(range(len(commits)), gs, color=ROSE, alpha=0.8)
    axes[1].set_xticks(range(len(commits)))
    axes[1].set_xticklabels([f'{h:.1f}h' for h in ch], rotation=45)
    axes[1].set_xlabel('Commit'); axes[1].set_ylabel('Gap from Full-Day (%)')
    axes[1].set_title('Myopia Cost')
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'commit.png'), dpi=300); plt.close()
    return results


# ══════════════════════════════════════════════════════════════
# ANALYSIS 3: VoT & COST
# ══════════════════════════════════════════════════════════════

def analysis_cost(lambdas, mus_init, alpha1, alpha2, base_config,
                  n_seeds=2, max_iter=200, lr=1.0, epsilon=0.1,
                  base_seed=42, out_dir='results/sensitivity'):
    """
    Comprehensive cost sensitivity.

    4 individual sweeps + 1 heatmap:
      (a) c_a: dispatch cost       → crossover: when does optimizer stop dispatching?
      (b) c_bp: passenger blocking  → when does blocking penalty drive dispatch?
      (c) α₁: passenger VoT        → how does pax time-value affect policy?
      (d) α₂: driver VoT           → how does driver cost affect policy?
      (e) α₁ × α₂ 2D heatmap      → interaction between pax and driver VoT

    For each, records: objective, total μ⁺, total μ⁻, E[n], pax_wait, taxi_idle
    """
    print("\n" + "="*60 + "\nANALYSIS 3: COST SENSITIVITY\n" + "="*60)
    dt = base_config.interval_length
    results = {}

    # ── Shared eval function ──
    def _sweep(param_name, values, make_config_fn, label):
        """Run one sweep and return results dict."""
        print(f"\n  {label}:")
        sweep = {}
        for v in values:
            cfg, a1, a2 = make_config_fn(v)
            print(f"    {param_name}={v}...", end=' ', flush=True)
            fd = run_multi(optimize_full_day, n_seeds, base_seed,
                lambdas=lambdas, mus_init=mus_init, alpha1=a1, alpha2=a2,
                config=cfg, max_iter=max_iter, lr=lr, epsilon=epsilon)
            dn = run_do_nothing(lambdas, mus_init, a1, a2, cfg)
            det = fd.get('details_mean', {})
            sweep[str(v)] = {
                'value': v,
                'obj': fd['obj_mean'], 'obj_std': fd['obj_std'],
                'dn_obj': dn['objective'],
                'impr': (dn['objective'] - fd['obj_mean']) / dn['objective'] * 100,
                'mu_add': float(fd['mu_add'].mean(axis=0).sum()),
                'mu_remove': float(fd['mu_remove'].mean(axis=0).sum()),
                'pax_wait': det.get('pax_wait', 0),
                'taxi_idle': det.get('taxi_idle', 0),
                'pax_block': det.get('pax_block', 0),
                'add_cost': det.get('add_cost', 0),
            }
            ratio_str = f", ratio={v/(alpha1[0]*dt):.2f}" if param_name == 'c_a' else ""
            print(f"obj={fd['obj_mean']:.0f}, mu+={sweep[str(v)]['mu_add']:.1f}{ratio_str}")
        return sweep

    # ── (a) Dispatch cost c_a ──
    ca_vals = [10, 20, 30, 50, 75, 100, 150, 200, 300, 500]
    def make_ca(v):
        cfg = copy.deepcopy(base_config); cfg.cost_per_vehicle_add = v
        return cfg, alpha1, alpha2
    results['c_a'] = _sweep('c_a', ca_vals, make_ca, 'Dispatch cost c_a')

    # ── (b) Passenger blocking penalty c_bp ──
    cbp_vals = [50, 100, 200, 400, 600, 1000, 2000, 5000]
    def make_cbp(v):
        cfg = copy.deepcopy(base_config); cfg.cost_pax_lost = v
        return cfg, alpha1, alpha2
    results['c_bp'] = _sweep('c_bp', cbp_vals, make_cbp, 'Blocking penalty c_bp')

    # ── (c) Passenger VoT α₁ ──
    a1_mults = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
    def make_a1(v):
        return base_config, alpha1 * v, alpha2
    results['alpha1'] = _sweep('α₁_mult', a1_mults, make_a1, 'Passenger VoT α₁')

    # ── (d) Driver VoT α₂ ──
    a2_mults = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
    def make_a2(v):
        return base_config, alpha1, alpha2 * v
    results['alpha2'] = _sweep('α₂_mult', a2_mults, make_a2, 'Driver VoT α₂')

    # ── (e) 2D heatmap: α₁ × α₂ ──
    hm_mults = [0.5, 0.75, 1.0, 1.5, 2.0]
    results['vot_2d'] = {}
    print(f"\n  VoT 2D sweep (α₁ × α₂):")
    for m1 in hm_mults:
        for m2 in hm_mults:
            k = f"{m1}_{m2}"
            print(f"    α₁×{m1}, α₂×{m2}...", end=' ', flush=True)
            fd = run_multi(optimize_full_day, n_seeds, base_seed,
                lambdas=lambdas, mus_init=mus_init,
                alpha1=alpha1*m1, alpha2=alpha2*m2,
                config=base_config, max_iter=max_iter, lr=lr, epsilon=epsilon)
            det = fd.get('details_mean', {})
            results['vot_2d'][k] = {
                'obj': fd['obj_mean'],
                'mu_add': float(fd['mu_add'].mean(axis=0).sum()),
                'pax_wait': det.get('pax_wait', 0),
                'taxi_idle': det.get('taxi_idle', 0),
            }
            print(f"obj={fd['obj_mean']:.0f}")

    _save(results, out_dir, 'cost_sensitivity.json')

    # ── PLOTS ──
    # Figure 1: Individual sweeps (2×2 grid, each with dual y-axis)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sweep_configs = [
        ('c_a', ca_vals, r'Dispatch cost $c_a$ (₹)', 'c_a', False),
        ('c_bp', cbp_vals, r'Blocking penalty $c_{bp}$ (₹)', 'c_bp', False),
        ('alpha1', a1_mults, r'Passenger VoT $\alpha_1$ multiplier', 'α₁', True),
        ('alpha2', a2_mults, r'Driver VoT $\alpha_2$ multiplier', 'α₂', True),
    ]

    for idx, (key, vals, xlabel, short, is_mult) in enumerate(sweep_configs):
        ax = axes[idx // 2][idx % 2]
        sw = results[key]

        objs = [sw[str(v)]['obj'] for v in vals]
        adds = [sw[str(v)]['mu_add'] for v in vals]
        dn_objs = [sw[str(v)]['dn_obj'] for v in vals]

        # Left axis: objective
        l1, = ax.plot(vals, objs, 'o-', color=TEAL, lw=2, markersize=6, label='Optimized')
        ax.plot(vals, dn_objs, 's--', color=GRAY, lw=1, markersize=4, alpha=0.6, label='Do-Nothing')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Objective', color=TEAL)
        ax.tick_params(axis='y', labelcolor=TEAL)

        # Right axis: dispatch volume
        ax2 = ax.twinx()
        l2, = ax2.plot(vals, adds, '^-', color=ROSE, lw=1.5, markersize=5, label=r'Total $\mu^+$')
        ax2.set_ylabel(r'Total $\mu^+$', color=ROSE)
        ax2.tick_params(axis='y', labelcolor=ROSE)

        # Crossover line for c_a
        if key == 'c_a':
            crossover = alpha1[0] * dt
            ax.axvline(x=crossover, color=AMBER, ls='--', lw=1.5, alpha=0.7,
                       label=f'$\\alpha_1 \\cdot \\Delta = {crossover:.0f}$')

        ax.legend(handles=[l1, l2], loc='upper left' if idx < 2 else 'best', fontsize=8)
        ax.set_title(f'{short} Sweep')
        if not is_mult and max(vals) / min(vals) > 10:
            ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'cost_sweeps.png'), dpi=300); plt.close()

    # Figure 2: Cost breakdown for c_a sweep
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Stacked cost components vs c_a
    ax = axes[0]
    comps = ['pax_wait', 'taxi_idle', 'add_cost', 'pax_block']
    comp_labels = ['Pax Wait', 'Taxi Idle', 'Dispatch Cost', 'Pax Blocking']
    comp_colors = [ROSE, TEAL, BLUE, AMBER]
    bottom = np.zeros(len(ca_vals))
    for ck, cl, cc in zip(comps, comp_labels, comp_colors):
        vals_arr = np.array([results['c_a'][str(v)].get(ck, 0) for v in ca_vals])
        ax.bar(range(len(ca_vals)), vals_arr, bottom=bottom, color=cc, alpha=0.8, label=cl)
        bottom += vals_arr
    ax.set_xticks(range(len(ca_vals)))
    ax.set_xticklabels([f'{v}' for v in ca_vals], rotation=45, fontsize=8)
    ax.set_xlabel(r'$c_a$ (₹)'); ax.set_ylabel('Cost')
    ax.set_title(r'Cost Breakdown vs $c_a$'); ax.legend(fontsize=7)

    # Panel 2: Dispatch volume + E[n] vs c_a
    ax = axes[1]
    adds = [results['c_a'][str(v)]['mu_add'] for v in ca_vals]
    ax.bar(range(len(ca_vals)), adds, color=TEAL, alpha=0.8)
    ax.set_xticks(range(len(ca_vals)))
    ax.set_xticklabels([f'{v}' for v in ca_vals], rotation=45, fontsize=8)
    ax.set_xlabel(r'$c_a$ (₹)'); ax.set_ylabel(r'Total $\mu^+$')
    ax.set_title('Dispatch Volume')
    # Annotate crossover
    crossover = alpha1[0] * dt
    for i, v in enumerate(ca_vals):
        if v >= crossover and (i == 0 or ca_vals[i-1] < crossover):
            ax.axvline(x=i-0.5, color=AMBER, ls='--', lw=1.5)
            ax.text(i, max(adds)*0.9, f'$c_a = \\alpha_1 \\Delta$\n={crossover:.0f}',
                    ha='center', fontsize=7, color=AMBER)

    # Panel 3: Improvement % vs c_a
    ax = axes[2]
    imprs = [results['c_a'][str(v)]['impr'] for v in ca_vals]
    ax.plot(ca_vals, imprs, 'o-', color=PLUM, lw=2, markersize=7)
    ax.set_xlabel(r'$c_a$ (₹)'); ax.set_ylabel('Improvement over DN (%)')
    ax.set_title('Value of Optimization')
    ax.axvline(x=crossover, color=AMBER, ls='--', lw=1.5, alpha=0.7,
               label=f'$\\alpha_1 \\cdot \\Delta = {crossover:.0f}$')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'cost_ca_detail.png'), dpi=300); plt.close()

    # Figure 3: 2D heatmaps
    nm = len(hm_mults)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    for col, (fld, ttl, cm) in enumerate([
        ('obj', 'Objective', 'YlOrRd'),
        ('mu_add', r'Total $\mu^+$', 'Purples'),
        ('pax_wait', 'Passenger Wait Cost', 'Reds')]):
        ax = axes[col]; mat = np.zeros((nm, nm))
        for i, m1 in enumerate(hm_mults):
            for j, m2 in enumerate(hm_mults):
                mat[i, j] = results['vot_2d'][f"{m1}_{m2}"][fld]
        im = ax.imshow(mat, cmap=cm, aspect='auto')
        ax.set_xticks(range(nm)); ax.set_xticklabels([f'{m}x' for m in hm_mults])
        ax.set_yticks(range(nm)); ax.set_yticklabels([f'{m}x' for m in hm_mults])
        ax.set_xlabel(r'Driver VoT ($\alpha_2$)')
        ax.set_ylabel(r'Pax VoT ($\alpha_1$)')
        ax.set_title(ttl)
        for i in range(nm):
            for j in range(nm):
                f = f'{mat[i,j]:.0f}'
                ax.text(j, i, f, ha='center', va='center', fontsize=7)
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'cost_vot_heatmap.png'), dpi=300); plt.close()

    # Figure 4: Crossover analysis — ratio c_a / (α₁·Δ) vs dispatch
    fig, ax = plt.subplots(figsize=(8, 5))
    ratios = [v / (alpha1[0] * dt) for v in ca_vals]
    adds = [results['c_a'][str(v)]['mu_add'] for v in ca_vals]
    ax.plot(ratios, adds, 'o-', color=TEAL, lw=2, markersize=8)
    ax.axvline(x=1.0, color=AMBER, ls='--', lw=2, label=r'$c_a / (\alpha_1 \cdot \Delta) = 1$')
    ax.fill_betweenx([0, max(adds)*1.1], 0, 1, alpha=0.05, color=TEAL)
    ax.fill_betweenx([0, max(adds)*1.1], 1, max(ratios)*1.1, alpha=0.05, color=ROSE)
    ax.text(0.5, max(adds)*0.85, 'Dispatch\nefficient', ha='center', fontsize=10, color=TEAL)
    ax.text(min(max(ratios)*0.7, 3), max(adds)*0.85, 'Waiting\ncheaper', ha='center', fontsize=10, color=ROSE)
    ax.set_xlabel(r'Cost ratio $c_a / (\alpha_1 \cdot \Delta)$')
    ax.set_ylabel(r'Total dispatch $\sum \mu^+$')
    ax.set_title('Dispatch Policy Crossover')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'cost_crossover.png'), dpi=300); plt.close()

    # Print summary table
    print(f"\n  {'─'*80}")
    print(f"  CROSSOVER ANALYSIS (c_a sweep):")
    print(f"  α₁={alpha1[0]:.2f}, Δ={dt:.0f}min, crossover c_a = α₁·Δ = {alpha1[0]*dt:.1f}")
    print(f"  {'c_a':>8} {'ratio':>8} {'obj':>10} {'μ⁺':>8} {'impr%':>8} {'pax_wait':>10} {'add_cost':>10}")
    print(f"  {'─'*66}")
    for v in ca_vals:
        s = results['c_a'][str(v)]
        r = v / (alpha1[0] * dt)
        marker = ' ← crossover' if abs(r - 1.0) < 0.15 else ''
        print(f"  {v:>8} {r:>8.2f} {s['obj']:>10.0f} {s['mu_add']:>8.1f} "
              f"{s['impr']:>7.1f}% {s['pax_wait']:>10.0f} {s['add_cost']:>10.0f}{marker}")

    print(f"\n  Plots saved to {out_dir}/")
    return results


# ══════════════════════════════════════════════════════════════
# ANALYSIS 4: DEMAND SCALING
# ══════════════════════════════════════════════════════════════

def analysis_demand(lambdas, mus_init, alpha1, alpha2, config,
                    n_seeds=2, max_iter=200, lr=1.0, epsilon=0.1,
                    base_seed=42, out_dir='results/sensitivity'):
    print("\n" + "="*60 + "\nANALYSIS 4: DEMAND SCALING\n" + "="*60)
    scales = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]; results = {}
    for s in scales:
        ls = lambdas*s; print(f"  lam x{s}...", end=' ', flush=True)
        fd = run_multi(optimize_full_day, n_seeds, base_seed,
            lambdas=ls, mus_init=mus_init, alpha1=alpha1, alpha2=alpha2,
            config=config, max_iter=max_iter, lr=lr, epsilon=epsilon)
        dn = run_do_nothing(ls, mus_init, alpha1, alpha2, config)
        impr = (dn['objective']-fd['obj_mean'])/dn['objective']*100
        results[str(s)] = {'fd': fd['obj_mean'], 'dn': dn['objective'], 'impr': impr,
                           'mu_add': float(fd['mu_add'].mean(axis=0).sum())}
        print(f"FD={fd['obj_mean']:.0f}, impr={impr:.1f}%")
    _save(results, out_dir, 'demand.json')

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fds = [results[str(s)]['fd'] for s in scales]
    dns = [results[str(s)]['dn'] for s in scales]
    ims = [results[str(s)]['impr'] for s in scales]
    ads = [results[str(s)]['mu_add'] for s in scales]

    axes[0].plot(scales, fds, 'o-', color=TEAL, label='Full-Day')
    axes[0].plot(scales, dns, 's--', color=GRAY, label='Do-Nothing')
    axes[0].set_xlabel('Demand Scale'); axes[0].set_ylabel('Objective')
    axes[0].set_title('Cost vs Demand'); axes[0].legend()

    axes[1].bar(range(len(scales)), ims, color=TEAL, alpha=0.8)
    axes[1].set_xticks(range(len(scales))); axes[1].set_xticklabels([f'{s}' for s in scales])
    axes[1].set_xlabel('Scale'); axes[1].set_ylabel('Improvement (%)')
    axes[1].set_title('Value of Optimization')

    axes[2].plot(scales, ads, 'o-', color=ROSE)
    axes[2].set_xlabel('Scale'); axes[2].set_ylabel(r'Total $\mu^+$')
    axes[2].set_title('Dispatch Volume')

    plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'demand.png'), dpi=300); plt.close()
    return results


# ══════════════════════════════════════════════════════════════
# ANALYSIS 5: INTERVAL LENGTH
# ══════════════════════════════════════════════════════════════

def analysis_delta(lambdas_base, mus_base, alpha1_base, alpha2_base, base_config,
                   n_seeds=2, max_iter=200, lr=1.0, epsilon=0.1,
                   base_seed=42, out_dir='results/sensitivity'):
    print("\n" + "="*60 + "\nANALYSIS 5: INTERVAL LENGTH\n" + "="*60)
    deltas = [1.0, 2.0, 3.0, 5.0, 10.0, 15.0]
    max_min = len(lambdas_base) * base_config.interval_length; results = {}
    for dl in deltas:
        cfg = copy.deepcopy(base_config); cfg.interval_length = dl; cfg.group_size = max(1, int(dl))
        ni = int(max_min / dl)
        # Resample
        r = base_config.interval_length / dl
        lam = np.interp(np.arange(ni), np.arange(len(lambdas_base))*r, lambdas_base)
        mus = np.interp(np.arange(ni), np.arange(len(mus_base))*r, mus_base)
        a1 = np.full(ni, alpha1_base[0]); a2 = np.resize(alpha2_base, ni)
        print(f"  D={dl:.0f}min, n={ni}...", end=' ', flush=True)
        t0 = time.time()
        fd = run_multi(optimize_full_day, n_seeds, base_seed,
            lambdas=lam, mus_init=mus, alpha1=a1, alpha2=a2,
            config=cfg, max_iter=max_iter, lr=lr, epsilon=epsilon)
        rt = time.time()-t0
        results[str(dl)] = {'delta': dl, 'n': ni, 'obj': fd['obj_mean'],
                            'std': fd['obj_std'], 'runtime': rt}
        print(f"obj={fd['obj_mean']:.0f}, time={rt:.1f}s")
    base_obj = results[str(deltas[0])]['obj']
    for k in results: results[k]['pct_err'] = (results[k]['obj']-base_obj)/base_obj*100
    _save(results, out_dir, 'delta.json')

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ds = deltas
    axes[0].errorbar(ds, [results[str(d)]['obj'] for d in ds],
                     yerr=[results[str(d)]['std'] for d in ds],
                     fmt='o-', color=TEAL, capsize=4)
    axes[0].axhline(y=base_obj, color=GRAY, ls='--', label=f'D={deltas[0]}')
    axes[0].set_xlabel(r'$\Delta$ (min)'); axes[0].set_ylabel('Objective')
    axes[0].set_title(r'Objective vs $\Delta$'); axes[0].legend()

    axes[1].plot(ds, [results[str(d)]['pct_err'] for d in ds], 'o-', color=ROSE)
    axes[1].axhline(y=1, color=GRAY, ls='--', alpha=0.5, label='1%')
    axes[1].set_xlabel(r'$\Delta$ (min)'); axes[1].set_ylabel('% Error')
    axes[1].set_title('PCA Accuracy'); axes[1].legend()

    axes[2].plot(ds, [results[str(d)]['runtime']/n_seeds for d in ds], 'o-', color=AMBER)
    axes[2].set_xlabel(r'$\Delta$ (min)'); axes[2].set_ylabel('Runtime (s)')
    axes[2].set_yscale('log'); axes[2].set_title('Computational Cost')

    plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'delta.png'), dpi=300); plt.close()
    return results


# ══════════════════════════════════════════════════════════════
# ANALYSIS 6: SS vs TRANSIENT GAP
# ══════════════════════════════════════════════════════════════

def analysis_ss_gap(lambdas, mus_init, alpha1, alpha2, base_config,
                    n_seeds=2, max_iter=200, lr=1.0, epsilon=0.1,
                    base_seed=42, out_dir='results/sensitivity'):
    """
    Compare transient-optimal vs steady-state-optimal controls evaluated
    on the transient model. The gap = price of ignoring transience.

    Steady-state: per-interval power iteration on P, no pi propagation.
    """
    print("\n" + "="*60 + "\nANALYSIS 6: SS vs TRANSIENT GAP\n" + "="*60)

    from model.generator import build_Q_non_erlang_vec, build_P_from_Q, make_state_vectors
    from optimizer_utils import unif_step, get_weight_matrix

    n = len(lambdas); tds = [0, 5, 10, 15, 20, 30, 40]; results = {}
    device = 'cpu'; dtype = torch.float32
    lt = torch.tensor(lambdas, dtype=dtype); m0t = torch.tensor(mus_init, dtype=dtype)
    a1t = torch.tensor(alpha1, dtype=dtype); a2t = torch.tensor(alpha2, dtype=dtype)

    for td in tds:
        cfg = copy.deepcopy(base_config)
        cfg.delay_non_reserved = td/3.0; cfg.delay_extra = 2.0*td/3.0
        p0, ps = cfg.get_delay_blocks()
        print(f"\n  delay={td}min (d_d={cfg.delay_non_reserved:.1f}, d_e={cfg.delay_extra:.1f})")

        # Transient-optimal
        print(f"    Transient...", end=' ', flush=True)
        tr = run_multi(optimize_full_day, n_seeds, base_seed,
            lambdas=lambdas, mus_init=mus_init, alpha1=alpha1, alpha2=alpha2,
            config=cfg, max_iter=max_iter, lr=lr, epsilon=epsilon)
        print(f"obj={tr['obj_mean']:.0f}")

        # SS-optimal (per-interval steady-state, no pi propagation)
        print(f"    SS opt...", end=' ', flush=True)
        ss_add, ss_rem = _run_ss_opt(lambdas, mus_init, alpha1, alpha2, cfg,
                                      max_iter=max_iter, lr=lr, epsilon=epsilon,
                                      seed=base_seed, device=device, dtype=dtype)
        print(f"done")

        # Evaluate SS controls on transient model
        print(f"    SS-on-transient...", end=' ', flush=True)
        ma_t = torch.tensor(ss_add, dtype=dtype); mr_t = torch.tensor(ss_rem, dtype=dtype)
        eff = build_eff_nr_zero_pad(m0t, ma_t, mr_t, p0, ps)
        pi0 = make_pi0(cfg, device, dtype)
        with torch.no_grad():
            obj_ss, det_ss = compute_objective_detailed(
                pi0, eff, lt, a1t, a2t, ma_t, mr_t, cfg, device, dtype)
        ss_on_tr = obj_ss.item()
        print(f"obj={ss_on_tr:.0f}")

        dn = run_do_nothing(lambdas, mus_init, alpha1, alpha2, cfg)
        gap = (ss_on_tr - tr['obj_mean']) / tr['obj_mean'] * 100

        tr_det = tr.get('details_mean', {})
        comp_gap = {k: det_ss.get(k, 0) - tr_det.get(k, 0) for k in tr_det} if tr_det else {}

        results[str(td)] = {'delay': td, 'transient': tr['obj_mean'],
            'ss_on_tr': ss_on_tr, 'dn': dn['objective'], 'gap_pct': gap,
            'comp_gap': comp_gap}
        print(f"    GAP = {gap:+.2f}%")

    # Save last-delay control profiles
    last_tr_add = tr['mu_add'].mean(axis=0)
    last_ss_add = ss_add
    _save(results, out_dir, 'ss_gap.json')

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    tr_o = [results[str(d)]['transient'] for d in tds]
    ss_o = [results[str(d)]['ss_on_tr'] for d in tds]
    dn_o = [results[str(d)]['dn'] for d in tds]
    gps = [results[str(d)]['gap_pct'] for d in tds]

    ax = axes[0][0]
    ax.plot(tds, tr_o, 'o-', color=TEAL, lw=2, label='Transient-optimal')
    ax.plot(tds, ss_o, 's-', color=ROSE, lw=2, label='SS on Transient')
    ax.plot(tds, dn_o, '^--', color=GRAY, label='Do-Nothing')
    ax.fill_between(tds, tr_o, ss_o, alpha=0.15, color=ROSE)
    ax.set_xlabel('Total Delay (min)'); ax.set_ylabel('Objective')
    ax.set_title('Cost Comparison'); ax.legend()

    ax = axes[0][1]
    ax.plot(tds, gps, 'o-', color=ROSE, lw=2.5, markersize=8)
    ax.fill_between(tds, 0, gps, alpha=0.2, color=ROSE)
    ax.set_xlabel('Total Delay (min)'); ax.set_ylabel('Gap (%)')
    ax.set_title('Price of Ignoring Transience')

    ax = axes[1][0]; x = np.arange(len(tds))
    cks = ['pax_wait', 'taxi_idle', 'pax_block', 'taxi_block', 'add_cost', 'remove_cost']
    cls = ['Pax Wait', 'Taxi Idle', 'Pax Block', 'Taxi Block', 'Add', 'Remove']
    ccs = [ROSE, TEAL, AMBER, PLUM, BLUE, GRAY]
    bp = np.zeros(len(tds)); bn = np.zeros(len(tds))
    for ck, cl, cc in zip(cks, cls, ccs):
        vs = [results[str(d)].get('comp_gap', {}).get(ck, 0) for d in tds]
        p = np.maximum(vs, 0); ng = np.minimum(vs, 0)
        ax.bar(x, p, bottom=bp, color=cc, alpha=0.8, label=cl)
        ax.bar(x, ng, bottom=bn, color=cc, alpha=0.8); bp += p; bn += ng
    ax.set_xticks(x); ax.set_xticklabels([f'{d}' for d in tds])
    ax.set_xlabel('Delay (min)'); ax.set_ylabel('Cost Diff')
    ax.set_title('Gap Breakdown'); ax.legend(fontsize=7)

    ax = axes[1][1]; t_ax = np.arange(n) * base_config.interval_length
    ax2 = ax.twinx(); ax2.fill_between(t_ax, lambdas, alpha=0.08, color=GRAY)
    ax2.set_ylabel(r'$\lambda$', color=GRAY)
    ax.plot(t_ax, last_tr_add, color=TEAL, lw=1.5, label=r'Transient $\mu^+$')
    ax.plot(t_ax, last_ss_add, color=ROSE, lw=1.5, ls='--', label=r'SS $\mu^+$')
    ax.set_xlabel('Time (min)'); ax.set_ylabel(r'$\mu^+$')
    ax.set_title(f'Controls (delay={tds[-1]}min)'); ax.legend(loc='upper left')

    plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'ss_gap.png'), dpi=300); plt.close()
    return results


def _run_ss_opt(lambdas, mus_init, alpha1, alpha2, config,
                max_iter=200, lr=1.0, epsilon=0.1, seed=42,
                device='cpu', dtype=torch.float32):
    """Optimize using per-interval steady-state (power iteration on P)."""
    from model.generator import build_Q_non_erlang_vec, build_P_from_Q, make_state_vectors

    torch.manual_seed(seed); rng = np.random.RandomState(seed); n = len(lambdas)
    p0, ps = config.get_delay_blocks(); KS, KP, M = config.K_S, config.K_P, config.M

    lt = torch.tensor(lambdas, dtype=dtype, device=device)
    m0 = torch.tensor(mus_init, dtype=dtype, device=device)
    a1t = torch.tensor(alpha1, dtype=dtype, device=device)
    a2t = torch.tensor(alpha2, dtype=dtype, device=device)
    sv = make_state_vectors(KS, KP, M, device=device, dtype=dtype)

    ma = torch.nn.Parameter(torch.tensor(rng.uniform(0, 0.1, n), dtype=dtype, device=device))
    mr = torch.nn.Parameter(torch.tensor(rng.uniform(0, 0.05, n), dtype=dtype, device=device))
    opt = torch.optim.Adam([ma, mr], lr=lr)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=15)

    Nn = KP + M + 1; N = (KS+1)*Nn; prev = None
    for step in range(max_iter):
        opt.zero_grad()
        eff = build_eff_nr_zero_pad(m0, ma, mr, p0, ps)
        obj = torch.tensor(0.0, device=device, dtype=dtype)
        for j in range(n):
            p = lt[j]; c = eff[j]; a1 = a1t[j]; a2 = a2t[j]
            ctl = config.fuel_cost + config.time_to_city * a2; dt = config.interval_length
            Q, _, _ = build_Q_non_erlang_vec(K_S=KS, K_P=KP, M=M, lam=c, alpha=p, tau=config.tau, device=device, dtype=dtype)
            P, _ = build_P_from_Q(Q); P = P.coalesce()
            pi = torch.ones(N, dtype=dtype, device=device)/N
            for _ in range(300):
                pn = torch.zeros_like(pi)
                pn.index_add_(0, P.indices()[1], P.values()*pi[P.indices()[0]]); pi = pn
            Ep = torch.dot(sv['w_pass'], pi); Et = torch.dot(sv['w_pick'], pi)
            Er = torch.dot(sv['w_stage'], pi)
            Ebp = torch.dot(sv['w_block_pax'], pi); Ebt = torch.dot(sv['w_block_taxi'], pi)
            obj = obj + (a1*Ep+a2*(Et+Er))*dt + ma[j]*dt*config.cost_per_vehicle_add \
                  + mr[j]*dt*ctl + config.cost_pax_lost*p*Ebp*dt + ctl*c*Ebt*dt
        obj.backward(); opt.step()
        with torch.no_grad():
            ma.data.clamp_(min=0.0); mr.data.clamp_(min=0.0)
            for j in range(n): mr.data[j].clamp_(max=mus_init[j])
            v = obj.item()
            if prev is not None and abs(prev-v) < epsilon: break
            prev = v
        sch.step(v)
    return ma.detach().cpu().numpy(), mr.detach().cpu().numpy()


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sensitivity Analysis')
    parser.add_argument('--n_intervals', type=int, default=None)
    parser.add_argument('--n_seeds', type=int, default=2)
    parser.add_argument('--max_iter', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--analysis', type=str, default='all',
                        choices=['all', 'delay', 'commit', 'cost', 'demand', 'delta', 'ss_gap'])
    parser.add_argument('--out_dir', type=str, default='results/sensitivity')
    args = parser.parse_args()

    config = QueueConfig()
    lambdas, mus_init = load_default_data(config)
    if args.n_intervals:
        lambdas = lambdas[:args.n_intervals]; mus_init = mus_init[:args.n_intervals]
    alpha1, alpha2 = config.get_alpha_arrays(size=len(lambdas))

    print("="*60 + "\nSENSITIVITY ANALYSIS\n" + "="*60)
    print(f"  Intervals: {len(lambdas)}, Seeds: {args.n_seeds}, Analysis: {args.analysis}")
    print("="*60)

    os.makedirs(args.out_dir, exist_ok=True)
    kw = dict(n_seeds=args.n_seeds, max_iter=args.max_iter,
              lr=args.lr, epsilon=args.epsilon, base_seed=args.seed, out_dir=args.out_dir)
    t0 = time.time()

    if args.analysis in ('all', 'delay'):   analysis_delay(lambdas, mus_init, alpha1, alpha2, config, **kw)
    if args.analysis in ('all', 'commit'):  analysis_commit(lambdas, mus_init, alpha1, alpha2, config, **kw)
    if args.analysis in ('all', 'cost'):    analysis_cost(lambdas, mus_init, alpha1, alpha2, config, **kw)
    if args.analysis in ('all', 'demand'):  analysis_demand(lambdas, mus_init, alpha1, alpha2, config, **kw)
    if args.analysis in ('all', 'delta'):   analysis_delta(lambdas, mus_init, alpha1, alpha2, config, **kw)
    if args.analysis in ('all', 'ss_gap'):  analysis_ss_gap(lambdas, mus_init, alpha1, alpha2, config, **kw)

    print(f"\nTotal: {time.time()-t0:.1f}s. Results in {args.out_dir}/")
