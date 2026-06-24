"""
run_method.py — run ONE optimizer method (per-seed for greedy/MPC) and persist results.

Splits the monolithic comparison so expensive methods (especially MPC, whose every
window re-optimizes all remaining intervals) can be run independently, incrementally,
and across machines. Each greedy/MPC sample is saved on its own with wall-clock time
and host, so times can be averaged later. Statistics + plotting live in plot_results.py.

On-disk layout (under --out_dir):
    run_config.json                       shared inputs (must match across runs/machines)
    full_day/    mu_add.npy mu_remove.npy meta.json
    do_nothing/  mu_add.npy mu_remove.npy meta.json
    greedy/seed_<S>/  mu_add.npy mu_remove.npy meta.json
    mpc/seed_<S>/     mu_add.npy mu_remove.npy meta.json

Examples:
    # full-day + do-nothing (single runs; you may already have these):
    python experiments/run_method.py --method full_day   --initial_state_dir results/initial_state
    python experiments/run_method.py --method do_nothing --initial_state_dir results/initial_state
    # MPC, one seed at a time, on any machine:
    python experiments/run_method.py --method mpc --seeds 42 --sample_state --initial_state_dir results/initial_state
    python experiments/run_method.py --method mpc --seeds 43 44 --sample_state --initial_state_dir results/initial_state
    # greedy, several seeds at once:
    python experiments/run_method.py --method greedy --seeds 42 43 44 45 46 --sample_state
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse, json, time, socket, platform
import numpy as np
import torch

from config import QueueConfig
from data import load_default_data
from optimizer_utils import (
    optimize_full_day, optimize_greedy, optimize_mpc,
    run_do_nothing, load_initial_state, evaluate_full_day,
)


# ──────────────────────────────────────────────────────────────
# INPUTS
# ──────────────────────────────────────────────────────────────

def build_inputs(config, n_intervals):
    """Rebuild the exact same inputs every run (deterministic)."""
    lambdas, mus_init = load_default_data(config)
    if n_intervals is not None:
        lambdas = lambdas[:n_intervals]
        mus_init = mus_init[:n_intervals]
    alpha1, alpha2 = config.get_alpha_arrays(size=len(lambdas))
    return lambdas, mus_init, alpha1, alpha2


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return str(o)


def write_or_check_run_config(out_dir, cfg):
    """Write run_config.json once; on later runs warn if inputs disagree.

    Comparability depends on identical inputs across every run/machine — this is
    the guard that flags a mismatched --commit / --n_intervals / initial state.
    """
    path = os.path.join(out_dir, 'run_config.json')
    if os.path.exists(path):
        with open(path) as f:
            existing = json.load(f)
        mism = {k: (existing.get(k), cfg[k]) for k in cfg
                if k != 'carryover_cost' and existing.get(k) != cfg[k]}
        if mism:
            print("  WARNING: run_config.json disagrees with current args —")
            for k, (old, new) in mism.items():
                print(f"    {k}: existing={old!r}  now={new!r}")
            print("  Runs must share identical inputs to be comparable. Not overwriting.")
    else:
        with open(path, 'w') as f:
            json.dump(cfg, f, indent=2, default=_json_default)


def save_run(subdir, r, elapsed, seed, method, extra=None):
    """Persist one run's controls + meta (objective, time, host)."""
    os.makedirs(subdir, exist_ok=True)
    np.save(os.path.join(subdir, 'mu_add.npy'), np.asarray(r['mu_add']))
    np.save(os.path.join(subdir, 'mu_remove.npy'), np.asarray(r['mu_remove']))
    meta = {
        'method': method,
        'seed': seed,
        'objective': float(r['objective']),
        'time_seconds': float(elapsed),
        'sampled_states': r.get('sampled_states', []),
        'hostname': socket.gethostname(),
        'platform': platform.platform(),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    if extra:
        meta.update(extra)
    with open(os.path.join(subdir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2, default=_json_default)
    return meta


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Run one optimizer method (per-seed for greedy/mpc); persist results + timing.')
    p.add_argument('--method', required=True,
                   choices=['full_day', 'do_nothing', 'greedy', 'mpc'])
    p.add_argument('--seeds', type=int, nargs='+', default=[42],
                   help='Seeds for greedy/mpc (one sample dir each). '
                        'full_day uses seeds[0]; do_nothing ignores seeds.')
    p.add_argument('--n_intervals', type=int, default=None)
    p.add_argument('--commit', type=int, default=36)
    p.add_argument('--buffer', type=int, default=None,
                   help='Greedy lookahead buffer (default: pad_mus)')
    p.add_argument('--max_iter', type=int, default=500)
    p.add_argument('--lr', type=float, default=1.0)
    p.add_argument('--epsilon', type=float, default=10)
    p.add_argument('--sample_state', action='store_true')
    p.add_argument('--pi0', type=str, default=None,
                   help='Path to pi0.npy (distribution only; no carryover)')
    p.add_argument('--initial_state_dir', type=str, default=None,
                   help='Initial-state dir (loads pi0 + carryover + carryover_cost)')
    p.add_argument('--import_full_day_from', type=str, default=None,
                   help='For --method full_day: load mu_add.npy/mu_remove.npy from this '
                        'dir (e.g. results/model_analysis/adam_delay) and RE-EVALUATE them '
                        'under the current pi0/carryover instead of re-optimizing. Valid only '
                        'when those controls were optimized from the same initial condition.')
    p.add_argument('--out_dir', type=str, default='results/comparison')
    args = p.parse_args()

    config = QueueConfig()
    lambdas, mus_init, alpha1, alpha2 = build_inputs(config, args.n_intervals)
    device = 'cpu'
    dtype = getattr(config, 'dtype_torch', torch.float32)

    # Initial state
    pi0 = None; carryover = None; co_cost = 0.0
    if args.initial_state_dir:
        pi0, carryover = load_initial_state(args.initial_state_dir, config, device, dtype)
        co_path = os.path.join(args.initial_state_dir, 'carryover_cost.npy')
        if os.path.exists(co_path):
            co_cost = float(np.load(co_path)[0])
    elif args.pi0:
        pi0 = args.pi0  # resolve_pi0 (inside optimizers) handles string paths

    os.makedirs(args.out_dir, exist_ok=True)
    write_or_check_run_config(args.out_dir, {
        'n_intervals': args.n_intervals,
        'commit': args.commit,
        'buffer': args.buffer,
        'sample_state': bool(args.sample_state),
        'max_iter': args.max_iter,
        'lr': args.lr,
        'epsilon': args.epsilon,
        'initial_state': args.initial_state_dir or args.pi0 or '(0,0) default',
        'carryover_cost': co_cost,
    })

    print("=" * 60)
    print(f"RUN METHOD: {args.method}")
    print(f"  intervals={len(lambdas)} commit={args.commit} "
          f"sample_state={args.sample_state}")
    print(f"  initial_state={args.initial_state_dir or args.pi0 or '(0,0) default'} "
          f"co_cost={co_cost:.2f}")
    print(f"  host={socket.gethostname()}  seeds={args.seeds}")
    print("=" * 60)

    # Common kwargs for the Adam-based optimizers (NOT do_nothing)
    common = dict(max_iter=args.max_iter, lr=args.lr, epsilon=args.epsilon,
                  pi0=pi0, carryover=carryover, device=device, dtype=dtype)

    if args.method == 'do_nothing':
        t0 = time.time()
        r = run_do_nothing(lambdas, mus_init, alpha1, alpha2, config,
                           pi0=pi0, device=device, dtype=dtype)
        dt = time.time() - t0
        save_run(os.path.join(args.out_dir, 'do_nothing'), r, dt, None, 'do_nothing')
        print(f"  do_nothing: obj={r['objective']:.2f} ({dt:.1f}s)")

    elif args.method == 'full_day':
        if args.import_full_day_from:
            src = args.import_full_day_from
            mu_add = np.load(os.path.join(src, 'mu_add.npy'))
            mu_remove = np.load(os.path.join(src, 'mu_remove.npy'))
            if len(mu_add) != len(lambdas):
                print(f"  WARNING: imported controls length {len(mu_add)} != "
                      f"n_intervals {len(lambdas)}; slicing to match.")
                mu_add = mu_add[:len(lambdas)]
                mu_remove = mu_remove[:len(lambdas)]
            # Re-score through evaluate_full_day — the SAME path greedy/MPC use —
            # under the current pi0/carryover, so the benchmark is apples-to-apples.
            t0 = time.time()
            total_obj, _, _ = evaluate_full_day(
                mu_add, mu_remove, lambdas, mus_init, alpha1, alpha2, config,
                pi0=pi0, carryover=carryover, device=device, dtype=dtype)
            dt = time.time() - t0
            r = {'mu_add': mu_add, 'mu_remove': mu_remove, 'objective': float(total_obj)}
            save_run(os.path.join(args.out_dir, 'full_day'), r, dt, None, 'full_day',
                     extra={'imported_from': src,
                            'note': 'controls imported and re-evaluated; '
                                    'time_seconds is eval-only, not optimization time'})
            print(f"  full_day (imported from {src}): obj={total_obj:.2f}  [re-eval {dt:.1f}s]")
        else:
            t0 = time.time()
            r = optimize_full_day(lambdas, mus_init, alpha1, alpha2, config,
                                  seed=args.seeds[0], **common)
            dt = time.time() - t0
            save_run(os.path.join(args.out_dir, 'full_day'), r, dt, args.seeds[0], 'full_day')
            print(f"  full_day: obj={r['objective']:.2f} ({dt:.1f}s)")

    else:  # greedy or mpc — one sample dir per seed, each timed
        for seed in args.seeds:
            t0 = time.time()
            if args.method == 'greedy':
                r = optimize_greedy(lambdas, mus_init, alpha1, alpha2, config,
                                   commit_size=args.commit, buffer_size=args.buffer,
                                   seed=seed, sample_state=args.sample_state, **common)
            else:
                r = optimize_mpc(lambdas, mus_init, alpha1, alpha2, config,
                                commit_size=args.commit,
                                seed=seed, sample_state=args.sample_state, **common)
            dt = time.time() - t0
            subdir = os.path.join(args.out_dir, args.method, f'seed_{seed}')
            save_run(subdir, r, dt, seed, args.method)
            print(f"  {args.method} seed={seed}: obj={r['objective']:.2f} "
                  f"({dt:.1f}s)  -> {subdir}")

    print("\nDone.")