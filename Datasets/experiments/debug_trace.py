"""
Debug Script: Full-day trace with per-interval diagnostics.

Runs full-day Adam from (0,0), then traces every interval showing:
  - lambda, mu_0, mu_add, mu_remove, eff_nr
  - Objective contribution
  - Top 5 states in pi
  - E[s], E[n], blocking probabilities
  - Transfer rate analysis

Also runs system diagnostic to check capacity/bottleneck.

Usage:
    python experiments/debug_trace.py
    python experiments/debug_trace.py --n_intervals 20 --max_iter 100
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import time

from config import QueueConfig
from data import load_default_data
from model.generator import build_Q_non_erlang_vec, build_P_from_Q, make_state_vectors
from model.simulation import uniformized_with_checkpoint_blocks
from optimizer_utils import (
    build_eff_nr_zero_pad, optimize_full_day, make_pi0,
    get_distribution_stats,
)


def system_diagnostic(lambdas, mus_init, config):
    """Print comprehensive system diagnostic."""
    print("=" * 70)
    print("SYSTEM DIAGNOSTIC")
    print("=" * 70)

    print(f"\n  STATE SPACE:")
    print(f"    K_S = {config.K_S} (max taxis in reserve)")
    print(f"    K_P = {config.K_P} (max taxis in pickup)")
    print(f"    M   = {config.M} (max passengers waiting)")
    print(f"    States = {(config.K_S+1)*(config.K_P+config.M+1)}")

    print(f"\n  PARAMETERS:")
    print(f"    tau (reserve->pickup rate) = {config.tau}/min")
    print(f"    interval_length = {config.interval_length} min")
    print(f"    data_scale_factor = {config.data_scale_factor}")
    print(f"    delay_non_reserved = {config.delay_non_reserved} min")
    print(f"    delay_extra = {config.delay_extra} min")
    pad_mu0, pad_mus = config.get_delay_blocks()
    print(f"    pad_mu0 = {pad_mu0}, pad_mus = {pad_mus}")

    print(f"\n  COSTS:")
    print(f"    cost_per_vehicle_add = {config.cost_per_vehicle_add}")
    print(f"    cost_pax_lost = {config.cost_pax_lost}")
    print(f"    fuel_cost = {config.fuel_cost}")
    print(f"    time_to_city = {config.time_to_city}")

    alpha1, alpha2 = config.get_alpha_arrays(size=len(lambdas))
    print(f"    alpha1 (pax VoT) = {alpha1[0]:.4f}/min")
    print(f"    alpha2 range = [{alpha2.min():.4f}, {alpha2.max():.4f}]/min")

    print(f"\n  DEMAND vs SUPPLY:")
    dt = config.interval_length
    n = len(lambdas)
    print(f"    Passenger arrival (lambda):")
    print(f"      mean = {lambdas.mean():.4f}/min")
    print(f"      max  = {lambdas.max():.4f}/min (at interval {np.argmax(lambdas)})")
    print(f"      min  = {lambdas.min():.4f}/min")
    print(f"      total/day = {lambdas.sum() * dt:.0f} passengers")

    print(f"    Taxi drop-off (mu_0):")
    print(f"      mean = {mus_init.mean():.4f}/min")
    print(f"      max  = {mus_init.max():.4f}/min")
    print(f"      min  = {mus_init.min():.4f}/min")
    print(f"      total/day = {mus_init.sum() * dt:.0f} taxis")

    gap = lambdas.mean() - mus_init.mean()
    print(f"    Mean gap (lambda - mu_0) = {gap:+.4f}/min")
    print(f"      {'DEMAND > SUPPLY' if gap > 0 else 'SUPPLY > DEMAND'}")

    print(f"\n  BOTTLENECK ANALYSIS:")
    print(f"    Transfer model: batch transfer at rate tau={config.tau}/min")
    print(f"    Each event moves k=min(s, K_P-n) taxis from reserve to pickup")
    print(f"    Effective flow when s taxis in reserve: tau * s = {config.tau} * s")
    print(f"")
    s_needed = lambdas.mean() / config.tau
    print(f"    To match mean demand ({lambdas.mean():.4f}/min):")
    print(f"      Need E[s] >= lambda/tau = {s_needed:.1f} taxis in reserve")
    print(f"      K_S = {config.K_S} {'(sufficient)' if config.K_S > s_needed else '(INSUFFICIENT!)'}")
    print(f"")
    s_peak = lambdas.max() / config.tau
    print(f"    To match peak demand ({lambdas.max():.4f}/min):")
    print(f"      Need E[s] >= {s_peak:.1f} taxis in reserve")
    print(f"      K_S = {config.K_S} {'(sufficient)' if config.K_S > s_peak else '(INSUFFICIENT!)'}")

    print(f"\n  DISPATCH ECONOMICS:")
    print(f"    Cost to add 1 taxi: {config.cost_per_vehicle_add}")
    print(f"    Benefit: saves ~{dt:.0f} min of pax wait * alpha1 = {alpha1[0]*dt:.2f}")
    print(f"    Cost/benefit ratio: {config.cost_per_vehicle_add / (alpha1[0]*dt):.1f}")
    print(f"    (>1 means dispatch is expensive relative to waiting cost)")


def trace_full_day(lambdas, mus_init, alpha1, alpha2, mu_add, mu_remove,
                   config, device='cpu', dtype=torch.float32,
                   print_every=1, top_k=5):
    """Trace through every interval with full diagnostics."""

    n = len(lambdas)
    pad_mu0, pad_mus = config.get_delay_blocks()
    K_S, K_P, M = config.K_S, config.K_P, config.M
    Nn = K_P + M + 1
    dt = config.interval_length

    # Build eff_nr
    eff_nr = build_eff_nr_zero_pad(mus_init, mu_add, mu_remove, pad_mu0, pad_mus)

    # State vectors
    sv = make_state_vectors(K_S, K_P, M, device=device, dtype=dtype)
    W = torch.stack([sv['w_pass'], sv['w_stage'], sv['w_pick'],
                     sv['w_block_pax'], sv['w_block_taxi']], dim=0)

    pi = make_pi0(config, device, dtype)

    print(f"\n{'='*90}")
    print(f"PER-INTERVAL TRACE")
    print(f"{'='*90}")
    print(f"  {'Int':>4} {'Time':>6} {'λ':>8} {'μ₀':>8} {'μ⁺':>8} {'μ⁻':>8} "
          f"{'eff_μ':>8} {'Cost':>10} {'E[s]':>8} {'E[n]':>8} "
          f"{'P(block_p)':>10} {'Top State':>20}")
    print(f"  {'-'*110}")

    total_obj = 0.0
    interval_costs = []

    for j in range(n):
        pax = float(lambdas[j])
        cars = float(eff_nr[j])
        a1, a2 = float(alpha1[j]), float(alpha2[j])
        ctl = config.fuel_cost + config.time_to_city * a2

        Q, _, _ = build_Q_non_erlang_vec(
            K_S=K_S, K_P=K_P, M=M,
            lam=cars, alpha=pax, tau=config.tau,
            device=device, dtype=dtype)
        P, gamma = build_P_from_Q(Q)
        P = P.coalesce()

        A_pass, A_resv, A_taxi, A_block_pax, A_block_taxi, pi_T = \
            uniformized_with_checkpoint_blocks(
                pi, P.indices()[0], P.indices()[1], P.values(), gamma, W,
                config.interval_length, max_K_cap=30000, tol_tail=1e-12, block_size=60)

        ap = A_pass.item(); ar = A_resv.item(); at = A_taxi.item()
        abp = A_block_pax.item(); abt = A_block_taxi.item()

        c_pax = a1 * ap
        c_taxi = a2 * (at + ar)
        c_add = mu_add[j] * dt * config.cost_per_vehicle_add
        c_rem = mu_remove[j] * dt * ctl
        c_plost = config.cost_pax_lost * pax * abp
        c_tlost = ctl * cars * abt
        interval_cost = c_pax + c_taxi + c_add + c_rem + c_plost + c_tlost
        total_obj += interval_cost
        interval_costs.append(interval_cost)

        # Expected state
        E_s = torch.dot(sv['s_vec'], pi_T).item()
        E_n = torch.dot(sv['n_vec'], pi_T).item()

        # Top k states
        topk = torch.topk(pi_T, k=min(top_k, len(pi_T)))
        top_state_str = f"(s={topk.indices[0].item()//Nn}," \
                        f"n={topk.indices[0].item()%Nn - M}):" \
                        f"{topk.values[0].item():.4f}"

        # Blocking probability
        # P(block_pax) = sum of pi where n = -M
        p_block_pax = 0.0
        for s in range(K_S + 1):
            idx = s * Nn + 0  # n = -M is at offset 0
            p_block_pax += pi_T[idx].item()

        if j % print_every == 0 or j == n - 1:
            t_min = j * dt
            print(f"  {j:>4} {t_min:>5.0f}m {pax:>8.4f} {mus_init[j]:>8.4f} "
                  f"{mu_add[j]:>8.4f} {mu_remove[j]:>8.4f} {cars:>8.4f} "
                  f"{interval_cost:>10.1f} {E_s:>8.2f} {E_n:>8.2f} "
                  f"{p_block_pax:>10.6f} {top_state_str:>20}")

            if j % (print_every * 5) == 0 and top_k > 1:
                # Print all top states
                for ki in range(min(top_k, 3)):
                    idx = topk.indices[ki].item()
                    s_val = idx // Nn
                    n_val = (idx % Nn) - M
                    prob = topk.values[ki].item()
                    if ki > 0:
                        print(f"  {'':>100} (s={s_val},n={n_val}):{prob:.4f}")

                # Print cost breakdown
                print(f"  {'':>60} costs: pax_wait={c_pax:.1f}, "
                      f"taxi_idle={c_taxi:.1f}, add={c_add:.1f}, "
                      f"block_pax={c_plost:.1f}")

        pi = pi_T

    print(f"  {'-'*110}")
    print(f"  TOTAL OBJECTIVE: {total_obj:.2f}")

    # Summary by block
    block_size = max(1, n // 8)
    print(f"\n  COST BY BLOCK ({block_size} intervals each):")
    print(f"  {'Block':>6} {'Intervals':>12} {'Cost':>12} {'Mean λ':>10} "
          f"{'Mean μ₀':>10} {'Mean μ⁺':>10} {'Mean eff_μ':>10}")
    print(f"  {'-'*72}")
    for b in range(0, n, block_size):
        be = min(b + block_size, n)
        bc = sum(interval_costs[b:be])
        print(f"  {b//block_size:>6} {b:>5}-{be-1:<5} {bc:>12.1f} "
              f"{lambdas[b:be].mean():>10.4f} {mus_init[b:be].mean():>10.4f} "
              f"{mu_add[b:be].mean():>10.4f} {eff_nr[b:be].mean():>10.4f}")

    # Transfer analysis
    print(f"\n  TRANSFER BOTTLENECK:")
    print(f"    Final E[s] = {E_s:.2f}")
    print(f"    Expected transfer flow = tau * E[s] = {config.tau} * {E_s:.2f} = {config.tau * E_s:.4f}/min")
    print(f"    Mean passenger demand = {lambdas.mean():.4f}/min")
    ratio = config.tau * E_s / lambdas.mean() if lambdas.mean() > 0 else 0
    print(f"    Ratio (supply/demand) = {ratio:.4f} {'(OK)' if ratio > 0.9 else '(UNDERSUPPLIED!)'}")

    return total_obj, interval_costs


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Debug Trace')
    parser.add_argument('--n_intervals', type=int, default=None)
    parser.add_argument('--max_iter', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--epsilon', type=float, default=1.0)
    parser.add_argument('--print_every', type=int, default=1,
                        help='Print every N intervals')
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--do_nothing', action='store_true',
                        help='Trace do-nothing (no optimization)')
    parser.add_argument('--control_dir', type=str, default=None,
                        help='Load controls from directory')
    args = parser.parse_args()

    config = QueueConfig()
    lambdas, mus_init = load_default_data(config)
    if args.n_intervals:
        lambdas = lambdas[:args.n_intervals]
        mus_init = mus_init[:args.n_intervals]
    alpha1, alpha2 = config.get_alpha_arrays(size=len(lambdas))
    n = len(lambdas)

    # Diagnostic
    system_diagnostic(lambdas, mus_init, config)

    # Get controls
    if args.do_nothing:
        print(f"\n{'='*70}")
        print(f"DO-NOTHING TRACE")
        print(f"{'='*70}")
        mu_add = np.zeros(n)
        mu_remove = np.zeros(n)
    elif args.control_dir:
        print(f"\n{'='*70}")
        print(f"TRACE WITH CONTROLS FROM {args.control_dir}")
        print(f"{'='*70}")
        mu_add = np.load(os.path.join(args.control_dir, 'mu_add.npy'))[:n]
        mu_remove = np.load(os.path.join(args.control_dir, 'mu_remove.npy'))[:n]
    else:
        print(f"\n{'='*70}")
        print(f"OPTIMIZING (max_iter={args.max_iter}, lr={args.lr}, eps={args.epsilon})")
        print(f"{'='*70}")
        t0 = time.time()
        result = optimize_full_day(
            lambdas, mus_init, alpha1, alpha2, config,
            max_iter=args.max_iter, lr=args.lr, epsilon=args.epsilon, seed=42)
        print(f"  Objective: {result['objective']:.2f} ({time.time()-t0:.1f}s)")
        mu_add = result['mu_add']
        mu_remove = result['mu_remove']

    # Control summary
    print(f"\n  CONTROL SUMMARY:")
    print(f"    sum(mu_add)    = {mu_add.sum():.4f}")
    print(f"    sum(mu_remove) = {mu_remove.sum():.4f}")
    print(f"    net add        = {mu_add.sum() - mu_remove.sum():+.4f}")
    print(f"    mu_add range   = [{mu_add.min():.4f}, {mu_add.max():.4f}]")
    print(f"    mu_remove range= [{mu_remove.min():.4f}, {mu_remove.max():.4f}]")

    # Check last intervals (should be near zero with zero-pad)
    pad_mu0, pad_mus = config.get_delay_blocks()
    print(f"\n  LAST {pad_mus} INTERVALS (should be ~0 with zero-pad):")
    for j in range(max(0, n - pad_mus), n):
        print(f"    [{j}] mu_add={mu_add[j]:.6f}, mu_remove={mu_remove[j]:.6f}")

    # Trace
    trace_full_day(
        lambdas, mus_init, alpha1, alpha2, mu_add, mu_remove,
        config, print_every=args.print_every, top_k=args.top_k)

    # Also trace do-nothing for comparison
    if not args.do_nothing and not args.control_dir:
        print(f"\n\n{'='*70}")
        print(f"DO-NOTHING COMPARISON TRACE")
        print(f"{'='*70}")
        trace_full_day(
            lambdas, mus_init, alpha1, alpha2,
            np.zeros(n), np.zeros(n),
            config, print_every=max(1, n // 20), top_k=3)


if __name__ == '__main__':
    main()
