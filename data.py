"""
Data loading for BLR Airport demand profiles.

Supports two dataset formats:
  1. NEW: blr_demand_profile.csv (single file, T1+T2 columns)
  2. OLD: Total_Passengers_Arrival.csv + Total_Passengers_departures.csv

CSV values are taxi-passenger rates per minute, already accounting for
load factor and taxi share (TAXI_SHARE=0.33).

pax_per_taxi divides BOTH lambda and mu_d: if 1.5 passengers share
each taxi on average, fewer taxis are needed for pickup AND fewer
taxis arrive from drop-off.
"""

import os
import numpy as np
import pandas as pd


def load_demand_profile(csv_path=None, terminal='T1', config=None):
    """
    Load taxi demand (lambda) and taxi supply (mu_d) rates from new format.

    Parameters
    ----------
    csv_path : str or None
        Path to blr_demand_profile.csv. If None, searches default locations.
    terminal : str, 'T1' or 'T2'
    config : QueueConfig or None
        Applies data_scale_factor and pax_per_taxi to BOTH rates.

    Returns
    -------
    lambdas : np.ndarray, taxi demand rates (per minute)
    mus_init : np.ndarray, taxi drop-off rates (per minute)
    """
    if csv_path is None:
        candidates = [
            'Datasets/blr_demand_profile.csv',
            'blr_demand_profile.csv',
            os.path.join(os.path.dirname(__file__), 'Datasets', 'blr_demand_profile.csv'),
        ]
        for c in candidates:
            if os.path.exists(c):
                csv_path = c
                break
        if csv_path is None:
            raise FileNotFoundError(
                f"Cannot find blr_demand_profile.csv. Searched: {candidates}")

    df = pd.read_csv(csv_path)

    lambda_col = f'lambda_{terminal}'
    mu_col = f'mu_d_{terminal}'

    if lambda_col not in df.columns:
        raise ValueError(f"Column '{lambda_col}' not found. Available: {list(df.columns)}")
    if mu_col not in df.columns:
        raise ValueError(f"Column '{mu_col}' not found. Available: {list(df.columns)}")

    lambdas = df[lambda_col].values.astype(np.float64)
    mus_init = df[mu_col].values.astype(np.float64)

    if config is not None:
        scale = getattr(config, 'data_scale_factor', 1.0)
        if scale != 1.0:
            lambdas = lambdas / scale
            mus_init = mus_init / scale

        # Group travel: both rates are derived from passenger counts
        # lambda = arriving pax / pax_per_taxi → taxis needed for pickup
        # mu_d = departing pax / pax_per_taxi → taxis arriving from drop-off
        ppt = getattr(config, 'pax_per_taxi', 1.0)
        if ppt != 1.0:
            lambdas = lambdas / ppt
            mus_init = mus_init / ppt

    return lambdas, mus_init


def load_old_format(arr_path=None, dep_path=None, config=None):
    """
    Load from old two-file format (Total_Passengers_Arrival/departures.csv).

    Each CSV has a 'Total_Passengers' column with per-minute counts.
    Data is aggregated into intervals of config.group_size minutes.

    Parameters
    ----------
    arr_path : str or None
    dep_path : str or None
    config : QueueConfig or None

    Returns
    -------
    lambdas : np.ndarray, taxi demand rates (per minute)
    mus_init : np.ndarray, taxi drop-off rates (per minute)
    """
    if arr_path is None:
        candidates = [
            'Datasets/old/Total_Passengers_Arrival.csv',
            'Datasets/Total_Passengers_Arrival.csv',
            os.path.join(os.path.dirname(__file__), 'Datasets', 'old', 'Total_Passengers_Arrival.csv'),
            os.path.join(os.path.dirname(__file__), 'Datasets', 'Total_Passengers_Arrival.csv'),
        ]
        for c in candidates:
            if os.path.exists(c):
                arr_path = c
                break
        if arr_path is None:
            raise FileNotFoundError("Cannot find Total_Passengers_Arrival.csv")

    if dep_path is None:
        dep_dir = os.path.dirname(arr_path)
        dep_path = os.path.join(dep_dir, 'Total_Passengers_departures.csv')

    arr_df = pd.read_csv(arr_path)
    dep_df = pd.read_csv(dep_path)

    group_size = config.group_size if config else 5

    arr_agg = aggregate_passengers(arr_df, group_size)
    dep_agg = aggregate_passengers(dep_df, group_size)

    lambdas = arr_agg['total_rate'].values.astype(np.float64)
    mus_init = dep_agg['total_rate'].values.astype(np.float64)

    # Ensure same length
    n = min(len(lambdas), len(mus_init))
    lambdas = lambdas[:n]
    mus_init = mus_init[:n]

    if config is not None:
        scale = getattr(config, 'data_scale_factor', 1.0)
        if scale != 1.0:
            lambdas = lambdas / scale
            mus_init = mus_init / scale

        ppt = getattr(config, 'pax_per_taxi', 1.0)
        if ppt != 1.0:
            lambdas = lambdas / ppt
            mus_init = mus_init / ppt

    return lambdas, mus_init


def load_default_data(config):
    """
    Load demand data based on config.dataset setting.

    config.dataset:
      'new' or 'blr_demand_profile' → new single-file format (default)
      'old' or 'legacy'             → old two-file format

    Returns (lambdas, mus_init).
    """
    dataset = getattr(config, 'dataset', 'new')

    if dataset in ('old', 'legacy'):
        return load_old_format(config=config)
    else:
        return load_demand_profile(terminal='T1', config=config)


# ──────────────────────────────────────────────────────────────
# Helper: aggregate minute-level data into intervals
# ──────────────────────────────────────────────────────────────

def aggregate_passengers(df, group_size):
    """
    Aggregate minute-level data into larger intervals.

    Parameters
    ----------
    df : DataFrame with 'Total_Passengers' column (per-minute counts)
    group_size : int, number of minutes per interval

    Returns
    -------
    DataFrame with 'total_rate' column (rate per minute within each interval)
    """
    n = len(df)
    n_groups = n // group_size
    rates = []
    for i in range(n_groups):
        chunk = df.iloc[i*group_size:(i+1)*group_size]
        total = chunk['Total_Passengers'].sum()
        rate = total / group_size
        rates.append({'group': i, 'total_rate': rate})
    return pd.DataFrame(rates)
