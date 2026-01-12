# SPDX-FileCopyrightText: 2026 Manuel Konrad
#
# SPDX-License-Identifier: MIT

"""Monitoring drift detection algorithms and utilities."""

import numpy as np
import pandas as pd
from scipy import stats

from drilldown.constants import MIN_WINDOW_SAMPLES


def compute_ks_statistic(
    reference: np.ndarray, current: np.ndarray
) -> tuple[float, float]:
    """Compute the Kolmogorov-Smirnov statistic for drift detection."""

    statistic, p_value = stats.ks_2samp(reference, current)
    return float(statistic), float(p_value)


def compute_rolling_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    timestamp_col: str,
    value_col: str,
    rolling_window: int,
    step_days: int = 1,
) -> pd.DataFrame:
    """Compute drift metrics over rolling windows using KS statistic."""
    # Prepare reference data
    ref_df = reference_df.copy()
    ref_df[timestamp_col] = pd.to_datetime(ref_df[timestamp_col])
    reference_data = ref_df[value_col].dropna().values

    if len(reference_data) == 0:
        return pd.DataFrame()

    # Prepare current data
    cur_df = current_df.copy()
    cur_df[timestamp_col] = pd.to_datetime(cur_df[timestamp_col])
    cur_df = cur_df.sort_values(by=timestamp_col)

    if len(cur_df) == 0:
        return pd.DataFrame()

    cur_df = cur_df.set_index(timestamp_col)

    # Calculate drift for each rolling window
    results = []
    min_date = cur_df.index.min()
    max_date = cur_df.index.max()

    current_date = min_date
    while current_date <= max_date:
        window_start = current_date - pd.Timedelta(days=rolling_window)
        window_end = current_date

        # Extract data for current window
        window_data = cur_df.loc[
            (cur_df.index >= window_start) & (cur_df.index <= window_end),
            value_col,
        ].dropna()

        # Calculate drift if we have sufficient data
        if len(window_data) >= MIN_WINDOW_SAMPLES:
            window_values = window_data.values

            ks_stat, p_value = compute_ks_statistic(reference_data, window_values)
            results.append(
                {
                    "timestamp": current_date,
                    "drift_score": ks_stat,
                    "p_value": p_value,
                    "window_mean": float(np.mean(window_values)),
                    "window_std": float(np.std(window_values)),
                    "n_samples": len(window_values),
                }
            )

        current_date += pd.Timedelta(days=step_days)

    return pd.DataFrame(results)
