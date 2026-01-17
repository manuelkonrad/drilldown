# SPDX-FileCopyrightText: 2026 Manuel Konrad
#
# SPDX-License-Identifier: MIT

"""Monitoring drift detection algorithms and utilities."""

import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from drilldown.constants import (
    MIN_ROLLING_PERIODS,
    MIN_WINDOW_SAMPLES,
)
from drilldown.utils import apply_theme


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


def _add_time_series_traces(
    fig: go.Figure,
    dim_df: pd.DataFrame,
    dim: str,
    timestamp_col: str,
    rolling_window: int,
    row: int,
    show_legend: bool,
) -> None:
    """Add time series traces including data points and rolling statistics."""
    fig.add_trace(
        go.Scatter(
            x=dim_df[timestamp_col],
            y=dim_df[dim],
            mode="markers",
            name="Data",
            marker=dict(size=4, opacity=0.5, color="#636EFA"),
            showlegend=show_legend,
            legendgroup="data",
        ),
        row=row,
        col=1,
    )

    dim_df_indexed = dim_df.set_index(timestamp_col)

    rolling_mean = (
        dim_df_indexed[dim]
        .rolling(f"{rolling_window}D", min_periods=MIN_ROLLING_PERIODS)
        .mean()
    )
    rolling_std = (
        dim_df_indexed[dim]
        .rolling(f"{rolling_window}D", min_periods=MIN_ROLLING_PERIODS)
        .std()
    )

    fig.add_trace(
        go.Scatter(
            x=rolling_mean.index,
            y=rolling_mean.values,
            mode="lines",
            name="Rolling Mean",
            line=dict(color="#00CC96", width=2),
            showlegend=show_legend,
            legendgroup="rolling_mean",
        ),
        row=row,
        col=1,
    )

    upper_rolling = rolling_mean + rolling_std
    lower_rolling = rolling_mean - rolling_std

    # Add upper bound trace
    fig.add_trace(
        go.Scatter(
            x=rolling_mean.index,
            y=upper_rolling,
            mode="lines",
            line=dict(color="rgba(0, 204, 150, 0.3)", width=0),
            showlegend=False,
            legendgroup="rolling_std_band",
            hoverinfo="skip",
            connectgaps=True,
        ),
        row=row,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=rolling_mean.index,
            y=lower_rolling,
            mode="lines",
            line=dict(color="rgba(0, 204, 150, 0.3)", width=0),
            fill="tonexty",
            fillcolor="rgba(0, 204, 150, 0.15)",
            name="Rolling ±1 std",
            showlegend=show_legend,
            legendgroup="rolling_std_band",
            connectgaps=True,
        ),
        row=row,
        col=1,
    )


def _add_reference_traces(
    fig: go.Figure,
    dim_df: pd.DataFrame,
    timestamp_col: str,
    ref_mean: float,
    ref_std: float,
    reference_start: datetime.datetime,
    reference_end: datetime.datetime,
    current_time_start: datetime.datetime,
    current_time_end: datetime.datetime,
    row: int,
    show_legend: bool,
) -> None:
    """Add reference period visualizations to the figure."""
    fig.add_trace(
        go.Scatter(
            x=[current_time_start, current_time_end],
            y=[ref_mean, ref_mean],
            mode="lines",
            name="Reference Mean",
            line=dict(color="#AB63FA", width=2, dash="dot"),
            showlegend=show_legend,
            legendgroup="ref_mean",
        ),
        row=row,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=[
                current_time_start,
                current_time_end,
                current_time_end,
                current_time_start,
            ],
            y=[
                ref_mean + ref_std,
                ref_mean + ref_std,
                ref_mean - ref_std,
                ref_mean - ref_std,
            ],
            fill="toself",
            fillcolor="rgba(171, 99, 250, 0.1)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Reference ±1 std",
            showlegend=show_legend,
            legendgroup="ref_std_band",
        ),
        row=row,
        col=1,
    )


def _add_drift_score_traces(
    fig: go.Figure,
    drift_df: pd.DataFrame,
    drift_thresholds: list,
    drift_type_label: str,
    row: int,
    show_legend: bool,
) -> None:
    """Add drift score bar chart with threshold lines to the figure."""
    if len(drift_df) == 0:
        return

    colors = []
    for score in drift_df["drift_score"]:
        color = "green"
        for threshold, _, c in drift_thresholds:
            if score <= threshold:
                color = c
                break
        colors.append(color)

    fig.add_trace(
        go.Bar(
            x=drift_df["timestamp"],
            y=drift_df["drift_score"],
            name=drift_type_label,
            marker=dict(color=colors),
            showlegend=False,
        ),
        row=row,
        col=1,
    )

    for threshold, label, color in drift_thresholds:
        fig.add_hline(
            y=threshold,
            line=dict(color=color, dash="dash", width=1),
            row=row,
            col=1,
        )


def create_monitor_figure(
    df: pd.DataFrame,
    ref_df: pd.DataFrame,
    timestamp_col: str,
    dimension: str,
    reference_start: datetime.datetime,
    reference_end: datetime.datetime,
    rolling_window: int,
    step_days: int = 1,
    theme: str | None = None,
) -> go.Figure:
    """Create monitoring figure for a single dimension with time series and drift metrics."""
    fig = make_subplots(
        rows=2,
        cols=1,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
    )

    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values(by=timestamp_col)

    ref_df = ref_df.copy()
    ref_df[timestamp_col] = pd.to_datetime(ref_df[timestamp_col])
    ref_df = ref_df.sort_values(by=timestamp_col)

    drift_type_label = "Drift Score"
    drift_thresholds = [
        (0.1, "Low", "green"),
        (0.2, "Medium", "orange"),
        (1.0, "High", "red"),
    ]

    dim_df = df[[timestamp_col, dimension]].dropna()
    ref_dim_df = ref_df[[timestamp_col, dimension]].dropna()

    if len(dim_df) == 0:
        return go.Figure()

    if len(ref_dim_df) > 0:
        ref_mean = float(np.mean(ref_dim_df[dimension]))
        ref_std = float(np.std(ref_dim_df[dimension]))
    else:
        ref_mean = float(np.mean(dim_df[dimension]))
        ref_std = float(np.std(dim_df[dimension]))

    current_time_start = dim_df[timestamp_col].min()
    current_time_end = dim_df[timestamp_col].max()

    # --- Time Series Plot ---
    _add_time_series_traces(
        fig=fig,
        dim_df=dim_df,
        dim=dimension,
        timestamp_col=timestamp_col,
        rolling_window=rolling_window,
        row=1,
        show_legend=True,
    )

    # --- Reference Period Visualization ---
    _add_reference_traces(
        fig=fig,
        dim_df=dim_df,
        timestamp_col=timestamp_col,
        ref_mean=ref_mean,
        ref_std=ref_std,
        reference_start=reference_start,
        reference_end=reference_end,
        current_time_start=current_time_start,
        current_time_end=current_time_end,
        row=1,
        show_legend=True,
    )

    # --- Drift Score Plot ---
    # Pass reference and current data separately - do not combine
    drift_df = compute_rolling_drift(
        reference_df=ref_dim_df,
        current_df=dim_df,
        timestamp_col=timestamp_col,
        value_col=dimension,
        rolling_window=rolling_window,
        step_days=step_days,
    )

    _add_drift_score_traces(
        fig=fig,
        drift_df=drift_df,
        drift_thresholds=drift_thresholds,
        drift_type_label=drift_type_label,
        row=2,
        show_legend=True,
    )

    # Update y-axis labels
    fig.update_yaxes(title_text=dimension, row=1, col=1)
    fig.update_yaxes(
        title_text=drift_type_label,
        row=2,
        col=1,
    )

    # Update layout
    fig.update_layout(
        height=600,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=50, r=50, t=80, b=50),
    )

    return apply_theme(fig, theme)
