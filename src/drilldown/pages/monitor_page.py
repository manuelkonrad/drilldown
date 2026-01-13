# SPDX-FileCopyrightText: 2026 Manuel Konrad
#
# SPDX-License-Identifier: MIT

"""Monitor page layout and callbacks."""

import datetime

import dash
import dash_mantine_components as dmc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback, clientside_callback, dcc, get_app
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots

from drilldown.constants import (
    DEFAULT_REFERENCE_DAYS_END,
    DEFAULT_REFERENCE_DAYS_START,
    DEFAULT_ROLLING_WINDOW,
    DEFAULT_STEP_DAYS,
    EMPTY_FIGURE_LAYOUT,
    GRAPH_CONFIG,
    GRAPH_STYLE,
    MIN_ROLLING_PERIODS,
    MONITOR_PREFIX,
    PAGE_CONTAINER_HEIGHT,
    SELECT_LABEL_STYLE,
    SELECT_MAX_DROPDOWN_HEIGHT,
)
from drilldown.feature_store import FeatureStore
from drilldown.pages.monitor.algorithms import compute_rolling_drift
from drilldown.utils import apply_theme, create_figure_tabs


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


@callback(
    Output(f"{MONITOR_PREFIX}-content-container", "children"),
    Output(f"{MONITOR_PREFIX}-dimensions-select", "data"),
    Output(f"{MONITOR_PREFIX}-refresh-store", "data"),
    Input(f"{MONITOR_PREFIX}-dimensions-select", "value"),
    Input(f"{MONITOR_PREFIX}-rolling-window", "value"),
    Input(f"{MONITOR_PREFIX}-step-days", "value"),
    Input(f"{MONITOR_PREFIX}-reference-date-picker", "value"),
    Input("main-store", "data"),
    Input("theme-toggle", "value"),
    prevent_initial_call=True,
)
def update_monitor(
    dimensions: list[str] | None,
    rolling_window: int | None,
    step_days: int | None,
    reference_date_range: list[str] | None,
    main_store: dict[str, dict] | None,
    theme: str | None,
) -> tuple[dmc.Tabs | dcc.Graph, list[dict], dict]:
    """Single callback to update monitor page - loads data, creates figures, and renders content."""
    empty_graph = dcc.Graph(
        figure=go.Figure(layout=EMPTY_FIGURE_LAYOUT),
        style=GRAPH_STYLE,
        config=GRAPH_CONFIG,
    )
    if not main_store or not main_store.get("data"):
        return empty_graph, [], {}

    if reference_date_range is None or any([d is None for d in reference_date_range]):
        raise PreventUpdate

    # Extract data from main_store
    df = pd.DataFrame(main_store["data"])
    columns = main_store["columns"]
    col_types = columns[2]

    # Build column lists - only numerical columns for monitoring
    numerical_columns = col_types.get("numerical", [])
    timestamp_column = columns[1]

    # Set default values
    if rolling_window is None:
        rolling_window = DEFAULT_ROLLING_WINDOW
    if step_days is None:
        step_days = DEFAULT_STEP_DAYS

    # Build dimension options - only numerical columns
    dimension_options = [{"value": col, "label": col} for col in numerical_columns]

    # Load reference data
    ref_df = pd.DataFrame()
    reference_start = pd.to_datetime(
        datetime.datetime.now() - datetime.timedelta(days=DEFAULT_REFERENCE_DAYS_START)
    )
    reference_end = pd.to_datetime(
        datetime.datetime.now() - datetime.timedelta(days=DEFAULT_REFERENCE_DAYS_END)
    )

    if (
        main_store.get("feature_store")
        and reference_date_range
        and len(reference_date_range) == 2
    ):
        # Reconstruct feature store
        feature_store = FeatureStore.model_validate_json(main_store["feature_store"])
        collection = main_store.get("collection")
        dataset = main_store.get("dataset")

        if collection and dataset:
            if (
                collection in feature_store.collections
                and dataset in feature_store.collections[collection]
            ):
                # Load reference data for the specified date range
                ref_start = datetime.datetime.fromisoformat(
                    str(reference_date_range[0])
                )
                ref_end = datetime.datetime.fromisoformat(str(reference_date_range[1]))

                ref_data, _ = feature_store.collections[collection][
                    dataset
                ].get_dataframe_date_range(
                    start=ref_start,
                    end=ref_end,
                    partitions=None,
                )
                ref_df = ref_data
                reference_start = pd.to_datetime(reference_date_range[0])
                reference_end = pd.to_datetime(reference_date_range[1])

    # Create figures for each dimension (for tabs)
    figures_dict = {}
    dimensions = dimensions or []
    for dim in dimensions:
        fig = create_monitor_figure(
            df=df,
            ref_df=ref_df,
            timestamp_col=timestamp_column,
            dimension=dim,
            reference_start=reference_start,
            reference_end=reference_end,
            rolling_window=rolling_window,
            step_days=step_days,
            theme=theme,
        )
        figures_dict[dim] = fig

    # Render the content
    if not figures_dict:
        return empty_graph, dimension_options, {}

    content = create_figure_tabs(figures_dict)
    return content, dimension_options, {}


def monitor_container(config):
    return dmc.Flex(
        [
            dmc.Flex(
                [
                    dmc.Grid(
                        [
                            dmc.GridCol(
                                dmc.Select(
                                    placeholder="Select monitor type",
                                    label="Monitor type",
                                    id=f"{MONITOR_PREFIX}-select",
                                    styles=SELECT_LABEL_STYLE,
                                    value="ks",
                                    data=[
                                        {
                                            "value": "ks",
                                            "label": "Kolmogorov-Smirnov",
                                        },
                                    ],
                                    w="100%",
                                    clearable=False,
                                    allowDeselect=False,
                                    persistence=True,
                                    persistence_type="session",
                                ),
                                span="auto",
                            ),
                            dmc.GridCol(
                                dmc.DatePickerInput(
                                    id=f"{MONITOR_PREFIX}-reference-date-picker",
                                    label="Reference range",
                                    styles=SELECT_LABEL_STYLE,
                                    type="range",
                                    value=[
                                        (
                                            datetime.datetime.now()
                                            - datetime.timedelta(
                                                days=DEFAULT_REFERENCE_DAYS_START
                                            )
                                        ).date(),
                                        (
                                            datetime.datetime.now()
                                            - datetime.timedelta(
                                                days=DEFAULT_REFERENCE_DAYS_END
                                            )
                                        ).date(),
                                    ],
                                    maw=300,
                                    miw=200,
                                    placeholder="Select reference date range",
                                    allowSingleDateInRange=False,
                                    clearable=False,
                                    persistence=True,
                                    persistence_type="session",
                                ),
                                span="content",
                            ),
                            dmc.GridCol(
                                dmc.NumberInput(
                                    id=f"{MONITOR_PREFIX}-rolling-window",
                                    label="Window size (days)",
                                    styles=SELECT_LABEL_STYLE,
                                    value=DEFAULT_ROLLING_WINDOW,
                                    min=1,
                                    max=30,
                                    step=1,
                                    w=150,
                                ),
                                span="content",
                            ),
                            dmc.GridCol(
                                dmc.NumberInput(
                                    id=f"{MONITOR_PREFIX}-step-days",
                                    label="Step size (days)",
                                    styles=SELECT_LABEL_STYLE,
                                    value=DEFAULT_STEP_DAYS,
                                    min=1,
                                    max=30,
                                    step=1,
                                    w=150,
                                ),
                                span="content",
                            ),
                        ],
                        w="100%",
                        gutter="xs",
                        overflow="hidden",
                    ),
                ],
                align="flex-end",
                gap="xs",
                pb="xs",
            ),
            dmc.MultiSelect(
                placeholder="Select dimensions and press enter",
                clearable=True,
                searchable=True,
                id=f"{MONITOR_PREFIX}-dimensions-select",
                pb="xs",
                persistence=True,
                persistence_type="session",
                persisted_props=["value", "data"],
                debounce=True,
                maxDropdownHeight=SELECT_MAX_DROPDOWN_HEIGHT,
            ),
            dcc.Loading(
                dmc.Box(
                    id=f"{MONITOR_PREFIX}-content-container",
                    style={
                        "height": "100%",
                        "width": "100%",
                    },
                ),
                parent_style={
                    "height": "100%",
                    "width": "100%",
                },
                type="dot",
            ),
            dcc.Store(id=f"{MONITOR_PREFIX}-refresh-store", data={}),
        ],
        style={
            "height": "100%",
            "width": "100%",
        },
        direction="column",
    )


clientside_callback(
    """
    (data) => {
        window.dispatchEvent(new Event('resize'));
        return null;
    }
    """,
    Input(f"{MONITOR_PREFIX}-refresh-store", "data"),
)


def layout(**kwargs):
    app = get_app()
    config = app.drilldown_config
    return dmc.Flex(
        [
            monitor_container(config=config),
        ],
        style={"height": PAGE_CONTAINER_HEIGHT},
        direction="column",
        p="xs",
    )


dash.register_page(
    __name__,
    path="/monitor/",
    title="Drift Monitoring",
    description="Perform statistical tests on rolling windows to detect data drift and change points.",
    icon="material-symbols:monitoring",
    layout=layout,
    redirect_from=["/monitor"],
    order=3,
)
