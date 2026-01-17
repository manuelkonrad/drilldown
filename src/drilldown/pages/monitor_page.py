# SPDX-FileCopyrightText: 2026 Manuel Konrad
#
# SPDX-License-Identifier: MIT

"""Monitor page layout and callbacks."""

import datetime

import dash
import dash_mantine_components as dmc
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback, clientside_callback, dcc, get_app
from dash.exceptions import PreventUpdate

from drilldown.constants import (
    DEFAULT_REFERENCE_DAYS_END,
    DEFAULT_REFERENCE_DAYS_START,
    DEFAULT_ROLLING_WINDOW,
    DEFAULT_STEP_DAYS,
    EMPTY_FIGURE_LAYOUT,
    GRAPH_CONFIG,
    GRAPH_STYLE,
    MONITOR_PREFIX,
    PAGE_CONTAINER_HEIGHT,
    SELECT_LABEL_STYLE,
    SELECT_MAX_DROPDOWN_HEIGHT,
)
from drilldown.feature_store import FeatureStore
from drilldown.pages.monitor.algorithms import create_monitor_figure
from drilldown.utils import create_figure_tabs


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
