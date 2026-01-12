# SPDX-FileCopyrightText: 2026 Manuel Konrad
#
# SPDX-License-Identifier: MIT

"""Analyze page layout and callbacks."""

import dash
import dash_mantine_components as dmc
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback, clientside_callback, dcc, get_app

from drilldown.constants import (
    ANALYZE_PREFIX,
    EMPTY_FIGURE_LAYOUT,
    GRAPH_CONFIG,
    GRAPH_STYLE,
    PAGE_CONTAINER_HEIGHT,
    SELECT_LABEL_STYLE,
    SELECT_MAX_DROPDOWN_HEIGHT,
)
from drilldown.pages.analyze.algorithms import (
    ANALYZE_TYPES,
    compute_correlation_analysis,
    compute_ebm_analysis,
    compute_ebm_local_explanation,
    compute_feature_importance,
    compute_what_if_analysis,
    create_correlation_figure,
    create_ebm_global_figure,
    create_ebm_local_figure,
    create_feature_importance_figure,
    create_what_if_figure,
)
from drilldown.utils import create_figure_tabs


def _render_figures_data(figures_data: dict) -> dmc.Box | dcc.Graph:
    """Helper function to render figures data as Dash components."""
    if not figures_data or not figures_data.get("type"):
        return dcc.Graph(
            figure=go.Figure(layout=EMPTY_FIGURE_LAYOUT),
            style=GRAPH_STYLE,
            config=GRAPH_CONFIG,
        )

    if figures_data["type"] == "single":
        # Render single graph
        return dcc.Graph(
            figure=figures_data.get("figure", {}),
            style=GRAPH_STYLE,
            config=GRAPH_CONFIG,
        )
    elif figures_data["type"] == "tabs":
        # Render tabs with multiple graphs
        figures = figures_data.get("figures", {})
        if not figures:
            return dcc.Graph(
                figure=go.Figure(layout=EMPTY_FIGURE_LAYOUT),
                style=GRAPH_STYLE,
                config=GRAPH_CONFIG,
            )

        # Use create_figure_tabs utility with summary as default
        return create_figure_tabs(figures, default_tab="Summary")

    return dcc.Graph(
        figure=go.Figure(layout=EMPTY_FIGURE_LAYOUT),
        style=GRAPH_STYLE,
        config=GRAPH_CONFIG,
    )


@callback(
    Output(f"{ANALYZE_PREFIX}-content-container", "children"),
    Output(f"{ANALYZE_PREFIX}-figures-store", "data"),
    Output(f"{ANALYZE_PREFIX}-dimensions-select", "data"),
    Output(f"{ANALYZE_PREFIX}-target-picker", "data"),
    Output(f"{ANALYZE_PREFIX}-results-store", "data"),
    Output(f"{ANALYZE_PREFIX}-target-filter-select", "data"),
    Output(f"{ANALYZE_PREFIX}-target-filter-select", "disabled"),
    Output(f"{ANALYZE_PREFIX}-target-samples-select", "data"),
    Output(f"{ANALYZE_PREFIX}-target-samples-select", "disabled"),
    Input(f"{ANALYZE_PREFIX}-select", "value"),
    Input(f"{ANALYZE_PREFIX}-dimensions-select", "value"),
    Input(f"{ANALYZE_PREFIX}-target-picker", "value"),
    Input(f"{ANALYZE_PREFIX}-target-filter-select", "value"),
    Input(f"{ANALYZE_PREFIX}-target-samples-select", "value"),
    Input("theme-toggle", "value"),
    Input("main-store", "data"),
)
def update_analyze_graph(
    analyze_type: str | None,
    dimensions: list[str] | None,
    target_field: str | None,
    target_filter: str | None,
    target_samples: str | None,
    theme: str | None,
    main_store: dict[str, dict] | None,
) -> tuple:
    """Update analyze graph based on selected analyze type and parameters."""
    is_ebm_local = analyze_type == "ebm_local"

    target_filter_data: list = []
    target_filter_disabled = not is_ebm_local
    target_samples_data: list = []
    target_samples_disabled = not is_ebm_local

    if not main_store or not main_store.get("data"):
        empty_content = _render_figures_data({})
        return (
            empty_content,
            {},
            [],
            [],
            {},
            target_filter_data,
            target_filter_disabled,
            target_samples_data,
            target_samples_disabled,
        )

    df = pd.DataFrame(main_store["data"])
    columns = main_store["columns"]
    col_types = columns[2]

    numerical_columns = col_types.get("numerical", [])
    categorical_columns = col_types.get("categorical", [])

    # Filter to numeric-compatible categorical columns to avoid SHAP errors
    filtered_categorical_columns = []
    for col in categorical_columns:
        if col in df.columns:
            try:
                pd.to_numeric(df[col], errors="coerce")
                if df[col].dtype in ["bool", "int8", "int16", "int32", "int64"]:
                    filtered_categorical_columns.append(col)
            except (ValueError, TypeError):
                pass

    all_target_columns = numerical_columns + filtered_categorical_columns

    grouped_target_options = []
    if numerical_columns:
        grouped_target_options.append(
            {
                "group": "Numerical Fields",
                "items": [{"value": col, "label": col} for col in numerical_columns],
            }
        )
    if filtered_categorical_columns:
        grouped_target_options.append(
            {
                "group": "Categorical Fields",
                "items": [
                    {"value": col, "label": col} for col in filtered_categorical_columns
                ],
            }
        )

    grouped_dimension_options = []
    if numerical_columns:
        grouped_dimension_options.append(
            {
                "group": "Numerical Fields",
                "items": [{"value": col, "label": col} for col in numerical_columns],
            }
        )
    if filtered_categorical_columns:
        grouped_dimension_options.append(
            {
                "group": "Categorical Fields",
                "items": [
                    {"value": col, "label": col} for col in filtered_categorical_columns
                ],
            }
        )

    if not target_field or not dimensions:
        empty_content = _render_figures_data({})
        return (
            empty_content,
            {},
            grouped_dimension_options,
            grouped_target_options,
            {},
            target_filter_data,
            target_filter_disabled,
            target_samples_data,
            target_samples_disabled,
        )

    if target_field not in all_target_columns:
        empty_content = _render_figures_data({})
        return (
            empty_content,
            {},
            grouped_dimension_options,
            grouped_target_options,
            {},
            target_filter_data,
            target_filter_disabled,
            target_samples_data,
            target_samples_disabled,
        )

    # Filter features to exclude the target variable
    features = [d for d in dimensions if d != target_field]
    if not features:
        empty_content = _render_figures_data({})
        return (
            empty_content,
            {},
            grouped_dimension_options,
            grouped_target_options,
            {},
            target_filter_data,
            target_filter_disabled,
            target_samples_data,
            target_samples_disabled,
        )

    figures_data = {}
    if analyze_type == "feature_importance":
        result = compute_feature_importance(df, target_field, features)
        fig = create_feature_importance_figure(result, theme)
        figures_data = {"type": "single", "figure": fig.to_dict()}
    elif analyze_type == "what_if":
        result = compute_what_if_analysis(df, target_field, features, features)
        figures_result = create_what_if_figure(result, theme)
        if isinstance(figures_result, dict) and figures_result.get("type") == "tabs":
            serialized_figures = {}
            for key, fig in figures_result["figures"].items():
                serialized_figures[key] = fig.to_dict()
            figures_data = {"type": "tabs", "figures": serialized_figures}
    elif analyze_type == "correlation":
        result = compute_correlation_analysis(df, target_field, features)
        fig = create_correlation_figure(result, theme)
        figures_data = {"type": "single", "figure": fig.to_dict()}
    elif analyze_type == "ebm_global":
        result = compute_ebm_analysis(df, target_field, features)
        figures_result = create_ebm_global_figure(result, theme)
        if figures_result.get("type") == "tabs":
            serialized_figures = {}
            for key, fig in figures_result["figures"].items():
                serialized_figures[key] = fig.to_dict()
            figures_data = {"type": "tabs", "figures": serialized_figures}
        else:
            figures_data = {
                "type": "single",
                "figure": figures_result.get("figure", go.Figure().to_dict()),
            }
    elif analyze_type == "ebm_local":
        primary_id_col = columns[0]

        if target_field in df.columns:
            unique_targets = sorted(df[target_field].dropna().unique().tolist())
            target_filter_data = [
                {"value": str(v), "label": str(v)} for v in unique_targets
            ]

        if target_filter and target_field in df.columns and primary_id_col:
            filtered_df = df[df[target_field].astype(str) == target_filter]
            target_samples_data = [
                {
                    "value": str(df.loc[idx, primary_id_col]),
                    "label": str(df.loc[idx, primary_id_col]),
                }
                for idx in filtered_df.index
                if pd.notna(df.loc[idx, primary_id_col])
            ]
        else:
            target_samples_data = []

        if not target_samples:
            empty_content = _render_figures_data({})
            return (
                empty_content,
                {},
                grouped_dimension_options,
                grouped_target_options,
                {},
                target_filter_data,
                target_filter_disabled,
                target_samples_data,
                target_samples_disabled,
            )

        result = compute_ebm_analysis(df, target_field, features)

        if target_samples and primary_id_col:
            try:
                matching_rows = df[
                    df[primary_id_col].astype(str) == target_samples
                ].index.tolist()

                if matching_rows:
                    sample_index = matching_rows[0]
                    local_result = compute_ebm_local_explanation(result, sample_index)
                    sample_id = df.loc[sample_index, primary_id_col]
                    local_result["sample_id"] = sample_id
                    local_result["target_field"] = target_field
                    fig = create_ebm_local_figure(local_result, theme)
                    figures_data = {"type": "single", "figure": fig.to_dict()}
                else:
                    figures_data = {
                        "type": "single",
                        "figure": go.Figure(layout=EMPTY_FIGURE_LAYOUT).to_dict(),
                    }
            except (ValueError, TypeError, IndexError):
                figures_data = {
                    "type": "single",
                    "figure": go.Figure(layout=EMPTY_FIGURE_LAYOUT).to_dict(),
                }
        else:
            figures_data = {
                "type": "single",
                "figure": go.Figure(layout=EMPTY_FIGURE_LAYOUT).to_dict(),
            }
    else:
        result = {}
        figures_data = {
            "type": "single",
            "figure": go.Figure(layout=EMPTY_FIGURE_LAYOUT).to_dict(),
        }

    rendered_content = _render_figures_data(figures_data)

    return (
        rendered_content,
        figures_data,
        grouped_dimension_options,
        grouped_target_options,
        result,
        target_filter_data,
        target_filter_disabled,
        target_samples_data,
        target_samples_disabled,
    )


def analyze_container(config):
    """Create the analyze page container layout."""
    return dmc.Flex(
        [
            dmc.Flex(
                [
                    dmc.Select(
                        placeholder="Select analysis type",
                        label="Analysis type",
                        styles=SELECT_LABEL_STYLE,
                        id=f"{ANALYZE_PREFIX}-select",
                        value="correlation",
                        data=ANALYZE_TYPES,
                        w="100%",
                        persistence=True,
                        persistence_type="session",
                        clearable=False,
                        allowDeselect=False,
                        maxDropdownHeight=SELECT_MAX_DROPDOWN_HEIGHT,
                    ),
                    dmc.Select(
                        placeholder="Select target field",
                        label="Target field",
                        styles=SELECT_LABEL_STYLE,
                        id=f"{ANALYZE_PREFIX}-target-picker",
                        w="100%",
                        searchable=True,
                        persistence=True,
                        persistence_type="session",
                        persisted_props=["value", "data"],
                        maxDropdownHeight=SELECT_MAX_DROPDOWN_HEIGHT,
                    ),
                    dmc.Select(
                        placeholder="Select target filter",
                        label="Target filter",
                        styles=SELECT_LABEL_STYLE,
                        id=f"{ANALYZE_PREFIX}-target-filter-select",
                        w="100%",
                        searchable=True,
                        clearable=True,
                        data=[],
                        disabled=True,
                        maxDropdownHeight=SELECT_MAX_DROPDOWN_HEIGHT,
                    ),
                    dmc.Select(
                        placeholder="Select target samples",
                        label="Target samples",
                        styles=SELECT_LABEL_STYLE,
                        id=f"{ANALYZE_PREFIX}-target-samples-select",
                        w="100%",
                        searchable=True,
                        clearable=True,
                        data=[],
                        disabled=True,
                        maxDropdownHeight=SELECT_MAX_DROPDOWN_HEIGHT,
                    ),
                ],
                justify="flex-start",
                align="center",
                gap="xs",
                w="100%",
                pb="xs",
            ),
            dmc.MultiSelect(
                placeholder="Select dimensions and press enter",
                clearable=True,
                searchable=True,
                id=f"{ANALYZE_PREFIX}-dimensions-select",
                pb="xs",
                persistence=True,
                persistence_type="session",
                persisted_props=["value", "data"],
                debounce=True,
                maxDropdownHeight=SELECT_MAX_DROPDOWN_HEIGHT,
            ),
            dcc.Loading(
                # This div will contain either a single graph or tabs
                dmc.Box(
                    id=f"{ANALYZE_PREFIX}-content-container",
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
            dcc.Store(id=f"{ANALYZE_PREFIX}-results-store", data={}),
            dcc.Store(id=f"{ANALYZE_PREFIX}-figures-store", data={}),
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
    Input(f"{ANALYZE_PREFIX}-figures-store", "data"),
)


def layout(**kwargs):
    app = get_app()
    config = app.drilldown_config

    # Analysis controls and visualization (full width, no grid)
    main_column = analyze_container(config=config)

    return dmc.Flex(
        [main_column],
        style={"height": PAGE_CONTAINER_HEIGHT},
        direction="column",
        p="xs",
    )


dash.register_page(
    __name__,
    path="/analyze/",
    title="Root Cause Analysis",
    description="Find correlations and causal relationships using advanced analysis methods.",
    icon="material-symbols:graph-1",
    layout=layout,
    redirect_from=["/analyze"],
    order=2,
)
