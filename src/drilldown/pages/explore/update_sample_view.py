# SPDX-FileCopyrightText: 2026 Manuel Konrad
#
# SPDX-License-Identifier: MIT

"""Callbacks for updating the sample view in the explore page."""

import json
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, callback, ctx
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots

from drilldown.constants import (
    CURVE_FORMATS,
    EMPTY_FIGURE_LAYOUT,
    EXPLORE_PREFIX,
    IMAGE_FORMATS,
)
from drilldown.feature_store import ColumnType, FeatureStore
from drilldown.utils import apply_theme


def _create_values_view(data: list[dict], columns: dict) -> str:
    """Create JSON representation of sample values, excluding embedded curves."""
    excluded_columns = []
    for col in columns.get("curve", []):
        if data and len(data) > 0:
            sample_val = data[0].get(col)
            if isinstance(sample_val, (dict, list)):
                excluded_columns.append(col)

    included_columns = [
        col for col in columns["custom_data"] if col not in excluded_columns
    ]
    filtered_data = [{col: row[col] for col in included_columns} for row in data]
    return json.dumps(filtered_data, indent=2)


def _resolve_sample_data(
    columns: dict,
    selected_rows: list[dict] | None,
    click_data: dict | None,
    existing_data: list[dict] | None,
) -> list[dict] | None:
    """Determine which sample data to display based on the latest user action."""
    triggered_id = ctx.triggered_id
    if (
        triggered_id == f"{EXPLORE_PREFIX}-graph"
        and click_data
        and click_data.get("points")
        and click_data["points"][0].get("customdata")
    ):
        return [
            dict(zip(columns["custom_data"], click_data["points"][0]["customdata"]))
        ]
    if triggered_id == f"{EXPLORE_PREFIX}-ag-grid" and selected_rows:
        return selected_rows
    return existing_data


def _extract_curve_points(
    curve_data: Any,
) -> tuple[str | None, list[float], str | None, list[float]]:
    """Extract x and y arrays from curve data structure."""
    x_label = None
    y_label = None
    x_vals = []
    y_vals = []

    df = pd.DataFrame(curve_data)
    for col in sorted(df.columns):
        if col.lower().startswith("x") and x_label is None:
            x_vals = df[col].tolist()
            match col.lower().split("_"):
                case ["x"] | ["x", ""]:
                    x_label = "x"
                case ["x", name]:
                    x_label = name
                case _:
                    x_label = col
        elif col.lower().startswith("y") and y_label is None:
            y_vals = df[col].tolist()
            match col.lower().split("_"):
                case ["y"] | ["y", ""]:
                    y_label = "y"
                case ["y", name]:
                    y_label = name
                case _:
                    y_label = col
    return x_label, x_vals, y_label, y_vals


def _create_image_figure(
    data: list[dict],
    columns: dict,
    sample_images_cols: list[str] | None,
    dataset,
    primary_key_col: str | None,
    theme: str | None = None,
) -> go.Figure:
    if len(data) > 3:
        data = data[-3:]

    sample_images_cols = sample_images_cols or []
    object_columns = set(columns.get("object", []))

    images: dict[str, list[tuple[Any, str | None]]] = {
        col: [] for col in sample_images_cols
    }

    for col in sample_images_cols:
        if col not in object_columns:
            continue
        for sample in data:
            sample_value = sample.get(col)
            if sample_value is None:
                continue
            column_config = dataset.columns[col]
            col_data = column_config.read_uri_object(sample_value)
            # Check if it's a URI_IMG column or has image format
            if column_config.column_type == ColumnType.URI_IMG or (
                column_config.object_format
                and column_config.object_format in IMAGE_FORMATS
            ):
                pk_value = sample.get(primary_key_col) if primary_key_col else None
                images[col].append((col_data, pk_value))

    rows_with_images = [col for col in sample_images_cols if images.get(col)]
    if not rows_with_images:
        return go.Figure(layout=EMPTY_FIGURE_LAYOUT)

    max_images_per_row = max(len(images[col]) for col in rows_with_images)

    specs: list[list[dict | None]] = []
    title_grid: list[list[str | None]] = []
    for col in rows_with_images:
        col_specs: list[dict | None] = []
        col_titles: list[str | None] = []
        col_images = images[col]
        for idx in range(max_images_per_row):
            if idx < len(col_images):
                pk_value = col_images[idx][1]
                title = f"{col} ({pk_value})" if pk_value is not None else col
                col_specs.append({})
                col_titles.append(title)
            else:
                col_specs.append(None)
                col_titles.append(None)
        specs.append(col_specs)
        title_grid.append(col_titles)

    fig = make_subplots(
        rows=len(rows_with_images),
        cols=max_images_per_row,
        vertical_spacing=0.05,
        horizontal_spacing=0.05,
        specs=specs,
    )

    for row_idx, col in enumerate(rows_with_images, start=1):
        for col_idx, (img_data, _) in enumerate(images[col], start=1):
            fig.add_trace(px.imshow(img_data).data[0], row=row_idx, col=col_idx)
            fig.update_xaxes(showticklabels=False, row=row_idx, col=col_idx)
            fig.update_yaxes(showticklabels=False, row=row_idx, col=col_idx)
            title = title_grid[row_idx - 1][col_idx - 1] or ""
            if title:
                fig.update_xaxes(
                    title_text=title,
                    title_font={"size": 12},
                    row=row_idx,
                    col=col_idx,
                )

    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
    return apply_theme(fig, theme)


def _create_curve_figure(
    data: list[dict],
    columns: dict,
    sample_curves_cols: list[str] | None,
    dataset,
    primary_key_col: str | None,
    overlay: bool = False,
    theme: str | None = None,
) -> go.Figure:
    sample_curves_cols = sample_curves_cols or []
    object_columns = set(columns.get("object", []))

    curve_rows: list[tuple[str, list[tuple[Any, str | None]]]] = []
    curve_titles: list[str] = []

    for col in sample_curves_cols:
        traces: list[tuple[Any, str | None]] = []
        column_config = dataset.columns.get(col)
        if not column_config:
            continue
        for sample in data:
            sample_value = sample.get(col)
            if sample_value is None:
                continue
            pk_value = sample.get(primary_key_col) if primary_key_col else None
            # Embedded curve already structured
            if column_config.column_type == ColumnType.CURVE:
                traces.append((sample_value, pk_value))
                continue
            # URI curve: load from storage
            if column_config.column_type == ColumnType.URI_CURVE:
                loaded = column_config.read_uri_object(sample_value)
                traces.append((loaded, pk_value))
                continue
            # Object column with curve format
            if col in object_columns and (
                column_config.object_format
                and column_config.object_format in CURVE_FORMATS
            ):
                loaded = column_config.read_uri_object(sample_value)
                traces.append((loaded, pk_value))
        if traces:
            curve_rows.append((col, traces))
            curve_titles.append(col)

    if not curve_rows:
        return go.Figure(layout=EMPTY_FIGURE_LAYOUT)

    # Create color mapping for each unique sample (primary key)
    # to ensure same row has same color across all subplots
    # Use dict to maintain insertion order while ensuring uniqueness

    # Generate colors for each unique primary key using Plotly's default color sequence
    color_sequence = px.colors.qualitative.Plotly

    # Track which primary keys have already been shown in the legend
    shown_in_legend: set[str | None] = set()

    # If overlay mode is enabled, put all curves in a single subplot
    if overlay:
        fig = go.Figure()
        col_color_map = {
            title: color_sequence[i % len(color_sequence)]
            for i, title in enumerate(curve_titles)
        }
        for curve_col, curve_list in curve_rows:
            for curve_idx, (curve_data, pk_value) in enumerate(curve_list):
                x_label, x_vals, y_label, y_vals = _extract_curve_points(curve_data)
                if not x_vals or not y_vals:
                    continue

                # Use primary key for legend label, not curve column name
                legend_label = (
                    str(curve_col)
                    if curve_col is not None
                    else f"Curve {curve_idx + 1}"
                )

                # Get consistent color for this primary key
                color = col_color_map.get(
                    curve_col, color_sequence[curve_idx % len(color_sequence)]
                )

                # Only show in legend if this is the first occurrence of this primary key
                show_legend = curve_col not in shown_in_legend
                if show_legend:
                    shown_in_legend.add(curve_col)

                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        mode="lines+markers",
                        name=legend_label,
                        showlegend=show_legend,
                        line=dict(color=color),
                        marker=dict(color=color),
                        legendgroup=str(curve_col)
                        if curve_col is not None
                        else f"Curve {curve_idx + 1}",
                        hovertemplate=f"(%{{x}}, %{{y}})<extra>{pk_value if pk_value is not None else ''}</extra>",
                    )
                )
    else:
        # Create separate subplots for each curve column (default behavior)
        unique_pk_dict: dict[Any, None] = {}
        for _, curve_list in curve_rows:
            for _, pk_value in curve_list:
                if pk_value not in unique_pk_dict:
                    unique_pk_dict[pk_value] = None
        unique_pk_values = list(unique_pk_dict.keys())
        pk_color_map = {
            pk: color_sequence[i % len(color_sequence)]
            for i, pk in enumerate(unique_pk_values)
        }

        fig = make_subplots(
            rows=len(curve_rows),
            cols=1,
            subplot_titles=curve_titles,
            vertical_spacing=0.1,
        )
        for idx, (curve_col, curve_list) in enumerate(curve_rows, start=1):
            for curve_idx, (curve_data, pk_value) in enumerate(curve_list):
                x_label, x_vals, y_label, y_vals = _extract_curve_points(curve_data)
                if not x_vals or not y_vals:
                    continue

                # Use primary key for legend label, not curve column name
                legend_label = (
                    str(pk_value) if pk_value is not None else f"Sample {curve_idx + 1}"
                )

                # Get consistent color for this primary key
                color = pk_color_map.get(
                    pk_value, color_sequence[curve_idx % len(color_sequence)]
                )

                # Only show in legend if this is the first occurrence of this primary key
                show_legend = pk_value not in shown_in_legend
                if show_legend:
                    shown_in_legend.add(pk_value)

                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        mode="lines+markers",
                        name=legend_label,
                        showlegend=show_legend,
                        line=dict(color=color),
                        marker=dict(color=color),
                        legendgroup=str(pk_value)
                        if pk_value is not None
                        else f"Sample {curve_idx + 1}",
                        hovertemplate=f"(%{{x}}, %{{y}})<extra>{pk_value if pk_value is not None else ''}</extra>",
                    ),
                    row=idx,
                    col=1,
                )
                # Set x and y axis labels for this subplot
                if x_label:
                    fig.update_xaxes(title_text=x_label, row=idx, col=1)
                if y_label:
                    fig.update_yaxes(title_text=y_label, row=idx, col=1)
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
    return apply_theme(fig, theme)


def _create_sample_view(
    data: list[dict],
    columns: dict,
    sample_images_cols: list[str] | None,
    sample_curves_cols: list[str] | None,
    dataset,
    primary_key_col: str | None,
    curves_overlay: bool = False,
    theme: str | None = None,
) -> tuple[go.Figure, go.Figure]:
    """Create separate figures for images and curves."""

    image_fig = _create_image_figure(
        data,
        columns,
        sample_images_cols,
        dataset,
        primary_key_col,
        theme,
    )
    curve_fig = _create_curve_figure(
        data,
        columns,
        sample_curves_cols,
        dataset,
        primary_key_col,
        curves_overlay,
        theme,
    )
    return image_fig, curve_fig


@callback(
    Output(f"{EXPLORE_PREFIX}-sample-values-code", "code"),
    Output(f"{EXPLORE_PREFIX}-sample-images-subplot", "figure"),
    Output(f"{EXPLORE_PREFIX}-sample-curves-subplot", "figure"),
    Output(f"{EXPLORE_PREFIX}-sample-view-store", "data"),
    Input(f"{EXPLORE_PREFIX}-store", "data"),
    Input(f"{EXPLORE_PREFIX}-ag-grid", "selectedRows"),
    Input(f"{EXPLORE_PREFIX}-graph", "clickData"),
    Input(f"{EXPLORE_PREFIX}-sample-images-select", "value"),
    Input(f"{EXPLORE_PREFIX}-sample-curves-select", "value"),
    Input(f"{EXPLORE_PREFIX}-sample-curves-overlay-switch", "checked"),
    Input("theme-toggle", "value"),
    State(f"{EXPLORE_PREFIX}-sample-view-store", "data"),
    prevent_initial_call=True,
)
def update_sample_view(
    explore_store: dict | None,
    selected_rows: list[dict] | None,
    click_data: dict | None,
    sample_images_cols: list[str] | None,
    sample_curves_cols: list[str] | None,
    curves_overlay: bool,
    theme: str | None,
    existing_data: list[dict] | None,
) -> tuple[str, go.Figure, go.Figure, list[dict]]:
    if not explore_store:
        raise PreventUpdate

    columns_config = explore_store["columns"]
    primary_key_col = columns_config[0]
    columns = columns_config[2]
    feature_store = FeatureStore.model_validate_json(explore_store["feature_store"])
    collection_name = explore_store["collection"]
    dataset_name = explore_store["dataset"]
    dataset = feature_store.collections[collection_name][dataset_name]

    data = _resolve_sample_data(columns, selected_rows, click_data, existing_data)
    if data is None:
        raise PreventUpdate
    # update sample view
    image_fig, curve_fig = _create_sample_view(
        data,
        columns,
        sample_images_cols,
        sample_curves_cols,
        dataset,
        primary_key_col,
        curves_overlay,
        theme,
    )

    # update values view
    values_view = _create_values_view(data, columns)

    return values_view, image_fig, curve_fig, data
