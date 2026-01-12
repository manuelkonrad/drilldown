# SPDX-FileCopyrightText: 2026 Manuel Konrad
#
# SPDX-License-Identifier: MIT

"""Dash callbacks for the drilldown application."""

import datetime
from typing import Any
from urllib.parse import quote, unquote

import dash_mantine_components as dmc
import pandas as pd
from dash import (
    ALL,
    Input,
    Output,
    State,
    callback,
    callback_context,
    clientside_callback,
    page_registry,
)
from dash.exceptions import PreventUpdate

from drilldown.constants import (
    SELECT_LABEL_STYLE,
    SIDEBAR_WIDTH_CLOSED,
    SIDEBAR_WIDTH_OPEN,
)
from drilldown.feature_store import FeatureStore

clientside_callback(
    """
    (val) => {
        document.documentElement.setAttribute(
            'data-mantine-color-scheme',
            val
        );
        return window.dash_clientside.no_update;
    }
    """,
    Output("theme-toggle", "id"),
    Input("theme-toggle", "value"),
)


@callback(
    Output("clipboard", "content"),
    Input("clipboard", "n_clicks"),
    State("dataset-select", "value"),
    State("collection-select", "value"),
    State("location", "href"),
    prevent_initial_call=True,
)
def update_query_params(n_clicks, dataset, collection, href):
    """Generate a shareable URL with the current collection and dataset selection."""
    if dataset is not None and collection is not None:
        return (
            f"{href}?collection={quote(collection, safe='')}"
            f"&dataset={quote(dataset, safe='')}"
        )
    elif collection is not None:
        return f"{href}?collection={quote(collection, safe='')}"
    else:
        return ""


@callback(
    Output("dataset-select", "value"),
    Output("collection-select", "value"),
    Output("location", "search"),
    Input("location", "pathname"),
    State("location", "search"),
    State("dataset-select", "value"),
    State("collection-select", "value"),
    prevent_initial_call=True,
)
def set_pickers_from_url(href, search, dataset, collection):
    """Parse URL query string to set dataset and collection picker values."""
    if "dataset=" in search:
        dataset = unquote(search.split("dataset=")[-1].split("&")[0])
    if "collection=" in search:
        collection = unquote(search.split("collection=")[-1].split("&")[0])
    return dataset, collection, search


@callback(
    Output("main-store", "data"),
    Output("collection-select", "data"),
    Output("dataset-select", "data"),
    Output("header-filters-container", "children"),
    Input("date-picker", "value"),
    Input("collection-select", "value"),
    Input("dataset-select", "value"),
    Input("header-filters-popover", "opened"),
    State({"type": "header-filter-multiselect", "index": ALL}, "value"),
    State({"type": "header-filter-multiselect", "index": ALL}, "id"),
    State("main-store", "data"),
)
def load_data(
    date_range,
    collection,
    dataset,
    popover_opened,
    filter_values,
    filter_ids,
    main_store,
):
    """Load data from the feature store based on selected filters."""
    if date_range[1] is None or popover_opened:
        raise PreventUpdate

    if main_store:
        feature_store = FeatureStore.model_validate_json(main_store["feature_store"])
    else:
        config = callback_context.custom_data["drilldown_config"]
        feature_store = FeatureStore(collection_paths=config["collection_paths"])

    collection_options = [
        {"value": item, "label": item}
        for item in sorted(list(feature_store.collections.keys()))
    ]
    if collection and collection in feature_store.collections:
        dataset_list = sorted(list(feature_store.collections[collection].keys()))
        dataset_options = [{"value": item, "label": item} for item in dataset_list]
    else:
        dataset_options = []

    if (
        collection in feature_store.collections
        and dataset in feature_store.collections[collection]
    ):
        try:
            raw_data, partition_options = feature_store.collections[collection][
                dataset
            ].get_dataframe_date_range(
                start=datetime.datetime.fromisoformat(str(date_range[0])),
                end=datetime.datetime.fromisoformat(str(date_range[1])),
                partitions=None,
            )
            columns = feature_store.collections[collection][
                dataset
            ].get_column_names_by_type()
            info = None

            categorical_columns = columns[2].get("categorical", [])

            filter_selections = {}
            if filter_values and filter_ids:
                for fid, val in zip(filter_ids, filter_values):
                    col_name = fid["index"]
                    if val:
                        filter_selections[col_name] = val

            filtered_data = raw_data.copy()
            if filter_selections:
                for col, selected_values in filter_selections.items():
                    if col in filtered_data.columns and selected_values:
                        filtered_data = filtered_data[
                            filtered_data[col].astype(str).isin(selected_values)
                        ]

            data = filtered_data.to_dict(orient="records")

            filter_children = _create_filter_multiselects(
                categorical_columns, raw_data, filter_selections
            )

        except Exception:
            data = None
            columns = None
            info = "Error loading dataset."
            filter_children = [dmc.Text("Error loading filters.", size="sm")]
    else:
        data = None
        dataset = None
        collection = None
        columns = None
        info = "No dataset selected."
        filter_children = [
            dmc.Text("Select a dataset to see filters.", size="sm", mt="xs"),
        ]

    return (
        {
            "data": data,
            "dataset": dataset,
            "collection": collection,
            "columns": columns,
            "feature_store": feature_store.to_json(),
            "info": info,
        },
        collection_options,
        dataset_options,
        filter_children,
    )


def _create_filter_multiselects(
    categorical_columns: list[str],
    df: pd.DataFrame | None,
    filter_selections: dict[str, list[str]] | None,
) -> list[Any]:
    """Create multi-select components for categorical columns."""
    if not categorical_columns:
        return [dmc.Text("No categorical columns available.", size="sm")]

    children: list[Any] = []
    for col in categorical_columns:
        if df is not None and col in df.columns:
            unique_values = sorted(df[col].dropna().unique().astype(str).tolist())
        else:
            unique_values = []

        options = [{"value": v, "label": v} for v in unique_values]
        current_value = filter_selections.get(col, []) if filter_selections else []

        children.append(
            dmc.MultiSelect(
                id={"type": "header-filter-multiselect", "index": col},
                label=col,
                placeholder=f"Filter {col}",
                styles=SELECT_LABEL_STYLE,
                data=options,
                value=current_value,
                clearable=True,
                searchable=True,
                w="100%",
                persistence=True,
                persistence_type="session",
                persisted_props=["value", "data"],
                debounce=True,
            )
        )

    return children


@callback(
    Output("app-shell", "navbar"),
    Input("burger", "opened"),
    State("app-shell", "navbar"),
)
def toggle_navbar(opened, navbar):
    """Toggle the navbar width based on burger menu state."""
    navbar["width"] = SIDEBAR_WIDTH_OPEN if opened else SIDEBAR_WIDTH_CLOSED
    return navbar


@callback(
    Output("header-title", "children"),
    Input("location", "pathname"),
)
def update_header_title(pathname):
    """Update the header title based on current page."""
    current_page = next(
        (
            page
            for page in page_registry.values()
            if page.get("relative_path") == pathname
        ),
        None,
    )
    return current_page["title"] if current_page else "None"
