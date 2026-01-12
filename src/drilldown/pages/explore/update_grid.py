# SPDX-FileCopyrightText: 2026 Manuel Konrad
#
# SPDX-License-Identifier: MIT

"""Callbacks for updating the grid in the explore page."""

from typing import Any

import dash_ag_grid as dag
import dash_mantine_components as dmc
import pandas as pd
from dash import Input, Output, Patch, State, callback

from drilldown.constants import (
    CURVE_FORMATS,
    DARK_THEME_CLASS,
    DEFAULT_NO_DATASET_MESSAGE,
    EXPLORE_PREFIX,
    IMAGE_FORMATS,
    LIGHT_THEME_CLASS,
    THEME_DARK,
)
from drilldown.feature_store import ColumnType, FeatureStore, TypeGroups


def _create_empty_grid(info: str, grid_class: str) -> dag.AgGrid:
    """Create an empty grid when no dataset is selected."""
    return dag.AgGrid(
        id=f"{EXPLORE_PREFIX}-ag-grid",
        className=grid_class,
        rowData=[],
        columnDefs=[{"field": " "}],
        style={"height": "100%", "width": "100%"},
        dashGridOptions={"rowSelection": "multiple"},
        defaultColDef={"sortable": False},
    )


def _build_columns(
    columns: tuple[str | None, str | None, dict[str, list[str]]],
) -> tuple[list[str], list[str]]:
    """Build grid columns and time/categorical/numerical columns."""
    index_col = columns[0]
    time_col = columns[1]
    col_types = columns[2]

    grid_columns = []
    time_cat_num_obj = []

    if time_col:
        grid_columns.append(time_col)
        time_cat_num_obj.append(time_col)

    if index_col:
        grid_columns.append(index_col)

    grid_columns.extend(filter(lambda x: x != index_col, col_types["identifier"]))

    for col_type in TypeGroups.DATETIME_VARS + ["categorical", "numerical"]:
        cols = col_types[col_type]
        if col_type == "datetime":
            cols = list(filter(lambda x: x != time_col, cols))
        grid_columns.extend(cols)
        time_cat_num_obj.extend(cols)

    grid_columns.extend(col_types["object"])
    grid_columns.extend(col_types["curve"])

    return grid_columns, time_cat_num_obj


def _build_custom_data(grid_columns: list[str], curve_columns: list[str]) -> list[str]:
    """Build custom data list from grid columns."""
    return list(grid_columns)


def _create_populated_grid(
    df: pd.DataFrame, grid_columns: list[str], grid_class: str
) -> dag.AgGrid:
    """Create a populated grid with data."""
    return dag.AgGrid(
        id=f"{EXPLORE_PREFIX}-ag-grid",
        rowData=df.to_dict("records"),
        columnDefs=[{"field": col, "headerName": col} for col in grid_columns],
        style={"height": "100%", "width": "100%"},
        className=grid_class,
        dashGridOptions={"rowSelection": "multiple"},
        defaultColDef={
            "filter": True,
            "sortable": True,
            "floatingFilter": True,
            "suppressMenu": True,
        },
        persistence=True,
        persistence_type="session",
    )


def _build_sample_columns_dropdown(sample_columns: list[str]) -> list[dict[str, str]]:
    """Build dropdown options for sample columns."""
    return [{"value": col, "label": col} for col in sample_columns]


def _build_dataset_info(
    feature_store: FeatureStore, collection: str, dataset: str, custom_data: list[str]
) -> list:
    """Build dataset information components."""
    dataset_item = feature_store.collections[collection][dataset]

    components = [
        dmc.Table(
            withTableBorder=False,
            withColumnBorders=True,
            layout="fixed",
            variant="vertical",
            children=[
                dmc.TableTbody(
                    [
                        dmc.TableTr(
                            [
                                dmc.TableTh(
                                    dmc.Text("Collection", fw=600, size="sm"), w=120
                                ),
                                dmc.TableTd(collection),
                            ]
                        ),
                        dmc.TableTr(
                            [
                                dmc.TableTh(dmc.Text("Dataset", fw=600, size="sm")),
                                dmc.TableTd(dataset),
                            ]
                        ),
                        dmc.TableTr(
                            [
                                dmc.TableTh(dmc.Text("Description", fw=600, size="sm")),
                                dmc.TableTd(dataset_item.description),
                            ]
                        ),
                    ]
                )
            ],
        )
    ]

    components.append(dmc.Space(h="xs"))

    # Build table data for fields
    table_data = []
    for col_name in custom_data:
        col = dataset_item.columns[col_name]
        table_data.append(
            {
                "Field": col_name,
                "Type": col.column_type,
                "Description": col.description,
            }
        )

    if table_data:
        components.append(
            dmc.Table(
                data={
                    "head": ["Field", "Type", "Description"],
                    "body": [
                        [row["Field"], row["Type"], row["Description"]]
                        for row in table_data
                    ],
                },
                striped=False,
                highlightOnHover=True,
                withTableBorder=False,
                withColumnBorders=True,
            )
        )

    return components


@callback(
    Output(f"{EXPLORE_PREFIX}-ag-grid", "className"),
    Input("theme-toggle", "value"),
    prevent_initial_call=True,
)
def update_grid_theme(theme: str) -> str:
    """Update grid theme based on theme toggle value."""
    if theme == THEME_DARK:
        return DARK_THEME_CLASS
    return LIGHT_THEME_CLASS


@callback(
    Output(f"{EXPLORE_PREFIX}-ag-grid", "dashGridOptions"),
    Input(f"{EXPLORE_PREFIX}-quick-filter-input", "value"),
    prevent_initial_call=True,
)
def update_quick_filter(filter_text: str) -> dict:
    """Update quick filter text for the grid."""
    options_path = Patch()
    options_path["quickFilterText"] = filter_text
    return options_path


@callback(
    Output(f"{EXPLORE_PREFIX}-ag-grid", "exportDataAsCsv"),
    Input(f"{EXPLORE_PREFIX}-export-button", "n_clicks"),
    prevent_initial_call=True,
)
def export_csv(n_clicks: int | None) -> bool:
    """Export grid data as CSV when export button is clicked."""
    if n_clicks:
        return True
    return False


@callback(
    Output(f"{EXPLORE_PREFIX}-ag-grid-container", "children"),
    Output(f"{EXPLORE_PREFIX}-store", "data"),
    Output(f"{EXPLORE_PREFIX}-color-field-select", "data"),
    Output(f"{EXPLORE_PREFIX}-dimensions-select", "data"),
    Output(f"{EXPLORE_PREFIX}-sample-images-select", "data"),
    Output(f"{EXPLORE_PREFIX}-sample-curves-select", "data"),
    Output(f"{EXPLORE_PREFIX}-dataset-info-container", "children"),
    Input("main-store", "data"),
    State("theme-toggle", "value"),
)
def update_grid_and_dropdowns(main_store: dict[str, Any] | None, theme: str) -> tuple:
    """Update grid and all dropdown components based on main store data."""
    if theme == THEME_DARK:
        grid_class = DARK_THEME_CLASS
    else:
        grid_class = LIGHT_THEME_CLASS

    if not main_store or not main_store.get("data"):
        info_text = (
            main_store.get("info", DEFAULT_NO_DATASET_MESSAGE)
            if main_store
            else DEFAULT_NO_DATASET_MESSAGE
        )
        info_components = [dmc.Text(info_text, size="sm")]
        return (
            _create_empty_grid(info_text, grid_class),
            None,
            None,
            None,
            None,
            None,
            info_components,
        )

    # Extract data from main_store
    df = pd.DataFrame(main_store["data"])
    collection = main_store["collection"]
    dataset = main_store["dataset"]
    columns = main_store["columns"]
    col_types = columns[2]

    # Build column lists
    grid_columns, time_cat_num_obj = _build_columns(columns)
    custom_data = _build_custom_data(grid_columns, col_types["curve"])

    # Update columns with custom_data
    col_types["custom_data"] = custom_data

    # Create grid
    grid = _create_populated_grid(df, grid_columns, grid_class)

    # Build dataset info
    feature_store = FeatureStore.model_validate_json(main_store["feature_store"])
    dataset_info = _build_dataset_info(feature_store, collection, dataset, custom_data)

    # Build dropdown options
    curve_columns = col_types["curve"]
    image_columns = []
    for col in col_types["object"]:
        col_obj = feature_store.collections[collection][dataset].columns[col]
        # Check if column type is URI_IMG or if object_format indicates an image
        if col_obj.column_type == ColumnType.URI_IMG or (
            col_obj.object_format and col_obj.object_format in IMAGE_FORMATS
        ):
            image_columns.append(col)
        elif col_obj.column_type == ColumnType.URI_CURVE or (
            col_obj.object_format and col_obj.object_format in CURVE_FORMATS
        ):
            curve_columns.append(col)

    sample_images_columns_dropdown = _build_sample_columns_dropdown(image_columns)
    sample_curves_columns_dropdown = _build_sample_columns_dropdown(curve_columns)

    # Build grouped options for both dimensions and color column
    numerical_columns = col_types.get("numerical", [])
    categorical_columns = col_types.get("categorical", [])
    timestamp_columns = col_types.get("timestamp", [])

    # Grouped dimension options (all column types)
    grouped_dimension_options = []
    if numerical_columns:
        grouped_dimension_options.append(
            {
                "group": "Numerical Fields",
                "items": [{"value": col, "label": col} for col in numerical_columns],
            }
        )
    if categorical_columns:
        grouped_dimension_options.append(
            {
                "group": "Categorical Fields",
                "items": [{"value": col, "label": col} for col in categorical_columns],
            }
        )
    if timestamp_columns:
        grouped_dimension_options.append(
            {
                "group": "Timestamp Fields",
                "items": [{"value": col, "label": col} for col in timestamp_columns],
            }
        )

    # Grouped color column options (numerical and categorical only)
    grouped_color_options = []
    if numerical_columns:
        grouped_color_options.append(
            {
                "group": "Numerical Fields",
                "items": [{"value": col, "label": col} for col in numerical_columns],
            }
        )
    if categorical_columns:
        grouped_color_options.append(
            {
                "group": "Categorical Fields",
                "items": [{"value": col, "label": col} for col in categorical_columns],
            }
        )

    return (
        grid,
        {
            "columns": columns,
            "feature_store": main_store["feature_store"],
            "collection": collection,
            "dataset": dataset,
            "dataset_info": dataset_info,
        },
        grouped_color_options,
        grouped_dimension_options,
        sample_images_columns_dropdown,
        sample_curves_columns_dropdown,
        dataset_info,
    )
