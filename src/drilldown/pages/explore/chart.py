# SPDX-FileCopyrightText: 2026 Manuel Konrad
#
# SPDX-License-Identifier: MIT

"""Chart components for the explore page."""

import dash_mantine_components as dmc
import plotly.graph_objects as go
from dash import dcc
from dash_iconify import DashIconify

from drilldown.constants import (
    BUTTON_ICON_SIZE,
    BUTTON_SIZE,
    EMPTY_FIGURE_LAYOUT,
    EXPLORE_PREFIX,
    SELECT_MAX_DROPDOWN_HEIGHT,
)

chart_container = dmc.Flex(
    [
        dmc.Flex(
            [
                dmc.Flex(
                    [
                        dmc.Select(
                            placeholder="Select chart type",
                            id=f"{EXPLORE_PREFIX}-chart-select",
                            value="scatter",
                            data=[
                                {"value": "scatter", "label": "Scatter Plot"},
                                {"value": "line", "label": "Time Series"},
                                {"value": "box", "label": "Box Plot"},
                                {"value": "hist", "label": "Histogram"},
                                {"value": "parallel", "label": "Parallel Coordinates"},
                                {"value": "cycle", "label": "Cycle Plot"},
                                {
                                    "value": "cluster",
                                    "label": "t-SNE Plot (K-Means / PCA)",
                                },
                            ],
                            w="100%",
                            clearable=False,
                            allowDeselect=False,
                            persistence=True,
                            persistence_type="session",
                            maxDropdownHeight=SELECT_MAX_DROPDOWN_HEIGHT,
                        ),
                        dmc.Select(
                            placeholder="Select color field",
                            id=f"{EXPLORE_PREFIX}-color-field-select",
                            w="100%",
                            clearable=True,
                            persistence=True,
                            persistence_type="session",
                            persisted_props=["value", "data"],
                            maxDropdownHeight=SELECT_MAX_DROPDOWN_HEIGHT,
                        ),
                    ],
                    justify="flex-start",
                    align="center",
                    gap="xs",
                    w="100%",
                ),
                dmc.Popover(
                    [
                        dmc.PopoverTarget(
                            dmc.ActionIcon(
                                DashIconify(
                                    icon="material-symbols:more-vert",
                                    width=BUTTON_ICON_SIZE,
                                ),
                                size=f"{BUTTON_SIZE}px",
                                variant="default",
                            ),
                        ),
                        dmc.PopoverDropdown(
                            [
                                dmc.Text("Histogram: Type", mb="xs"),
                                dmc.SegmentedControl(
                                    id=f"{EXPLORE_PREFIX}-histtype-select",
                                    value="1D",
                                    data=[
                                        {"value": "1D", "label": "1D"},
                                        {"value": "2D", "label": "2D"},
                                        {"value": "2D_contour", "label": "2D contour"},
                                    ],
                                    mb="xs",
                                    fullWidth=True,
                                    persistence=True,
                                    persistence_type="session",
                                ),
                                dmc.Text("Histogram: Function", mb="xs"),
                                dmc.SegmentedControl(
                                    id=f"{EXPLORE_PREFIX}-histfunc-select",
                                    value="count",
                                    data=[
                                        {"value": "count", "label": "count"},
                                        {
                                            "value": "count_shared_x",
                                            "label": "count (shared x)",
                                        },
                                        {"value": "avg", "label": "avg"},
                                        {"value": "sum", "label": "sum"},
                                        {"value": "min", "label": "min"},
                                        {"value": "max", "label": "max"},
                                    ],
                                    mb="xs",
                                    fullWidth=True,
                                    persistence=True,
                                    persistence_type="session",
                                ),
                                dmc.Text("Histogram: Bar Mode", mb="xs"),
                                dmc.SegmentedControl(
                                    id=f"{EXPLORE_PREFIX}-barmode-select",
                                    value="group",
                                    data=[
                                        {"value": "group", "label": "group"},
                                        {"value": "relative", "label": "relative"},
                                        {"value": "overlay", "label": "overlay"},
                                    ],
                                    mb="xs",
                                    fullWidth=True,
                                    persistence=True,
                                    persistence_type="session",
                                ),
                                dmc.Text("Histogram: Y-Axis Scale", mb="xs"),
                                dmc.SegmentedControl(
                                    id=f"{EXPLORE_PREFIX}-hist-yscale",
                                    value="linear",
                                    data=[
                                        {"value": "linear", "label": "linear"},
                                        {"value": "log", "label": "log"},
                                    ],
                                    mb="xs",
                                    fullWidth=True,
                                    persistence=True,
                                    persistence_type="session",
                                ),
                                dmc.Text(
                                    "Histogram: Number of Bins (0 = auto)", mb="xs"
                                ),
                                dmc.NumberInput(
                                    id=f"{EXPLORE_PREFIX}-histbins-input",
                                    value=0,
                                    min=0,
                                    max=200,
                                    step=1,
                                    mb="xs",
                                    persistence=True,
                                    persistence_type="session",
                                ),
                                dmc.Text("K-Means: Number of Clusters", mb="xs"),
                                dmc.NumberInput(
                                    id=f"{EXPLORE_PREFIX}-kmeans-nclusters",
                                    value=3,
                                    min=2,
                                    max=20,
                                    step=1,
                                    mb="xs",
                                    persistence=True,
                                    persistence_type="session",
                                ),
                                dmc.Text("PCA: Number of Components", mb="xs"),
                                dmc.NumberInput(
                                    id=f"{EXPLORE_PREFIX}-pca-ndim",
                                    value=8,
                                    min=2,
                                    max=50,
                                    step=1,
                                    mb="xs",
                                    persistence=True,
                                    persistence_type="session",
                                ),
                            ],
                        ),
                    ],
                    id=f"{EXPLORE_PREFIX}-chart-options-popover",
                    width=400,
                    position="bottom-end",
                    shadow="md",
                    keepMounted=True,
                ),
            ],
            align="center",
            gap="xs",
            pt="xs",
            pb="xs",
        ),
        dmc.MultiSelect(
            placeholder="Select dimensions and press enter",
            clearable=True,
            id=f"{EXPLORE_PREFIX}-dimensions-select",
            pb="xs",
            persistence=True,
            persistence_type="session",
            persisted_props=["value", "data"],
            debounce=True,
            maxDropdownHeight=SELECT_MAX_DROPDOWN_HEIGHT,
        ),
        dcc.Loading(
            dcc.Graph(
                id=f"{EXPLORE_PREFIX}-graph",
                figure=go.Figure(layout=EMPTY_FIGURE_LAYOUT),
                style={"height": "100%", "width": "100%"},
                config={
                    "autosizable": True,
                    "frameMargins": 0,
                    "responsive": True,
                    "modeBarButtonsToRemove": [
                        "select2d",
                        "lasso2d",
                    ],
                },
            ),
            parent_style={
                "height": "100%",
                "width": "100%",
            },
            type="dot",
        ),
    ],
    style={
        "height": "100%",
        "width": "100%",
    },
    direction="column",
)
