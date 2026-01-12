# SPDX-FileCopyrightText: 2026 Manuel Konrad
#
# SPDX-License-Identifier: MIT

"""Sample view components for the explore page."""

import dash_mantine_components as dmc
import plotly.graph_objects as go
from dash import dcc
from dash_iconify import DashIconify

from drilldown.constants import (
    BUTTON_ICON_SIZE,
    BUTTON_SIZE,
    EMPTY_FIGURE_LAYOUT,
    EXPLORE_PREFIX,
    GRAPH_CONFIG,
    GRAPH_STYLE,
    SELECT_MAX_DROPDOWN_HEIGHT,
)

sample_images_container = dmc.Flex(
    [
        dmc.Flex(
            [
                dmc.Box(
                    dmc.MultiSelect(
                        placeholder="Select images",
                        id=f"{EXPLORE_PREFIX}-sample-images-select",
                        persistence=True,
                        persistence_type="session",
                        persisted_props=["value", "data"],
                        maxDropdownHeight=SELECT_MAX_DROPDOWN_HEIGHT,
                    ),
                    flex="1",
                ),
            ],
            align="flex-start",
            gap="xs",
            w="100%",
            pb="xs",
            pt="xs",
        ),
        dcc.Graph(
            figure=go.Figure(layout=EMPTY_FIGURE_LAYOUT),
            config=GRAPH_CONFIG,
            style=GRAPH_STYLE,
            id=f"{EXPLORE_PREFIX}-sample-images-subplot",
        ),
    ],
    style={
        "height": "100%",
        "width": "100%",
    },
    gap=0,
    direction="column",
)

sample_curves_container = dmc.Flex(
    [
        dmc.Flex(
            [
                dmc.Box(
                    dmc.MultiSelect(
                        placeholder="Select curves",
                        id=f"{EXPLORE_PREFIX}-sample-curves-select",
                        persistence=True,
                        persistence_type="session",
                        persisted_props=["value", "data"],
                        maxDropdownHeight=SELECT_MAX_DROPDOWN_HEIGHT,
                    ),
                    flex="1",
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
                                dmc.Switch(
                                    size="sm",
                                    id=f"{EXPLORE_PREFIX}-sample-curves-overlay-switch",
                                    radius="sm",
                                    label="Overlay",
                                    checked=False,
                                    persistence=True,
                                    persistence_type="session",
                                ),
                            ],
                        ),
                    ],
                    width=200,
                    position="bottom-end",
                    shadow="md",
                    keepMounted=True,
                ),
            ],
            align="flex-start",
            gap="xs",
            w="100%",
            pb="xs",
            pt="xs",
        ),
        dcc.Graph(
            figure=go.Figure(layout=EMPTY_FIGURE_LAYOUT),
            config=GRAPH_CONFIG,
            style=GRAPH_STYLE,
            id=f"{EXPLORE_PREFIX}-sample-curves-subplot",
        ),
    ],
    style={
        "height": "100%",
        "width": "100%",
    },
    gap=0,
    direction="column",
)
