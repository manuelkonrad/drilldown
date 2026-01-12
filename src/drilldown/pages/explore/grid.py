# SPDX-FileCopyrightText: 2026 Manuel Konrad
#
# SPDX-License-Identifier: MIT

"""Grid components for the explore page."""

import dash_ag_grid as dag
import dash_mantine_components as dmc
from dash_iconify import DashIconify

from drilldown.constants import BUTTON_ICON_SIZE, BUTTON_SIZE, EXPLORE_PREFIX

grid_container = dmc.Flex(
    [
        dmc.Grid(
            [
                dmc.GridCol(
                    dmc.TextInput(
                        id=f"{EXPLORE_PREFIX}-quick-filter-input",
                        placeholder="Filter table",
                    ),
                    span="auto",
                    pr="xs",
                ),
                dmc.GridCol(
                    dmc.Flex(
                        [
                            dmc.ActionIcon(
                                DashIconify(
                                    icon="material-symbols:download-sharp",
                                    width=BUTTON_ICON_SIZE,
                                ),
                                id=f"{EXPLORE_PREFIX}-export-button",
                                size=f"{BUTTON_SIZE}px",
                                variant="default",
                            ),
                        ],
                        gap="xs",
                    ),
                    span="content",
                ),
            ],
            gutter=0,
            w="100%",
        ),
        dmc.Flex(
            dag.AgGrid(
                id=f"{EXPLORE_PREFIX}-ag-grid",
                className="ag-theme-quartz ag-theme-drilldown",
                rowData=[],
                columnDefs=[
                    {
                        "field": " ",
                    }
                ],
                style={"height": "100%", "width": "100%", "display": "none"},
                dashGridOptions={
                    "rowSelection": "multiple",
                },
                defaultColDef={"sortable": False},
            ),
            id=f"{EXPLORE_PREFIX}-ag-grid-container",
            style={"height": "100%", "width": "100%"},
        ),
    ],
    direction="column",
    gap="xs",
    pt="xs",
    style={"height": "100%", "width": "100%"},
)
