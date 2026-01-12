# SPDX-FileCopyrightText: 2026 Manuel Konrad
#
# SPDX-License-Identifier: MIT

"""Utility functions for the drilldown application."""

import dash_mantine_components as dmc
import plotly.graph_objects as go
import plotly.io as pio
from dash import dcc

from drilldown.constants import (
    EMPTY_FIGURE_LAYOUT,
    GRAPH_CONFIG,
    GRAPH_STYLE,
    PLOTLY_DARK_BG_COLOR,
    PLOTLY_THEME_DARK,
    PLOTLY_THEME_LIGHT,
    THEME_DARK,
)


def apply_theme(fig: go.Figure, theme: str | None) -> go.Figure:
    """Apply theme to a Plotly figure. Defaults to light if theme is None."""
    theme_name = PLOTLY_THEME_DARK if theme == THEME_DARK else PLOTLY_THEME_LIGHT
    template = pio.templates[theme_name]
    fig.update_layout(template=template)
    if theme_name == PLOTLY_THEME_DARK:
        fig.update_layout(
            paper_bgcolor=PLOTLY_DARK_BG_COLOR,
            plot_bgcolor=PLOTLY_DARK_BG_COLOR,
        )
    return fig


def create_figure_tabs(
    figures: dict[str, go.Figure],
    default_tab: str | None = None,
) -> dmc.Tabs:
    """Create a tabbed interface for multiple figures."""
    if not figures:
        return dcc.Graph(
            figure=go.Figure(layout=EMPTY_FIGURE_LAYOUT),
            style=GRAPH_STYLE,
            config=GRAPH_CONFIG,
        )

    tabs_list = []
    tabs_panels = []

    # Create tabs and panels for each figure
    for tab_name, figure in figures.items():
        tabs_list.append(dmc.TabsTab(tab_name, value=tab_name))
        tabs_panels.append(
            dmc.TabsPanel(
                dcc.Graph(
                    figure=figure,
                    style=GRAPH_STYLE,
                    config=GRAPH_CONFIG,
                ),
                value=tab_name,
            )
        )

    # Use provided default or first tab
    selected_tab = (
        default_tab
        if default_tab and default_tab in figures
        else list(figures.keys())[0]
    )

    return dmc.Tabs(
        [
            dmc.TabsList(tabs_list),
            *tabs_panels,
        ],
        value=selected_tab,
        color="blue",
        placement="left",
        orientation="vertical",
        styles={
            "root": {"height": "100%", "width": "100%"},
            "panel": {"height": "100%", "width": "50%"},
            "tab": {
                "width": "content",
                "overflow": "hidden",
                "max-width": "400px",
                "min-width": "100px",
            },
        },
    )
