# SPDX-FileCopyrightText: 2026 Manuel Konrad
#
# SPDX-License-Identifier: MIT

"""Layout components for the drilldown application."""

# ruff: noqa: F401 (required for importing callbacks)

import datetime
from typing import Any

import dash_mantine_components as dmc
from dash import (
    dcc,
    get_asset_url,
    get_relative_path,
    page_container,
    page_registry,
)
from dash_iconify import DashIconify

import drilldown.callbacks
from drilldown.config import Config
from drilldown.constants import (
    BUTTON_ICON_SIZE,
    BUTTON_SIZE,
    HEADER_HEIGHT,
    HEADER_INPUT_STYLE,
    SIDEBAR_FONT_SIZE,
    SIDEBAR_ICON_SIZE,
    SIDEBAR_WIDTH_CLOSED,
    THEME_DARK,
    THEME_LIGHT,
    TOGGLE_ICON_SIZE,
)

theme_toggle = dmc.SegmentedControl(
    id="theme-toggle",
    value=THEME_DARK,
    data=[
        {
            "label": dmc.Center(
                DashIconify(
                    icon="material-symbols:moon-stars-outline", width=TOGGLE_ICON_SIZE
                )
            ),
            "value": THEME_DARK,
        },
        {
            "label": dmc.Center(
                DashIconify(
                    icon="material-symbols:sunny-outline", width=TOGGLE_ICON_SIZE
                )
            ),
            "value": THEME_LIGHT,
        },
    ],
    size="xs",
    persistence=True,
    persistence_type="session",
    color="gray",
    variant="light",
)


def header(config: Config) -> Any:
    """Create the header component with data selectors and controls."""
    return dmc.Box(
        [
            dmc.Grid(
                [
                    dmc.GridCol(
                        dmc.Flex(
                            dmc.Title(
                                "Home",
                                order=4,
                                pb=4,
                                visibleFrom="lg",
                                id="header-title",
                                c="var(--mantine-color-dark-0)",
                                style={
                                    "white-space": "nowrap",
                                    "overflow": "hidden",
                                    "text-overflow": "ellipsis",
                                },
                            ),
                            pl="xs",
                            visibleFrom="sm",
                            justify="flex-start",
                            align="center",
                            h=HEADER_HEIGHT,
                        ),
                        span="auto",
                    ),
                    dmc.GridCol(
                        dmc.Flex(
                            [
                                dmc.Select(
                                    placeholder="Select collection",
                                    id="collection-select",
                                    value="__undefined__",
                                    mr="xs",
                                    w=200,
                                    data=[
                                        {
                                            "value": "__undefined__",
                                            "label": "n/a",
                                        }
                                    ],
                                    styles=HEADER_INPUT_STYLE,
                                    searchable=True,
                                    clearable=False,
                                    persistence=True,
                                    persistence_type="session",
                                ),
                                dmc.Select(
                                    placeholder="Select dataset",
                                    id="dataset-select",
                                    value="__undefined__",
                                    w=200,
                                    data=[
                                        {
                                            "value": "__undefined__",
                                            "label": "n/a",
                                        }
                                    ],
                                    styles=HEADER_INPUT_STYLE,
                                    searchable=True,
                                    clearable=False,
                                    persistence=True,
                                    persistence_type="session",
                                ),
                                dmc.Flex(
                                    dcc.Clipboard(
                                        id="clipboard",
                                        className="custom-clipboard",
                                        title="Copy link to clipboard",
                                    ),
                                    mr="xs",
                                    w=f"{BUTTON_SIZE}px",
                                    h=f"{BUTTON_SIZE}px",
                                    align="center",
                                    justify="center",
                                    style={
                                        "border-right": "1px solid var(--mantine-color-dark-4)",
                                        "border-top": "1px solid var(--mantine-color-dark-4)",
                                        "border-bottom": "1px solid var(--mantine-color-dark-4)",
                                    },
                                ),
                                dmc.DatePickerInput(
                                    id="date-picker",
                                    type="range",
                                    value=[
                                        datetime.datetime.now().date()
                                        - datetime.timedelta(
                                            days=config.default_interval
                                        ),
                                        datetime.datetime.now().date(),
                                    ],
                                    maw=350,
                                    miw=200,
                                    placeholder="Select date range",
                                    allowSingleDateInRange=False,
                                    clearable=False,
                                    persistence=True,
                                    persistence_type="session",
                                    styles=HEADER_INPUT_STYLE,
                                ),
                                dmc.Popover(
                                    [
                                        dmc.PopoverTarget(
                                            dmc.ActionIcon(
                                                DashIconify(
                                                    icon="material-symbols:filter-list",
                                                    width=BUTTON_ICON_SIZE,
                                                ),
                                                size=f"{BUTTON_SIZE}px",
                                                variant="transparent",
                                                color="var(--mantine-color-dark-0)",
                                                styles={
                                                    "root": {
                                                        "--ai-hover-color": "#ffffff",
                                                        "border-right": "1px solid var(--mantine-color-dark-4)",
                                                        "border-top": "1px solid var(--mantine-color-dark-4)",
                                                        "border-bottom": "1px solid var(--mantine-color-dark-4)",
                                                    }
                                                },
                                            ),
                                        ),
                                        dmc.PopoverDropdown(
                                            dmc.ScrollArea(
                                                dmc.Flex(
                                                    [
                                                        dmc.Text(
                                                            "Select a dataset to see filters.",
                                                            size="sm",
                                                            mt="xs",
                                                        ),
                                                    ],
                                                    id="header-filters-container",
                                                    gap="xs",
                                                    direction="column",
                                                    mah=400,
                                                ),
                                                h="auto",
                                                offsetScrollbars=True,
                                            ),
                                            p="sm",
                                        ),
                                    ],
                                    id="header-filters-popover",
                                    width=400,
                                    position="bottom-end",
                                    shadow="md",
                                    keepMounted=True,
                                    closeOnClickOutside=False,
                                ),
                            ],
                            h=HEADER_HEIGHT,
                            align="center",
                            justify="flex-start",
                            ml=0,
                            p="sm",
                            wrap="nowrap",
                        ),
                        span="content",
                    ),
                    dmc.GridCol(
                        [
                            dmc.Flex(
                                [
                                    dmc.Flex(
                                        [
                                            dmc.Image(
                                                src=get_relative_path(
                                                    "/internal_assets/logo.svg"
                                                ),
                                                h="100%",
                                                w="auto",
                                            ),
                                        ],
                                        h=HEADER_HEIGHT,
                                        p="xs",
                                        justify="flex-end",
                                        align="center",
                                        visibleFrom="xl",
                                    ),
                                    dmc.Flex(
                                        [
                                            dmc.Popover(
                                                [
                                                    dmc.PopoverTarget(
                                                        dmc.ActionIcon(
                                                            DashIconify(
                                                                icon="material-symbols:more-vert",
                                                                width=BUTTON_ICON_SIZE,
                                                            ),
                                                            size=f"{BUTTON_SIZE}px",
                                                            variant="transparent",
                                                            color="var(--mantine-color-dark-0)",
                                                            styles={
                                                                "root": {
                                                                    "--ai-hover-color": "#ffffff",
                                                                }
                                                            },
                                                        ),
                                                    ),
                                                    dmc.PopoverDropdown(
                                                        dmc.Flex(
                                                            [
                                                                theme_toggle,
                                                            ],
                                                            gap="md",
                                                            direction="column",
                                                        ),
                                                        p="sm",
                                                    ),
                                                ],
                                                width=150,
                                                position="bottom-end",
                                                shadow="md",
                                                keepMounted=True,
                                            ),
                                        ],
                                        h=HEADER_HEIGHT,
                                        p="xs",
                                        justify="flex-end",
                                        align="center",
                                    ),
                                ],
                                justify="flex-end",
                                align="center",
                            ),
                        ],
                        span="auto",
                    ),
                ],
                h=HEADER_HEIGHT,
                gutter=0,
                overflow="hidden",
            )
        ],
        style={
            "padding": 0,
            "margin": 0,
        },
        h=HEADER_HEIGHT,
    )


def navbar(config: Config) -> Any:
    """Create the navigation sidebar with page links."""
    navlink_padding = (SIDEBAR_WIDTH_CLOSED - SIDEBAR_ICON_SIZE) // 2
    return dmc.AppShellNavbar(
        [
            dmc.Flex(
                [
                    dmc.Flex(
                        dmc.Burger(
                            id="burger",
                            opened=False,
                            size="sm",
                            color="var(--mantine-color-dark-0)",
                        ),
                        justify="center",
                        align="center",
                        h=HEADER_HEIGHT,
                        w=SIDEBAR_WIDTH_CLOSED,
                    ),
                ],
                align="center",
                justify="flex-end",
                mb="sm",
            ),
            *[
                dmc.NavLink(
                    label=page["title"],
                    leftSection=DashIconify(
                        icon=page["icon"],
                        width=SIDEBAR_ICON_SIZE,
                    )
                    if config.custom_icons.get(page["title"]) is None
                    else dmc.Flex(
                        dmc.Image(
                            src=get_asset_url(config.custom_icons[page["title"]]),
                            w="auto",
                            h="auto",
                        ),
                        justify="center",
                        align="center",
                        h=SIDEBAR_ICON_SIZE,
                        w=SIDEBAR_ICON_SIZE,
                    ),
                    href=page["relative_path"],
                    active="exact",
                    variant="light",
                    color="blue",
                    classNames={
                        "root": "navlink-custom",
                    },
                    styles={
                        "root": {
                            "padding": 0,
                        },
                        "section": {
                            "padding": f"{navlink_padding}px",
                            "margin": 0,
                        },
                        "label": {
                            "font-size": SIDEBAR_FONT_SIZE,
                            "padding": f"{navlink_padding}px",
                            "vertical-align": "2px",
                            "white-space": "nowrap",
                            "overflow": "hidden",
                            "text-overflow": "ellipsis",
                        },
                    },
                )
                for page_module, page in page_registry.items()
            ],
        ],
        id="navbar",
        withBorder=False,
    )


def layout(config: Config) -> Any:
    """Create the main application layout."""
    header_line = dmc.AppShellHeader(
        [
            dmc.BackgroundImage(
                src=config.header_image,
                w="100%",
                h=config.header_line_height,
            )
            if config.header_image
            else dmc.Box(
                w="100%",
                h=config.header_line_height,
                bg=config.header_background,
            ),
        ],
    )

    return dmc.AppShell(
        [
            dcc.Location(id="location", refresh="callback-nav"),
            header_line,
            navbar(config),
            dmc.AppShellMain(
                [
                    header(config),
                    dmc.Box(
                        page_container,
                        style={
                            "background-color": "light-dark(var(--mantine-color-white), var(--mantine-color-dark-8))",
                            "border-radius": "8px",
                        },
                        mb="6px",
                        mr="6px",
                    ),
                    dcc.Store(id="main-store", data=None),
                ],
            ),
        ],
        id="app-shell",
        header={
            "height": config.header_line_height,
        },
        navbar={
            "width": SIDEBAR_WIDTH_CLOSED,
            "breakpoint": 0,
            "collapsed": {
                "mobile": False,
                "desktop": False,
            },
        },
        styles={
            "root": {
                "backgroundColor": "var(--mantine-color-dark-6)",
            },
            "navbar": {
                "backgroundColor": "var(--mantine-color-dark-6)",
            },
        },
    )
