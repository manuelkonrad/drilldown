# SPDX-FileCopyrightText: 2026 Manuel Konrad
#
# SPDX-License-Identifier: MIT

"""Explore page layout and callbacks."""

# ruff: noqa: F401 (required for importing callbacks)

import dash
import dash_mantine_components as dmc
from dash import Input, Output, dcc, html

from drilldown.constants import (
    EXPLORE_PREFIX,
    PAGE_CONTAINER_HEIGHT,
    TAB_LIST_OFFSET,
)
from drilldown.pages.explore.chart import chart_container
from drilldown.pages.explore.grid import grid_container
from drilldown.pages.explore.sample_view import (
    sample_curves_container,
    sample_images_container,
)
from drilldown.pages.explore.update_chart import update_chart
from drilldown.pages.explore.update_grid import (
    update_grid_and_dropdowns,
)
from drilldown.pages.explore.update_sample_view import (
    update_sample_view,
)


def layout(**kwargs):
    left_tabs = dmc.Flex(
        dmc.Tabs(
            [
                dmc.TabsList(
                    [
                        dmc.TabsTab(
                            "Table",
                            value="table",
                        ),
                        dmc.TabsTab(
                            "Charts",
                            value="charts",
                        ),
                    ],
                ),
                dmc.TabsPanel(
                    grid_container,
                    value="table",
                    style={
                        "height": f"calc( 100% - {TAB_LIST_OFFSET} )",
                        "width": "100%",
                    },
                ),
                dmc.TabsPanel(
                    chart_container,
                    value="charts",
                    style={
                        "height": f"calc( 100% - {TAB_LIST_OFFSET} )",
                        "width": "100%",
                    },
                ),
            ],
            id=f"{EXPLORE_PREFIX}-left-tabs",
            value="table",
            color="blue",
            style={
                "height": "100%",
                "width": "100%",
            },
            styles={"list": {"minWidth": "150px"}, "root": {"overflow": "hidden"}},
            persistence=True,
            persistence_type="session",
        ),
        h="100%",
        p="xs",
        w="67%",
        style={
            "background-color": "light-dark(var(--mantine-color-white), var(--mantine-color-dark-8))",
            "border-radius": "8px",
        },
    )

    right_tabs = dmc.Flex(
        dmc.Tabs(
            [
                dmc.TabsList(
                    [
                        dmc.TabsTab(
                            "Images",
                            value="sample-images",
                        ),
                        dmc.TabsTab(
                            "Curves",
                            value="sample-curves",
                        ),
                        dmc.TabsTab(
                            "Values",
                            value="sample-values",
                        ),
                        dmc.TabsTab(
                            "Description",
                            value="dataset-info",
                        ),
                    ],
                ),
                dmc.TabsPanel(
                    sample_images_container,
                    value="sample-images",
                    style={
                        "height": f"calc( 100% - {TAB_LIST_OFFSET} )",
                        "width": "100%",
                    },
                ),
                dmc.TabsPanel(
                    sample_curves_container,
                    value="sample-curves",
                    style={
                        "height": f"calc( 100% - {TAB_LIST_OFFSET} )",
                        "width": "100%",
                    },
                ),
                dmc.TabsPanel(
                    dmc.ScrollArea(
                        dmc.CodeHighlight(
                            language="json",
                            code="No sample selected.",
                            id=f"{EXPLORE_PREFIX}-sample-values-code",
                        ),
                        h="100%",
                        offsetScrollbars=True,
                    ),
                    value="sample-values",
                    style={
                        "height": f"calc( 100% - {TAB_LIST_OFFSET} )",
                        "width": "100%",
                    },
                    pt="xs",
                ),
                dmc.TabsPanel(
                    dmc.ScrollArea(
                        dmc.Flex(
                            id=f"{EXPLORE_PREFIX}-dataset-info-container",
                            direction="column",
                            gap="sm",
                        ),
                        h="100%",
                        offsetScrollbars=True,
                    ),
                    value="dataset-info",
                    style={
                        "height": f"calc( 100% - {TAB_LIST_OFFSET} )",
                        "width": "100%",
                    },
                    pt="xs",
                ),
            ],
            id=f"{EXPLORE_PREFIX}-right-tabs",
            value="sample-images",
            color="blue",
            style={
                "height": "100%",
                "width": "100%",
            },
            styles={"list": {"minWidth": "350px"}, "root": {"overflow": "hidden"}},
            persistence=True,
            persistence_type="session",
        ),
        p="xs",
        h="100%",
        w="calc(33% - 8px)",
        style={
            "background-color": "light-dark(var(--mantine-color-white), var(--mantine-color-dark-8))",
            "border-radius": "8px",
            "overflow": "hidden",
        },
    )

    # Draggable separator between left and right tabs
    separator = html.Div(
        id=f"{EXPLORE_PREFIX}-layout-separator",
        style={
            "width": "8px",
            "cursor": "col-resize",
            "backgroundColor": "var(--mantine-color-dark-6)",
            "position": "relative",
            "zIndex": "1",
        },
    )

    return dmc.Flex(
        [
            left_tabs,
            separator,
            right_tabs,
            dcc.Store(id=f"{EXPLORE_PREFIX}-store", data=None),
            dcc.Store(id=f"{EXPLORE_PREFIX}-sample-view-store", data=None),
            dcc.Store(id=f"{EXPLORE_PREFIX}-draggable-trigger", data=0),
        ],
        id=f"{EXPLORE_PREFIX}-upper-container",
        style={
            "width": "100%",
            "height": PAGE_CONTAINER_HEIGHT,
            "visibility": "hidden",
            "background-color": "var(--mantine-color-dark-6)",
        },
    )


# Clientside callback to initialize draggable separator
dash.clientside_callback(
    """
    function(trigger) {
        let retryCount = 0;
        const maxRetries = 50;

        function initDraggableSeparator() {
            const separator = document.getElementById('explore-layout-separator');
            const container = document.getElementById('explore-upper-container');
            const storageKey = 'explore-layout-split';

            if (!separator || !container) {
                retryCount++;
                if (retryCount < maxRetries) {
                    setTimeout(initDraggableSeparator, 100);
                } else {
                    // Fallback: make container visible even if initialization fails
                    if (container) {
                        container.style.visibility = 'visible';
                    }
                }
                return trigger;
            }

            const leftTabs = container.children[0];
            const rightTabs = container.children[2];

            if (!leftTabs || !rightTabs || separator.hasAttribute('data-initialized')) {
                // Fallback: ensure visibility if already initialized or missing elements
                container.style.visibility = 'visible';
                return trigger;
            }

            separator.setAttribute('data-initialized', 'true');

            let isDragging = false;
            let lastPercent = null;
            let containerRect = null;
            const separatorHalfWidth = separator.offsetWidth / 2;

            const applySplit = (percent) => {
                const minPercent = 10;
                const maxPercent = 90;
                const clampedPercent = Math.max(minPercent, Math.min(maxPercent, percent));
                leftTabs.style.width = clampedPercent + '%';
                rightTabs.style.width = 'calc(' + (100 - clampedPercent) + '% - 8px)';
                lastPercent = clampedPercent;
            };

            // Apply persisted split for the session, if present
            const savedSplit = sessionStorage.getItem(storageKey);
            if (savedSplit !== null) {
                const parsed = parseFloat(savedSplit);
                if (!Number.isNaN(parsed)) {
                    applySplit(parsed);
                }
            }

            // Reveal container once sizing is applied (prevents flash of default widths)
            container.style.visibility = 'visible';

            separator.addEventListener('mousedown', function(e) {
                isDragging = true;
                containerRect = container.getBoundingClientRect();
                separator.style.cursor = 'col-resize';
                document.body.style.cursor = 'col-resize';
                document.body.style.userSelect = 'none';
                e.preventDefault();

                function onMouseMove(e) {
                    if (!isDragging || !containerRect) return;
                    const containerWidth = container.offsetWidth;
                    const pointerX = e.clientX - containerRect.left - separatorHalfWidth;
                    const newLeftPercent = (pointerX / containerWidth) * 100;
                    applySplit(newLeftPercent);
                }

                function onMouseUp() {
                    if (isDragging) {
                        isDragging = false;
                        containerRect = null;
                        separator.style.cursor = 'col-resize';
                        document.body.style.cursor = '';
                        document.body.style.userSelect = '';
                        if (lastPercent !== null) {
                            sessionStorage.setItem(storageKey, lastPercent);
                        }
                    }
                    document.removeEventListener('mousemove', onMouseMove);
                    document.removeEventListener('mouseup', onMouseUp);
                }

                document.addEventListener('mousemove', onMouseMove);
                document.addEventListener('mouseup', onMouseUp);
            });
        }

        setTimeout(initDraggableSeparator, 100);
        return trigger;
    }
    """,
    Output(f"{EXPLORE_PREFIX}-draggable-trigger", "data"),
    Input(f"{EXPLORE_PREFIX}-draggable-trigger", "data"),
)


dash.register_page(
    __name__,
    path="/explore/",
    title="Explore",
    description="Browse multimodal data and gain insights through interactive visualizations.",
    icon="material-symbols:explore-outline",
    layout=layout,
    redirect_from=["/explore"],
    order=1,
)
