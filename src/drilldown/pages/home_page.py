# SPDX-FileCopyrightText: 2026 Manuel Konrad
#
# SPDX-License-Identifier: MIT

"""Home page layout."""

import dash_mantine_components as dmc
from dash import (
    dcc,
    get_app,
    page_registry,
    register_page,
)

from drilldown.config import Config
from drilldown.constants import CARD_SIZE, PAGE_CONTAINER_HEIGHT


def _get_card(page: dict, config: Config) -> dmc.Card:
    """Build a navigation card for a page."""
    relative_path = page["relative_path"]
    title = page.get("title", "")
    description = page.get("description", "")
    return dcc.Link(
        dmc.Card(
            [
                dmc.Flex(
                    [
                        dmc.Text(title, fw=600, mb="xs"),
                        dmc.Text(description, size="md", c="dimmed"),
                    ],
                    gap="md",
                    direction="column",
                ),
            ],
            withBorder=True,
            shadow="sm",
            radius="md",
            w=CARD_SIZE // 2,
            h=CARD_SIZE // 1.5,
            classNames={"root": "card-root-custom"},
        ),
        href=relative_path,
        style={"textDecoration": "none"},
    )


def layout(**kwargs):
    app = get_app()
    config = app.drilldown_config
    return dmc.Flex(
        dmc.Flex(
            [
                dmc.Center(
                    dmc.Text(
                        "Discover tools and data",
                        variant="gradient",
                        gradient={"from": "cyan", "to": "blue", "deg": 45},
                        style={"fontSize": 76},
                    ),
                    w="100%",
                ),
                dmc.Flex(
                    [
                        _get_card(page, config)
                        for page in page_registry.values()
                        if page["path"] != "/"
                    ],
                    justify="center",
                    align="center",
                    gap="sm",
                    mt="xl",
                    mb="md",
                    w="100%",
                ),
            ],
            align="flex-start",
            justify="center",
            direction="column",
            mt="xl",
        ),
        align="flex-start",
        justify="center",
        p="xs",
        h=PAGE_CONTAINER_HEIGHT,
        w="100%",
        style={"overflow": "hidden"},
    )


register_page(
    __name__,
    path="/",
    title="Home",
    icon="material-symbols:stacks-outline",
    layout=layout,
    order=0,
)
