# SPDX-FileCopyrightText: 2026 Manuel Konrad
#
# SPDX-License-Identifier: MIT

"""Configuration settings for the drilldown application."""

from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Configuration class for the Dash app.

    Attributes:
        demo: Enable demo mode with sample data.
        debug: Enable debug mode for development.
        host: Host address to bind the server.
        port: Port number to bind the server.
        gunicorn_options: Options for gunicorn server.
        use_built_in_stylesheets: Whether to use built-in CSS.
        url_base_pathname: Base URL pathname for the app.
        dash_kwargs: Additional keyword arguments for Dash app.
        assets_folder: Path to assets folder.
        collection_paths: Paths to data collections.
        theme: Mantine theme configuration.
        custom_icons: Custom icon definitions.
        title: Application title.
        accent_color: Primary accent color.
        header_line_height: Height of header accent line.
        header_background: CSS background for header accent.
        header_image: Optional header image path.
        default_interval: Default date range interval in days.
    """

    model_config = SettingsConfigDict(cli_parse_args=True)

    # Server configuration
    demo: bool = False
    debug: bool = False
    host: str = "127.0.0.1"
    port: int = 8050
    gunicorn_options: dict[str, Any] = {
        "workers": 3,
        "bind": f"{host}:{port}",
    }
    use_built_in_stylesheets: bool = True
    url_base_pathname: str = "/"
    dash_kwargs: dict[str, Any] = {}

    # Collections and assets
    assets_folder: str = "./assets"
    collection_paths: list[str] = []

    # Theme configuration
    theme: dict[str, Any] = {
        "primaryColor": "blue",
        "defaultRadius": 0,
        "colors": {
            "dark": [
                "#c9c9c9",
                "#8aa3b8",
                "#627382",
                "#4f5d69",
                "#323a42",
                "#2c343b",
                "#22292e",
                "#1b2024",
                "#171b1f",
                "#0f1214",
            ],
        },
        "white": "#ffffff",
    }

    # Custom icons
    custom_icons: dict[str, Any] = {}

    # Additional configuration options
    title: str = "drilldown"
    accent_color: str = "blue"
    header_line_height: int = 2
    header_background: str = "linear-gradient(90deg, #0ca678 0%, #1098ad 25%, #1c7ed6 50%, #1098ad 75%, #0ca678 100%)"
    header_image: str | None = None
    default_interval: int = 7
