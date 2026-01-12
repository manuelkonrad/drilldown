# SPDX-FileCopyrightText: 2026 Manuel Konrad
#
# SPDX-License-Identifier: MIT

"""Main application module for the drilldown Dash application."""

import tempfile
from pathlib import Path
from typing import Any

import dash_mantine_components as dmc
from dash import Dash, hooks
from flask import send_from_directory
from gunicorn.app.base import BaseApplication

from drilldown.config import Config
from drilldown.constants import INTERNAL_ASSETS_DIRNAME
from drilldown.helpers.create_demo_data import create_synthetic_dataset
from drilldown.layout import layout


class DrilldownApplication(BaseApplication):
    """Custom Gunicorn application for production serving."""

    def __init__(self, app: Any, gunicorn_options: dict[str, Any] | None = None):
        self.application = app
        self.gunicorn_options = gunicorn_options or {}
        super().__init__()

    def load_config(self) -> None:
        """Load configuration from gunicorn_options into Gunicorn config."""
        for key, val in self.gunicorn_options.items():
            if key in self.cfg.settings and val is not None:
                self.cfg.set(key.lower(), val)

    def load(self) -> Any:
        """Return the WSGI application."""
        return self.application


def initialize_app(config: Config | None = None) -> Dash:
    """Create and return the Dash app instance."""
    config = config or Config()
    dmc.add_figure_templates()
    custom_stylesheets: list[str] = []
    if config.use_built_in_stylesheets:
        custom_stylesheets += [
            f"{config.url_base_pathname.rstrip('/')}/{INTERNAL_ASSETS_DIRNAME}/{css_file.name}"
            for css_file in sorted(
                Path(__file__).parent.glob(f"{INTERNAL_ASSETS_DIRNAME}/*.css")
            )
        ]
    app = Dash(
        __name__,
        external_stylesheets=dmc.styles.ALL + custom_stylesheets,
        assets_folder=config.assets_folder,
        use_pages=True,
        pages_folder=str(Path(__file__).parent / "pages"),
        title=config.title,
        url_base_pathname=config.url_base_pathname,
        **config.dash_kwargs,
    )
    app.drilldown_config = config
    app.layout = dmc.MantineProvider(
        layout(config),
        defaultColorScheme="auto",
        theme=config.theme,
    )

    @hooks.custom_data("drilldown_config")
    def drilldown_config_context(_ctx: Any) -> dict[str, Any]:
        return config.model_dump()

    server = app.server

    if config.use_built_in_stylesheets:

        @server.route(
            f"{config.url_base_pathname.rstrip('/')}/{INTERNAL_ASSETS_DIRNAME}/<path:filename>"
        )
        def serve_stylesheets(filename: str) -> Any:
            return send_from_directory(
                str(Path(__file__).parent / INTERNAL_ASSETS_DIRNAME), filename
            )

    return app


def run() -> None:
    """Run the Dash app."""
    config = Config()
    if config.debug or config.demo:
        with tempfile.TemporaryDirectory() as tmpdir:
            if config.demo:
                demo_path = str(Path(tmpdir) / "demo")
                create_synthetic_dataset(
                    output_dir=demo_path,
                )
                config.collection_paths = [str(demo_path)]
            app = initialize_app(config)
            app.run(
                debug=config.debug,
                host=config.host,
                port=config.port,
            )
    else:
        app = initialize_app(config)
        DrilldownApplication(
            app=app.server,
            gunicorn_options=dict(config.gunicorn_options),
        ).run()
