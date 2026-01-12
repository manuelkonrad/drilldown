# SPDX-FileCopyrightText: 2026 Manuel Konrad
#
# SPDX-License-Identifier: MIT

INTERNAL_ASSETS_DIRNAME = "internal_assets"
SIDEBAR_WIDTH_OPEN = 250
SIDEBAR_WIDTH_CLOSED = 51
SIDEBAR_ICON_SIZE = 28
SIDEBAR_FONT_SIZE = 16
BUTTON_ICON_SIZE = 24
BUTTON_SIZE = 36
HEADER_HEIGHT = 50
TOGGLE_ICON_SIZE = 20
DATA_POPOVER_WIDTH = 350
PAGE_CONTAINER_HEIGHT = f"calc( 100dvh - var(--app-shell-header-offset, 0rem) - var(--app-shell-footer-offset, 0rem) - {HEADER_HEIGHT}px - 6px )"
CARD_SIZE = 400
SELECT_MAX_DROPDOWN_HEIGHT = 400
TAB_LIST_OFFSET = "(2.25rem * var(--mantine-scale))"

# Theme constants
THEME_DARK = "dark"
THEME_LIGHT = "light"
PLOTLY_THEME_DARK = "plotly_dark"
PLOTLY_THEME_LIGHT = "plotly_white"
PLOTLY_DARK_BG_COLOR = "#171b1f"
HEADER_INPUT_STYLE = {
    "input": {
        "color": "var(--mantine-color-dark-0)",
        "backgroundColor": "var(--mantine-color-dark-8)",
        "border": "1px solid var(--mantine-color-dark-4)",
    },
}
SELECT_LABEL_STYLE = {
    "label": {
        "margin-left": "3px",
        "white-space": "nowrap",
        "overflow": "hidden",
        "text-overflow": "ellipsis",
    }
}

# Page prefixes
EXPLORE_PREFIX = "explore"
ANALYZE_PREFIX = "analyze"
MONITOR_PREFIX = "monitor"

# explore constants
DEFAULT_NO_DATASET_MESSAGE = "No dataset selected."
DARK_THEME_CLASS = "ag-theme-quartz-dark ag-theme-drilldown-dark"
LIGHT_THEME_CLASS = "ag-theme-quartz ag-theme-drilldown"
SAMPLE_INFO_VALUE = "__sample_info__"
SAMPLE_INFO_LABEL = "Sample Values"
DEFAULT_MAX_DIMENSIONS = 3

# Sample view constants
PLOT_TYPE_IMAGE = "image"
PLOT_TYPE_CURVE = "curve"
PLOT_TYPE_INFO = "info"

IMAGE_FORMATS = ["image", "bmp", "jpeg", "jpg", "png"]
CURVE_FORMATS = ["json", "yaml"]

TRIGGER_TYPE_AG = "ag"
TRIGGER_TYPE_CLICK = "click"
TRIGGER_TYPE_DEFAULT = "default"

# Graph configuration
GRAPH_CONFIG = {
    "autosizable": True,
    "frameMargins": 0,
    "responsive": True,
    "modeBarButtonsToRemove": ["select2d", "lasso2d"],
}

GRAPH_STYLE = {"height": "100%", "width": "100%"}
EMPTY_FIGURE_LAYOUT = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "xaxis": {
        "visible": False,
    },
    "yaxis": {
        "visible": False,
    },
}

# Analyze page constants
N_ESTIMATORS = 100
RANDOM_STATE = 42
MAX_SHAP_SAMPLES = 100
WHAT_IF_PERCENTILE_LOW = 5
WHAT_IF_PERCENTILE_HIGH = 95
WHAT_IF_N_POINTS = 50
MAX_EBM_LOCAL_SAMPLES = 20
MAX_EBM_LOCAL_DISPLAY_SAMPLES = 5

# Monitor page constants
MIN_WINDOW_SAMPLES = 5
MIN_ROLLING_PERIODS = 3
DEFAULT_ROLLING_WINDOW = 3
DEFAULT_STEP_DAYS = 1
DEFAULT_REFERENCE_DAYS_START = 14
DEFAULT_REFERENCE_DAYS_END = 8
