# SPDX-FileCopyrightText: 2026 Manuel Konrad
#
# SPDX-License-Identifier: MIT

"""Chart update logic for the explore page."""

from dataclasses import dataclass
from typing import Callable

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from dash import Input, Output, callback, clientside_callback
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from drilldown.constants import (
    EMPTY_FIGURE_LAYOUT,
    EXPLORE_PREFIX,
    PLOTLY_THEME_DARK,
    PLOTLY_THEME_LIGHT,
    THEME_DARK,
)

NANOSECONDS_PER_SECOND = 1_000_000_000
COLUMN_TYPES_IDX = 2


def get_timestamp_columns(columns: dict) -> list[str]:
    """Return timestamp columns from the explore store structure."""
    return columns[COLUMN_TYPES_IDX].get("timestamp", [])


@dataclass
class ChartContext:
    """Context object containing all data needed for chart generation.

    Attributes:
        df: The pandas DataFrame containing the data to visualize.
        dimensions: List of column names selected for visualization axes.
        color_column: Column name used for color encoding, or None if not set.
        categorical_color_column: Column name if color_column is categorical, else None.
        numerical_color_column: Column name if color_column is numerical, else None.
        timestamp_column: Primary timestamp column for time series ordering.
        custom_data: List of column names to include in hover data.
        theme_name: Plotly template name (e.g., 'plotly_dark', 'plotly_white').
        histtype: Histogram type ('1D', '2D', or '2D_contour').
        histfunc: Histogram aggregation function (e.g., 'count', 'sum', 'avg').
        barmode: Bar chart mode for histograms ('relative', 'overlay', 'group').
        histlog: Whether to use log scale for histogram y-axis.
        histbins: Number of bins for histogram (0 for auto).
        kmeans_clusters: Number of clusters for K-means clustering.
        kmeans_pca: Number of PCA components for K-means clustering.
        columns: Dictionary containing column type mappings from explore_store.
    """

    df: pd.DataFrame
    dimensions: list[str]
    color_column: str | None
    categorical_color_column: str | None
    numerical_color_column: str | None
    timestamp_column: str | None
    custom_data: list[str]
    theme_name: str
    histtype: str
    histfunc: str
    barmode: str
    histlog: bool
    histbins: int
    kmeans_clusters: int
    kmeans_pca: int
    columns: dict
    color_discrete_sequence = px.colors.qualitative.Plotly
    color_continuous_scale = px.colors.sequential.Viridis


def filter_numerical_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    """Filter to columns that are numeric and have at least one non-NaN value."""
    valid_columns = []
    for col in columns:
        if col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                non_nan_count = df[col].notna().sum()
                if non_nan_count > 0:
                    valid_columns.append(col)
    return valid_columns


def convert_to_numerical(
    df: pd.DataFrame, columns: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    """Convert compatible columns to numerical format, returning valid column names."""
    df_processed = df.copy()
    valid_columns = []

    for col in columns:
        if col not in df_processed.columns:
            continue

        if pd.api.types.is_bool_dtype(df_processed[col]):
            df_processed[col] = df_processed[col].astype(float)

        if pd.api.types.is_numeric_dtype(df_processed[col]):
            if df_processed[col].notna().sum() > 0:
                valid_columns.append(col)
            continue

        try:
            converted = pd.to_numeric(df_processed[col], errors="coerce")
            if converted.notna().sum() > 0:
                df_processed[col] = converted
                valid_columns.append(col)
        except (ValueError, TypeError):
            pass

    return df_processed, valid_columns


def convert_timestamp_to_unix(series: pd.Series) -> pd.Series:
    """Convert a pandas Series of timestamps to unix time in seconds."""
    converted = pd.to_datetime(series, errors="coerce")
    unix_time = (converted.astype("int64") // NANOSECONDS_PER_SECOND).astype("float64")
    unix_time[converted.isna()] = float("nan")
    return unix_time


def create_scatter_chart(ctx: ChartContext) -> go.Figure:
    """Create a scatter plot or scatter matrix chart."""
    if len(ctx.dimensions) == 1:
        # Single dimension: use index for x-axis
        figure = px.scatter(
            ctx.df.reset_index(),
            x="index",
            y=ctx.dimensions[0],
            color=ctx.color_column,
            template=ctx.theme_name,
            custom_data=ctx.custom_data,
            opacity=0.7,
            color_continuous_scale=ctx.color_continuous_scale,
            color_discrete_sequence=ctx.color_discrete_sequence,
        )
        figure.update_xaxes(title_text="Index")
    elif len(ctx.dimensions) == 2:
        x_column = ctx.dimensions[0]
        y_column = ctx.dimensions[1]
        figure = px.scatter(
            ctx.df,
            x=x_column,
            y=y_column,
            color=ctx.color_column,
            template=ctx.theme_name,
            custom_data=ctx.custom_data,
            opacity=0.7,
            color_continuous_scale=ctx.color_continuous_scale,
            color_discrete_sequence=ctx.color_discrete_sequence,
        )
    else:
        figure = px.scatter_matrix(
            ctx.df,
            dimensions=ctx.dimensions,
            color=ctx.color_column,
            template=ctx.theme_name,
            custom_data=ctx.custom_data,
            opacity=0.7,
            color_continuous_scale=ctx.color_continuous_scale,
            color_discrete_sequence=ctx.color_discrete_sequence,
        )
        figure.update_traces(diagonal_visible=False, showupperhalf=False)
    return figure


def create_line_chart(ctx: ChartContext) -> go.Figure | None:
    """Create a time series scatter chart with proper ordering."""
    if len(ctx.dimensions) == 0:
        return None

    # Sort by timestamp for proper time series ordering
    df_sorted = ctx.df.copy()
    if ctx.timestamp_column and ctx.timestamp_column in df_sorted.columns:
        df_sorted = df_sorted.sort_values(by=ctx.timestamp_column)

    # Determine dimensions to plot (exclude timestamp if it's in dimensions)
    timestamp_cols = get_timestamp_columns(ctx.columns)
    dims_to_plot = [dim for dim in ctx.dimensions if dim not in timestamp_cols]

    # If no dimensions to plot (only timestamp selected), return None
    if not dims_to_plot:
        return None

    figure = make_subplots(
        rows=len(dims_to_plot),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
    )

    for i, dim in enumerate(dims_to_plot):
        scatter_figure = px.scatter(
            df_sorted,
            x=ctx.timestamp_column,
            y=dim,
            color=ctx.color_column,
            custom_data=ctx.custom_data,
            opacity=0.7,
            color_discrete_sequence=ctx.color_discrete_sequence,
            color_continuous_scale=ctx.color_continuous_scale,
        )
        for trace in scatter_figure.data:
            trace.showlegend = i == 0
            figure.add_trace(
                trace,
                row=i + 1,
                col=1,
            )
        figure.update_layout(
            coloraxis=scatter_figure.layout.coloraxis,
        )
        figure.update_yaxes(title_text=dim, row=i + 1, col=1)

    figure.update_xaxes(
        title_text=ctx.timestamp_column or "index",
        row=len(dims_to_plot),
        col=1,
    )
    return figure


def create_histogram_1d(ctx: ChartContext) -> go.Figure:
    """Create a 1D histogram chart."""
    offset = int(not ctx.histfunc.startswith("count"))
    if ctx.histfunc != "count":
        # Remove non-numerical dimensions for non-count histograms
        ctx.dimensions = filter_numerical_columns(ctx.df, ctx.dimensions)

    figure = make_subplots(
        rows=len(ctx.dimensions) - offset,
        cols=1,
        shared_xaxes=ctx.histfunc == "count_shared_x",
        vertical_spacing=0.02,
    )

    for i, dim in enumerate(ctx.dimensions[offset:]):
        # Determine nbins: use histbins if > 0, otherwise let plotly auto-determine
        nbins_param = ctx.histbins if ctx.histbins > 0 else None

        hist_figure = px.histogram(
            ctx.df,
            x=dim if ctx.histfunc.startswith("count") else ctx.dimensions[0],
            y=None if ctx.histfunc.startswith("count") else dim,
            histfunc=ctx.histfunc.split("_")[0],
            color=ctx.categorical_color_column,
            nbins=nbins_param,
            opacity=0.7,
            color_discrete_sequence=ctx.color_discrete_sequence,
            log_y=ctx.histlog,
        )
        for trace in hist_figure.data:
            trace.showlegend = i == 0
            figure.add_trace(
                trace,
                row=i + 1,
                col=1,
            )
        figure.update_yaxes(
            title_text=f"{ctx.histfunc.split('_')[0]}({dim})",
            row=i + 1,
            col=1,
            type="log" if ctx.histlog else "linear",
        )

    figure.update_xaxes(
        title_text="value" if ctx.histfunc.startswith("count") else ctx.dimensions[0],
        row=len(ctx.dimensions) - offset,
        col=1,
    )
    figure.update_layout(barmode=ctx.barmode)
    return figure


def create_histogram_2d(ctx: ChartContext) -> go.Figure:
    """Create a 2D histogram/density chart."""
    # Filter to only numerical dimensions for 2D histograms
    numerical_dims = filter_numerical_columns(ctx.df, ctx.dimensions)

    if len(numerical_dims) < 2:
        # Need at least 2 numerical dimensions for 2D histogram
        figure = go.Figure()
        figure.add_annotation(
            text="2D histogram requires at least 2 numerical dimensions",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14),
        )
        figure.update_layout(template=ctx.theme_name)
        return figure

    figure = make_subplots(
        rows=len(numerical_dims) - 1,
        cols=len(numerical_dims) - 1,
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.02,
        horizontal_spacing=0.02,
    )

    for i, dim1 in enumerate(numerical_dims[:-1]):
        for j, dim2 in enumerate(numerical_dims[i + 1 :]):
            if ctx.histtype == "2D":
                hist_figure = px.density_heatmap(
                    ctx.df,
                    x=dim1,
                    y=dim2,
                    z=ctx.numerical_color_column
                    if ctx.histfunc.split("_")[0] != "count"
                    else None,
                    histfunc=ctx.histfunc.split("_")[0],
                )
            elif ctx.histtype == "2D_contour":
                hist_figure = px.density_contour(
                    ctx.df,
                    x=dim1,
                    y=dim2,
                    z=ctx.numerical_color_column
                    if ctx.histfunc.split("_")[0] != "count"
                    else None,
                    histfunc=ctx.histfunc.split("_")[0],
                )
                hist_figure.update_traces(contours_coloring="fill")
            else:
                continue

            for trace in hist_figure.data:
                figure.add_trace(
                    trace,
                    row=i + j + 1,
                    col=i + 1,
                )

            if j == len(numerical_dims[i + 1 :]) - 1:
                figure.update_xaxes(title_text=dim1, row=i + j + 1, col=i + 1)
            if i == 0:
                figure.update_yaxes(title_text=dim2, row=i + j + 1, col=i + 1)

    return figure


def create_histogram_chart(ctx: ChartContext) -> go.Figure:
    """Create histogram chart (1D or 2D based on histtype)."""
    if ctx.histtype == "1D":
        return create_histogram_1d(ctx)
    return create_histogram_2d(ctx)


def create_box_chart(ctx: ChartContext) -> go.Figure:
    """Create a box plot chart."""
    # Filter to only numerical dimensions first
    numerical_dims = filter_numerical_columns(ctx.df, ctx.dimensions)

    if not numerical_dims:
        return go.Figure(layout=EMPTY_FIGURE_LAYOUT)

    figure = make_subplots(
        rows=len(numerical_dims),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
    )

    for i, dim in enumerate(numerical_dims):
        category_orders = {}
        if ctx.color_column and ctx.color_column in ctx.df.columns:
            try:
                category_orders = {
                    ctx.color_column: sorted(ctx.df[ctx.color_column].unique())
                }
            except TypeError:
                # Handle non-sortable types
                category_orders = {}

        box_figure = px.box(
            ctx.df,
            x=ctx.categorical_color_column,
            y=dim,
            color=ctx.categorical_color_column,
            template=ctx.theme_name,
            custom_data=ctx.custom_data,
            category_orders=category_orders,
            color_discrete_sequence=ctx.color_discrete_sequence,
        )
        for trace in box_figure.data:
            figure.add_trace(
                trace,
                row=i + 1,
                col=1,
            )
        figure.update_yaxes(title_text=dim, row=i + 1, col=1)

    figure.update_xaxes(
        title_text=ctx.color_column or "", row=len(numerical_dims), col=1
    )
    figure.update_layout(showlegend=False)
    return figure


def cast_color_column_to_numeric(
    df: pd.DataFrame, ctx: ChartContext
) -> tuple[pd.DataFrame, str | None]:
    """Cast the color column to numeric if it's categorical."""
    color_column = ctx.numerical_color_column
    if ctx.categorical_color_column:
        # Convert if boolean
        if pd.api.types.is_bool_dtype(df[ctx.categorical_color_column]):
            df[ctx.categorical_color_column] = df[ctx.categorical_color_column].astype(
                float
            )
            color_column = ctx.categorical_color_column
    return df, color_column


def create_parallel_chart(ctx: ChartContext) -> go.Figure:
    """Create a parallel coordinates chart."""
    # Filter to only numerical columns for parallel coordinates
    numerical_dims = filter_numerical_columns(ctx.df, ctx.dimensions)
    if not numerical_dims:
        return go.Figure(layout=EMPTY_FIGURE_LAYOUT)

    df, color_column = cast_color_column_to_numeric(ctx.df, ctx)

    fig = px.parallel_coordinates(
        df,
        dimensions=numerical_dims,
        color=color_column,
        template=ctx.theme_name,
        color_continuous_scale=ctx.color_continuous_scale,
    )

    fig.update_layout(
        margin={
            "t": 70,
            "b": 30,
            "l": 40,
            "r": 40,
        },
    )
    return fig


def create_cycle_plot(ctx: ChartContext) -> go.Figure:
    """Create a cycle plot using timestamp dimensions with shared axis range."""
    timestamp_dims = [
        dim for dim in ctx.dimensions if dim in get_timestamp_columns(ctx.columns)
    ]
    if not timestamp_dims:
        return go.Figure(layout=EMPTY_FIGURE_LAYOUT)

    df = ctx.df.copy()
    converted_dims: list[str] = []

    for dim in timestamp_dims:
        if dim not in df.columns:
            continue

        unix_time = convert_timestamp_to_unix(df[dim])
        df[dim] = unix_time
        converted_dims.append(dim)

    if not converted_dims:
        return go.Figure(layout=EMPTY_FIGURE_LAYOUT)

    min_values = []
    max_values = []
    for dim in converted_dims:
        cleaned = df[dim].dropna()
        if cleaned.empty:
            continue
        min_values.append(cleaned.min())
        max_values.append(cleaned.max())

    if not min_values or not max_values:
        return go.Figure(layout=EMPTY_FIGURE_LAYOUT)

    global_min = min(min_values)
    global_max = max(max_values)

    df, color_column = cast_color_column_to_numeric(df, ctx)
    fig = px.parallel_coordinates(
        df,
        dimensions=converted_dims,
        color=color_column,
        template=ctx.theme_name,
        color_continuous_scale=ctx.color_continuous_scale,
    )

    if fig.data:
        for dimension in fig.data[0].dimensions:
            dimension.range = [global_min, global_max]

    fig.update_layout(
        margin={
            "t": 70,
            "b": 30,
            "l": 40,
            "r": 40,
        },
    )
    return fig


def create_cluster_chart(ctx: ChartContext) -> go.Figure:
    """Create PCA + t-SNE visualization with KMeans clustering."""
    df_processed, numerical_dims = convert_to_numerical(ctx.df, ctx.dimensions)

    if len(numerical_dims) < 2:
        figure = go.Figure()
        figure.add_annotation(
            text="Need at least 2 numerical dimensions for clustering",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14),
        )
        return figure

    X = df_processed[numerical_dims].copy()
    X = X.fillna(X.median()).fillna(0)

    n_samples = len(X)
    if n_samples < 5:
        figure = go.Figure()
        figure.add_annotation(
            text="Need at least 5 samples for clustering analysis",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14),
        )
        return figure

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_components_pca = min(ctx.kmeans_pca, len(numerical_dims), n_samples - 1)
    if n_components_pca >= 2:
        pca = PCA(n_components=n_components_pca, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
    else:
        X_pca = X_scaled

    perplexity = max(2, min(30, n_samples - 1))

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    X_tsne = tsne.fit_transform(X_pca)

    viz_df = ctx.df.copy().reset_index(drop=True)
    viz_df["t-SNE-d1"] = X_tsne[:, 0]
    viz_df["t-SNE-d2"] = X_tsne[:, 1]

    custom_data = [col for col in ctx.custom_data if col in viz_df.columns]

    if ctx.color_column and ctx.color_column in ctx.df.columns:
        color_label = ctx.color_column
    else:
        n_clusters = min(ctx.kmeans_clusters, n_samples // 3)
        n_clusters = max(2, n_clusters)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        color_data = [f"Cluster {i}" for i in cluster_labels]
        color_label = "Cluster"
        viz_df[color_label] = color_data

    figure = px.scatter(
        viz_df,
        x="t-SNE-d1",
        y="t-SNE-d2",
        color=color_label,
        template=ctx.theme_name,
        color_discrete_sequence=ctx.color_discrete_sequence,
        custom_data=custom_data,
        opacity=0.7,
    )

    figure.update_layout(
        xaxis_title="t-SNE Dimension 1",
        yaxis_title="t-SNE Dimension 2",
    )

    return figure


ChartFunction = Callable[[ChartContext], go.Figure | None]

CHART_REGISTRY: dict[str, ChartFunction] = {
    "scatter": create_scatter_chart,
    "line": create_line_chart,
    "hist": create_histogram_chart,
    "box": create_box_chart,
    "parallel": create_parallel_chart,
    "cycle": create_cycle_plot,
    "cluster": create_cluster_chart,
}


def register_chart_type(name: str, chart_func: ChartFunction) -> None:
    """Register a new chart type with the given name."""
    CHART_REGISTRY[name] = chart_func


def get_available_chart_types() -> list[str]:
    """Get list of available chart types."""
    return list(CHART_REGISTRY.keys())


clientside_callback(
    """
    (fig, leftTab, rightTab) => {
        window.dispatchEvent(new Event('resize'));
        return null;
    }
    """,
    Input(f"{EXPLORE_PREFIX}-graph", "figure"),
    Input(f"{EXPLORE_PREFIX}-left-tabs", "value"),
    Input(f"{EXPLORE_PREFIX}-right-tabs", "value"),
)


@callback(
    Output(f"{EXPLORE_PREFIX}-graph", "figure"),
    Input(f"{EXPLORE_PREFIX}-ag-grid", "virtualRowData"),
    Input(f"{EXPLORE_PREFIX}-store", "data"),
    Input(f"{EXPLORE_PREFIX}-chart-select", "value"),
    Input(f"{EXPLORE_PREFIX}-color-field-select", "value"),
    Input(f"{EXPLORE_PREFIX}-dimensions-select", "value"),
    Input(f"{EXPLORE_PREFIX}-histtype-select", "value"),
    Input(f"{EXPLORE_PREFIX}-histfunc-select", "value"),
    Input(f"{EXPLORE_PREFIX}-barmode-select", "value"),
    Input(f"{EXPLORE_PREFIX}-hist-yscale", "value"),
    Input(f"{EXPLORE_PREFIX}-histbins-input", "value"),
    Input(f"{EXPLORE_PREFIX}-kmeans-nclusters", "value"),
    Input(f"{EXPLORE_PREFIX}-pca-ndim", "value"),
    Input("theme-toggle", "value"),
    Input("burger", "opened"),
)
def update_chart(
    virtual_row_data,
    explore_store,
    chart_select,
    color_column,
    dimensions,
    histtype,
    histfunc,
    barmode,
    histlog,
    histbins,
    kmeans_clusters,
    kmeans_pca,
    theme,
    sidebar_open,
):
    """Main callback to update the chart based on user selections."""
    theme_name = PLOTLY_THEME_DARK if theme == THEME_DARK else PLOTLY_THEME_LIGHT

    if explore_store and virtual_row_data and chart_select and len(dimensions) > 0:
        columns = explore_store["columns"]
        custom_data = columns[COLUMN_TYPES_IDX]["custom_data"]
        timestamp_columns = get_timestamp_columns(columns)
        df = pd.DataFrame(virtual_row_data)

        # Determine timestamp column
        timestamp_column = (
            dimensions[0] if dimensions[0] in timestamp_columns else columns[1]
        )

        # Determine color column types
        categorical_color_column = (
            color_column
            if color_column in columns[COLUMN_TYPES_IDX]["categorical"]
            else None
        )
        numerical_color_column = (
            color_column
            if color_column in columns[COLUMN_TYPES_IDX]["numerical"]
            else None
        )

        # Create chart context
        ctx = ChartContext(
            df=df,
            dimensions=dimensions,
            color_column=color_column,
            categorical_color_column=categorical_color_column,
            numerical_color_column=numerical_color_column,
            timestamp_column=timestamp_column,
            custom_data=custom_data,
            theme_name=theme_name,
            histtype=histtype,
            histfunc=histfunc,
            barmode=barmode,
            histlog=(histlog == "log") if histlog is not None else False,
            histbins=histbins if histbins is not None else 0,
            kmeans_clusters=kmeans_clusters if kmeans_clusters is not None else 3,
            kmeans_pca=kmeans_pca if kmeans_pca is not None else 2,
            columns=columns,
        )

        # Get chart function from registry
        chart_func = CHART_REGISTRY.get(chart_select)
        if chart_func:
            figure = chart_func(ctx)
            if figure is None:
                figure = go.Figure(layout=EMPTY_FIGURE_LAYOUT)
        else:
            figure = go.Figure(layout=EMPTY_FIGURE_LAYOUT)
    else:
        figure = go.Figure(layout=EMPTY_FIGURE_LAYOUT)

    # Apply common layout settings
    template = pio.templates[theme_name]
    if theme_name == PLOTLY_THEME_DARK:
        figure.update_layout(
            paper_bgcolor="#171b1f",
            plot_bgcolor="#171b1f",
        )
    else:
        figure.update_layout(
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
        )
    return go.Figure(figure).update_layout(
        margin=dict(
            l=figure.layout.margin.l or 20,
            r=figure.layout.margin.r or 20,
            t=figure.layout.margin.t or 20,
            b=figure.layout.margin.b or 20,
        ),
        template=template,
        clickmode="event",
        dragmode=None,
    )
