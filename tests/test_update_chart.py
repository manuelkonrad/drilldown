# SPDX-FileCopyrightText: 2026 Manuel Konrad
#
# SPDX-License-Identifier: MIT

"""Tests for the update_chart module."""

import pandas as pd
import plotly.graph_objects as go
import pytest

from drilldown.pages.explore.update_chart import (
    CHART_REGISTRY,
    COLUMN_TYPES_IDX,
    ChartContext,
    convert_timestamp_to_unix,
    convert_to_numerical,
    create_box_chart,
    create_cluster_chart,
    create_cycle_plot,
    create_histogram_1d,
    create_histogram_2d,
    create_line_chart,
    create_parallel_chart,
    create_scatter_chart,
    filter_numerical_columns,
    get_available_chart_types,
    register_chart_type,
)


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
            ),
            "value1": [10.5, 20.3, 15.7, 25.1, 30.2],
            "value2": [5.0, 10.0, 7.5, 12.5, 15.0],
            "value3": [100.0, 200.0, 150.0, 250.0, 300.0],
            "category": ["A", "B", "A", "B", "A"],
        }
    )


@pytest.fixture
def sample_context(sample_df):
    """Create a sample ChartContext for testing."""
    return ChartContext(
        df=sample_df,
        dimensions=["value1", "value2"],
        color_column="category",
        categorical_color_column="category",
        numerical_color_column=None,
        timestamp_column="timestamp",
        custom_data=["timestamp", "value1", "value2", "category"],
        theme_name="plotly_dark",
        histtype="1D",
        histfunc="count",
        barmode="relative",
        histlog=False,
        histbins=0,
        kmeans_clusters=3,
        kmeans_pca=2,
        columns={
            COLUMN_TYPES_IDX: {
                "timestamp": ["timestamp"],
                "categorical": ["category"],
                "numerical": ["value1", "value2", "value3"],
            }
        },
    )


# =============================================================================
# Chart Registry Tests
# =============================================================================
class TestChartRegistry:
    """Tests for the chart registry."""

    def test_chart_registry_has_all_types(self):
        """Test that all expected chart types are registered."""
        expected_types = [
            "scatter",
            "line",
            "hist",
            "box",
            "parallel",
            "cycle",
            "cluster",
        ]
        for chart_type in expected_types:
            assert chart_type in CHART_REGISTRY

    def test_get_available_chart_types(self):
        """Test getting available chart types."""
        types = get_available_chart_types()
        assert "scatter" in types
        assert "cluster" in types
        assert len(types) >= 7

    def test_register_chart_type(self):
        """Test registering a new chart type."""

        def custom_chart(ctx):
            return go.Figure()

        register_chart_type("custom_test", custom_chart)
        assert "custom_test" in CHART_REGISTRY
        assert CHART_REGISTRY["custom_test"] == custom_chart
        # Clean up
        del CHART_REGISTRY["custom_test"]


# =============================================================================
# Column Filtering Tests
# =============================================================================
class TestColumnFiltering:
    """Tests for column filtering utilities."""

    def test_filter_numerical_columns(self, sample_df):
        """Test filtering numerical columns."""
        columns = ["value1", "value2", "category"]
        result = filter_numerical_columns(sample_df, columns)
        assert "value1" in result
        assert "value2" in result
        assert "category" not in result

    def test_filter_numerical_columns_with_nan(self):
        """Test filtering columns with NaN values."""
        df = pd.DataFrame(
            {
                "valid": [1.0, 2.0, 3.0],
                "all_nan": [float("nan")] * 3,
                "partial_nan": [1.0, float("nan"), 3.0],
            }
        )
        result = filter_numerical_columns(df, ["valid", "all_nan", "partial_nan"])
        assert "valid" in result
        assert "all_nan" not in result
        assert "partial_nan" in result

    def test_convert_to_numerical_already_numeric(self, sample_df):
        """Test conversion of already numeric columns."""
        df_result, valid_cols = convert_to_numerical(sample_df, ["value1", "value2"])
        assert "value1" in valid_cols
        assert "value2" in valid_cols

    def test_convert_to_numerical_string_numbers(self):
        """Test conversion of string numbers to numeric."""
        df = pd.DataFrame({"str_num": ["1", "2", "3"], "non_num": ["a", "b", "c"]})
        df_result, valid_cols = convert_to_numerical(df, ["str_num", "non_num"])
        assert "str_num" in valid_cols
        assert "non_num" not in valid_cols

    def test_convert_to_numerical_boolean_columns(self):
        """Test conversion of boolean columns to float."""
        df = pd.DataFrame(
            {
                "bool_col": [True, False, True, False],
                "value": [1.0, 2.0, 3.0, 4.0],
                "category": ["A", "B", "A", "B"],
            }
        )
        df_result, valid_cols = convert_to_numerical(
            df, ["bool_col", "value", "category"]
        )
        assert "bool_col" in valid_cols
        assert "value" in valid_cols
        assert "category" not in valid_cols
        # Verify boolean was converted to float
        assert df_result["bool_col"].dtype == float
        assert df_result["bool_col"].tolist() == [1.0, 0.0, 1.0, 0.0]


# =============================================================================
# Scatter Chart Tests
# =============================================================================
class TestScatterChart:
    """Tests for scatter chart creation."""

    def test_scatter_chart_2d(self, sample_context):
        """Test creating a 2D scatter plot."""
        figure = create_scatter_chart(sample_context)
        assert isinstance(figure, go.Figure)
        assert len(figure.data) > 0

    def test_scatter_chart_matrix(self, sample_context):
        """Test creating a scatter matrix with >2 dimensions."""
        sample_context.dimensions = ["value1", "value2", "value3"]
        figure = create_scatter_chart(sample_context)
        assert isinstance(figure, go.Figure)


# =============================================================================
# Line Chart Tests
# =============================================================================
class TestLineChart:
    """Tests for line chart creation."""

    def test_line_chart_basic(self, sample_context):
        """Test creating a basic line chart."""
        sample_context.dimensions = ["timestamp", "value1", "value2"]
        sample_context.columns[COLUMN_TYPES_IDX]["timestamp"] = ["timestamp"]
        figure = create_line_chart(sample_context)
        assert isinstance(figure, go.Figure)
        assert len(figure.data) > 0

    def test_line_chart_insufficient_dimensions(self, sample_context):
        """Test line chart with insufficient dimensions."""
        # Only timestamp dimension - should return None
        sample_context.dimensions = ["timestamp"]
        sample_context.columns[COLUMN_TYPES_IDX]["timestamp"] = ["timestamp"]
        figure = create_line_chart(sample_context)
        assert figure is None

        # No dimensions - should return None
        sample_context.dimensions = []
        figure = create_line_chart(sample_context)
        assert figure is None

    def test_line_chart_single_numerical_dimension(self, sample_context):
        """Test line chart with single numerical dimension."""
        sample_context.dimensions = ["value1"]
        sample_context.columns[COLUMN_TYPES_IDX]["timestamp"] = ["timestamp"]
        figure = create_line_chart(sample_context)
        # Should create a chart with timestamp on x-axis and value1 on y-axis
        assert isinstance(figure, go.Figure)
        assert len(figure.data) > 0

    def test_line_chart_sorts_by_timestamp(self, sample_context):
        """Test that line chart sorts data by timestamp."""
        sample_context.df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2024-01-03",
                        "2024-01-01",
                        "2024-01-05",
                        "2024-01-02",
                        "2024-01-04",
                    ]
                ),
                "value1": [15.7, 10.5, 30.2, 20.3, 25.1],
                "value2": [7.5, 5.0, 15.0, 10.0, 12.5],
            }
        )
        sample_context.dimensions = ["timestamp", "value1"]
        sample_context.columns[COLUMN_TYPES_IDX]["timestamp"] = ["timestamp"]
        sample_context.custom_data = ["timestamp", "value1", "value2"]
        sample_context.color_column = None
        sample_context.categorical_color_column = None

        figure = create_line_chart(sample_context)
        assert isinstance(figure, go.Figure)

    def test_line_chart_with_color_column(self, sample_context):
        """Test that line chart uses color column for coloring points."""
        sample_context.dimensions = ["timestamp", "value1"]
        sample_context.columns[COLUMN_TYPES_IDX]["timestamp"] = ["timestamp"]
        sample_context.color_column = "category"
        sample_context.categorical_color_column = "category"

        figure = create_line_chart(sample_context)
        assert isinstance(figure, go.Figure)
        # With color column, we should have traces for each category
        assert len(figure.data) > 0


# =============================================================================
# Histogram Chart Tests
# =============================================================================
class TestHistogramChart:
    """Tests for histogram chart creation."""

    def test_histogram_1d_basic(self, sample_context):
        """Test creating a 1D histogram."""
        figure = create_histogram_1d(sample_context)
        assert isinstance(figure, go.Figure)

    def test_histogram_2d_basic(self, sample_context):
        """Test creating a 2D histogram."""
        sample_context.histtype = "2D"
        figure = create_histogram_2d(sample_context)
        assert isinstance(figure, go.Figure)

    def test_histogram_2d_contour(self, sample_context):
        """Test creating a 2D contour histogram."""
        sample_context.histtype = "2D_contour"
        figure = create_histogram_2d(sample_context)
        assert isinstance(figure, go.Figure)

    def test_histogram_2d_filters_non_numeric(self, sample_context):
        """Test that 2D histogram filters non-numeric dimensions."""
        sample_context.histtype = "2D"
        sample_context.dimensions = ["value1", "value2", "category"]
        figure = create_histogram_2d(sample_context)
        assert isinstance(figure, go.Figure)
        # Should work with only numerical dimensions


# =============================================================================
# Box Chart Tests
# =============================================================================
class TestBoxChart:
    """Tests for box chart creation."""

    def test_box_chart_basic(self, sample_context):
        """Test creating a basic box chart."""
        figure = create_box_chart(sample_context)
        assert isinstance(figure, go.Figure)

    def test_box_chart_filters_non_numeric(self, sample_context):
        """Test that box chart filters non-numeric dimensions."""
        sample_context.dimensions = ["value1", "category"]
        figure = create_box_chart(sample_context)
        assert isinstance(figure, go.Figure)


# =============================================================================
# Parallel Coordinates Tests
# =============================================================================
class TestParallelChart:
    """Tests for parallel coordinates chart creation."""

    def test_parallel_chart_basic(self, sample_context):
        """Test creating a parallel coordinates chart."""
        sample_context.numerical_color_column = "value1"
        figure = create_parallel_chart(sample_context)
        assert isinstance(figure, go.Figure)

    def test_parallel_chart_filters_non_numeric(self, sample_context):
        """Test that parallel chart filters non-numeric dimensions."""
        sample_context.dimensions = ["value1", "value2", "category"]
        figure = create_parallel_chart(sample_context)
        assert isinstance(figure, go.Figure)


class TestCycleChart:
    """Tests for cycle plot creation."""

    def test_cycle_chart_uses_timestamp_dimensions(self, sample_context):
        """Test creating a cycle plot from timestamp dimensions."""
        sample_context.dimensions = ["timestamp", "value1"]
        sample_context.columns[COLUMN_TYPES_IDX]["timestamp"] = ["timestamp"]
        sample_context.numerical_color_column = "value1"

        figure = create_cycle_plot(sample_context)

        unix_series = convert_timestamp_to_unix(sample_context.df["timestamp"])
        expected_min = float(unix_series.min(skipna=True))
        expected_max = float(unix_series.max(skipna=True))
        expected_range = (expected_min, expected_max)

        assert isinstance(figure, go.Figure)
        assert figure.data
        assert len(figure.data[0].dimensions) == 1
        assert list(figure.data[0].dimensions[0].range) == list(expected_range)

    def test_cycle_chart_ignores_non_timestamp_dimensions(self, sample_context):
        """Test that cycle plot gracefully handles missing timestamp dimensions."""
        sample_context.dimensions = ["value1", "value2"]
        sample_context.columns[COLUMN_TYPES_IDX]["timestamp"] = ["timestamp"]

        figure = create_cycle_plot(sample_context)
        assert isinstance(figure, go.Figure)


# =============================================================================
# Cluster Chart Tests
# =============================================================================
class TestClusterChart:
    """Tests for cluster (PCA/t-SNE) chart creation."""

    def test_cluster_chart_basic(self, sample_context):
        """Test creating a basic cluster chart."""
        sample_context.dimensions = ["value1", "value2", "value3"]
        sample_context.df = pd.DataFrame(
            {
                "value1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "value2": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                "value3": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                "category": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
            }
        )
        figure = create_cluster_chart(sample_context)
        assert isinstance(figure, go.Figure)

    def test_cluster_chart_with_color_column(self, sample_context):
        """Test cluster chart uses provided color column."""
        sample_context.dimensions = ["value1", "value2", "value3"]
        sample_context.color_column = "category"
        sample_context.df = pd.DataFrame(
            {
                "value1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "value2": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                "value3": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                "category": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
            }
        )
        figure = create_cluster_chart(sample_context)
        assert isinstance(figure, go.Figure)

    def test_cluster_chart_auto_clustering(self, sample_context):
        """Test cluster chart applies automatic KMeans when no color column."""
        sample_context.dimensions = ["value1", "value2", "value3"]
        sample_context.color_column = None
        sample_context.df = pd.DataFrame(
            {
                "value1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "value2": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                "value3": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            }
        )
        figure = create_cluster_chart(sample_context)
        assert isinstance(figure, go.Figure)
        # Check that cluster coloring was applied
        assert len(figure.data) > 0

    def test_cluster_chart_insufficient_dimensions(self, sample_context):
        """Test cluster chart with insufficient dimensions."""
        sample_context.dimensions = ["value1"]
        figure = create_cluster_chart(sample_context)
        assert isinstance(figure, go.Figure)
        # Should have an annotation explaining the error
        assert len(figure.layout.annotations) > 0

    def test_cluster_chart_insufficient_samples(self, sample_context):
        """Test cluster chart with insufficient samples."""
        sample_context.dimensions = ["value1", "value2"]
        sample_context.df = pd.DataFrame({"value1": [1, 2], "value2": [3, 4]})
        figure = create_cluster_chart(sample_context)
        assert isinstance(figure, go.Figure)
        # Should have an annotation explaining the error

    def test_cluster_chart_handles_nan(self, sample_context):
        """Test cluster chart handles NaN values gracefully."""
        sample_context.dimensions = ["value1", "value2"]
        sample_context.df = pd.DataFrame(
            {
                "value1": [1, 2, float("nan"), 4, 5, 6, 7, 8, 9, 10],
                "value2": [2, float("nan"), 6, 8, 10, 12, 14, 16, 18, 20],
            }
        )
        figure = create_cluster_chart(sample_context)
        assert isinstance(figure, go.Figure)
