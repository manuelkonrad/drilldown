# SPDX-FileCopyrightText: 2026 Manuel Konrad
#
# SPDX-License-Identifier: MIT

"""Tests for the monitoring drift detection functions."""

import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path for dynamic module loading in get_monitor_functions()
_SRC_PATH = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(_SRC_PATH))


# Helper to extract functions from module without triggering Dash registration
def get_monitor_functions():
    """Extract drift detection functions from monitor_page.py.

    This helper dynamically loads monitor functions to avoid triggering
    Dash page registration during test collection.

    Returns:
        Dictionary containing compute_ks_statistic, compute_rolling_drift,
        and optionally create_monitor_figure.
    """
    # Import algorithm functions from the new module
    from drilldown.pages.monitor.algorithms import (
        compute_ks_statistic,
        compute_rolling_drift,
    )

    namespace = {
        "compute_ks_statistic": compute_ks_statistic,
        "compute_rolling_drift": compute_rolling_drift,
    }

    # Try to import figure creation function, but don't fail if dash isn't available
    try:
        # Import figure creation from monitor_page.py
        monitor_path = _SRC_PATH / "drilldown" / "pages" / "monitor_page.py"
        module_code = monitor_path.read_text()

        # Execute only up to the dash.register_page call
        code_to_exec = module_code.split("dash.register_page")[0]

        # Create a namespace with required imports
        namespace.update(
            {
                "datetime": datetime,
                "np": np,
                "pd": pd,
            }
        )

        # Add plotly imports
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        namespace["px"] = px
        namespace["go"] = go
        namespace["make_subplots"] = make_subplots

        exec(code_to_exec, namespace)
    except (ImportError, ModuleNotFoundError):
        # If dash or other dependencies aren't available, skip figure creation tests
        namespace["create_monitor_figure"] = None

    return namespace


# Get the functions
_namespace = get_monitor_functions()
compute_ks_statistic = _namespace["compute_ks_statistic"]
compute_rolling_drift = _namespace["compute_rolling_drift"]
create_monitor_figure = _namespace.get("create_monitor_figure")


class TestKSStatistic:
    """Tests for Kolmogorov-Smirnov statistic calculation."""

    def test_ks_statistic_same_distribution(self):
        """Test that KS statistic is low for same distribution."""
        np.random.seed(42)
        reference = np.random.normal(0, 1, 1000)
        current = np.random.normal(0, 1, 1000)

        ks_stat, p_value = compute_ks_statistic(reference, current)

        # Same distribution should have low KS statistic
        assert ks_stat < 0.1
        # p-value should be high (no significant difference)
        assert p_value > 0.05

    def test_ks_statistic_different_distribution(self):
        """Test that KS statistic is high for different distributions."""
        np.random.seed(42)
        reference = np.random.normal(0, 1, 1000)
        current = np.random.normal(2, 1, 1000)  # Shifted mean

        ks_stat, p_value = compute_ks_statistic(reference, current)

        # Different distribution should have high KS statistic
        assert ks_stat > 0.3
        # p-value should be low (significant difference)
        assert p_value < 0.05

    def test_ks_statistic_identical_data(self):
        """Test KS statistic with identical data."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        ks_stat, p_value = compute_ks_statistic(data, data)

        # Identical data should have KS = 0
        assert ks_stat == 0.0
        assert p_value >= 0.0

    def test_ks_statistic_empty_array(self):
        """Test KS statistic handles empty arrays gracefully."""
        reference = np.array([1.0, 2.0, 3.0])
        current = np.array([])

        # Should not raise an error
        try:
            ks_stat, p_value = compute_ks_statistic(reference, current)
            # Result should be defined
            assert isinstance(ks_stat, float)
            assert isinstance(p_value, float)
        except ValueError:
            # scipy raises ValueError for empty arrays, which is acceptable
            pass


class TestRollingDrift:
    """Tests for rolling drift computation."""

    @pytest.fixture
    def sample_reference_df(self):
        """Create a sample reference DataFrame."""
        np.random.seed(42)
        dates = pd.date_range(start="2024-01-01", periods=30, freq="D")
        values = np.random.normal(10, 2, 30)
        return pd.DataFrame({"timestamp": dates, "value": values})

    @pytest.fixture
    def sample_current_df(self):
        """Create a sample current DataFrame with slight drift."""
        np.random.seed(43)
        dates = pd.date_range(start="2024-01-31", periods=70, freq="D")
        values = np.random.normal(12, 2, 70)  # Current with drift
        return pd.DataFrame({"timestamp": dates, "value": values})

    def test_rolling_drift_basic(self, sample_reference_df, sample_current_df):
        """Test basic rolling drift computation."""
        result = compute_rolling_drift(
            reference_df=sample_reference_df,
            current_df=sample_current_df,
            timestamp_col="timestamp",
            value_col="value",
            rolling_window=7,
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert "timestamp" in result.columns
        assert "drift_score" in result.columns
        assert "window_mean" in result.columns
        assert "window_std" in result.columns

    def test_rolling_drift_with_step_days(self, sample_reference_df, sample_current_df):
        """Test rolling drift with custom step size."""
        result = compute_rolling_drift(
            reference_df=sample_reference_df,
            current_df=sample_current_df,
            timestamp_col="timestamp",
            value_col="value",
            rolling_window=7,
            step_days=3,
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_rolling_drift_empty_reference(self, sample_current_df):
        """Test rolling drift with empty reference period."""
        # Empty reference DataFrame
        empty_ref = pd.DataFrame({"timestamp": [], "value": []})

        result = compute_rolling_drift(
            reference_df=empty_ref,
            current_df=sample_current_df,
            timestamp_col="timestamp",
            value_col="value",
            rolling_window=7,
        )

        # Should return empty DataFrame
        assert len(result) == 0

    def test_rolling_drift_empty_current(self, sample_reference_df):
        """Test rolling drift with empty current period."""
        # Empty current DataFrame
        empty_current = pd.DataFrame({"timestamp": [], "value": []})

        result = compute_rolling_drift(
            reference_df=sample_reference_df,
            current_df=empty_current,
            timestamp_col="timestamp",
            value_col="value",
            rolling_window=7,
        )

        # Should return empty DataFrame
        assert len(result) == 0


class TestCreateMonitorFigure:
    """Tests for monitor figure creation."""

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for figure tests."""
        np.random.seed(42)
        dates = pd.date_range(start="2024-01-01", periods=60, freq="D")
        return pd.DataFrame(
            {
                "timestamp": dates,
                "metric1": np.random.normal(100, 10, 60),
                "metric2": np.random.normal(50, 5, 60),
            }
        )

    @pytest.fixture
    def sample_ref_df(self):
        """Create sample reference DataFrame for figure tests."""
        np.random.seed(42)
        dates = pd.date_range(start="2024-01-01", periods=20, freq="D")
        return pd.DataFrame(
            {
                "timestamp": dates,
                "metric1": np.random.normal(100, 10, 20),
                "metric2": np.random.normal(50, 5, 20),
            }
        )

    @pytest.mark.skipif(
        create_monitor_figure is None, reason="Dash dependencies not available"
    )
    def test_create_figure_single_dimension(self, sample_df, sample_ref_df):
        """Test figure creation with single dimension."""
        fig = create_monitor_figure(
            df=sample_df,
            ref_df=sample_ref_df,
            timestamp_col="timestamp",
            dimension="metric1",
            reference_start=datetime.datetime(2024, 1, 1),
            reference_end=datetime.datetime(2024, 1, 20),
            rolling_window=7,
        )

        assert fig is not None
        # Should have traces for rolling stats, reference, and drift
        assert len(fig.data) > 0
