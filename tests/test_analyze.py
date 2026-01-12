# SPDX-FileCopyrightText: 2026 Manuel Konrad
#
# SPDX-License-Identifier: MIT

"""Tests for the analyze module."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_data():
    """Create sample data for testing analyze functions."""
    np.random.seed(42)
    n_samples = 100

    # Create correlated features
    x1 = np.random.randn(n_samples)
    x2 = x1 * 0.5 + np.random.randn(n_samples) * 0.5
    x3 = np.random.randn(n_samples)

    # Target depends on features
    y = 2 * x1 + 0.5 * x2 - 1 * x3 + np.random.randn(n_samples) * 0.1

    return pd.DataFrame(
        {
            "feature1": x1,
            "feature2": x2,
            "feature3": x3,
            "target": y,
        }
    )


class TestFeatureImportance:
    """Tests for feature importance analysis."""

    def test_compute_feature_importance_basic(self, sample_data):
        """Test basic feature importance computation."""
        from drilldown.pages.analyze.algorithms import compute_feature_importance

        result = compute_feature_importance(
            sample_data,
            target="target",
            features=["feature1", "feature2", "feature3"],
        )

        assert "error" not in result
        assert "features" in result
        assert "importances" in result
        assert len(result["features"]) == 3
        assert len(result["importances"]) == 3
        assert all(imp >= 0 for imp in result["importances"])

    def test_compute_feature_importance_with_missing_values(self, sample_data):
        """Test feature importance with missing values in data."""
        from drilldown.pages.analyze.algorithms import compute_feature_importance

        df = sample_data.copy()
        df.loc[0, "feature1"] = np.nan
        df.loc[5, "feature2"] = np.nan

        result = compute_feature_importance(
            df,
            target="target",
            features=["feature1", "feature2", "feature3"],
        )

        assert "error" not in result
        assert "importances" in result

    def test_compute_feature_importance_model_score(self, sample_data):
        """Test that model score is returned."""
        from drilldown.pages.analyze.algorithms import compute_feature_importance

        result = compute_feature_importance(
            sample_data,
            target="target",
            features=["feature1", "feature2", "feature3"],
        )

        assert "model_score" in result
        assert 0 <= result["model_score"] <= 1


class TestCorrelationAnalysis:
    """Tests for correlation analysis."""

    def test_compute_correlation_basic(self, sample_data):
        """Test basic correlation computation."""
        from drilldown.pages.analyze.algorithms import compute_correlation_analysis

        result = compute_correlation_analysis(
            sample_data,
            target="target",
            features=["feature1", "feature2", "feature3"],
        )

        assert "error" not in result
        assert "correlations" in result
        assert "sorted_features" in result
        assert len(result["correlations"]) == 3

    def test_compute_correlation_values_in_range(self, sample_data):
        """Test that correlation values are in valid range."""
        from drilldown.pages.analyze.algorithms import compute_correlation_analysis

        result = compute_correlation_analysis(
            sample_data,
            target="target",
            features=["feature1", "feature2", "feature3"],
        )

        for corr in result["correlations"].values():
            assert -1 <= corr <= 1

    def test_compute_correlation_sorted_by_abs(self, sample_data):
        """Test that features are sorted by absolute correlation."""
        from drilldown.pages.analyze.algorithms import compute_correlation_analysis

        result = compute_correlation_analysis(
            sample_data,
            target="target",
            features=["feature1", "feature2", "feature3"],
        )

        sorted_features = result["sorted_features"]
        correlations = result["correlations"]

        for i in range(len(sorted_features) - 1):
            assert abs(correlations[sorted_features[i]]) >= abs(
                correlations[sorted_features[i + 1]]
            )

    def test_compute_correlation_with_few_samples(self):
        """Test correlation analysis with very few samples."""
        from drilldown.pages.analyze.algorithms import compute_correlation_analysis

        df = pd.DataFrame(
            {
                "feature1": [1, 2],
                "target": [1, 2],
            }
        )

        result = compute_correlation_analysis(
            df,
            target="target",
            features=["feature1"],
        )

        # Should return error due to insufficient data
        assert "error" in result


class TestWhatIfAnalysis:
    """Tests for what-if analysis."""

    def test_compute_what_if_basic(self, sample_data):
        """Test basic what-if analysis."""
        from drilldown.pages.analyze.algorithms import compute_what_if_analysis

        result = compute_what_if_analysis(
            sample_data,
            target="target",
            feature="feature1",
            features=["feature1", "feature2", "feature3"],
        )

        assert "results" in result
        feature_result = result["results"]["feature1"]
        assert "error" not in feature_result
        assert "feature" in feature_result
        assert "feature_range" in feature_result
        assert "predictions" in feature_result
        assert len(feature_result["feature_range"]) == len(
            feature_result["predictions"]
        )

    def test_compute_what_if_predictions_reasonable(self, sample_data):
        """Test that what-if predictions are within reasonable range."""
        from drilldown.pages.analyze.algorithms import compute_what_if_analysis

        result = compute_what_if_analysis(
            sample_data,
            target="target",
            feature="feature1",
            features=["feature1", "feature2", "feature3"],
        )

        feature_result = result["results"]["feature1"]
        predictions = feature_result["predictions"]
        target_mean = sample_data["target"].mean()
        target_std = sample_data["target"].std()

        for pred in predictions:
            assert target_mean - 5 * target_std < pred < target_mean + 5 * target_std

    def test_compute_what_if_current_mean(self, sample_data):
        """Test that current mean is included in result."""
        from drilldown.pages.analyze.algorithms import compute_what_if_analysis

        result = compute_what_if_analysis(
            sample_data,
            target="target",
            feature="feature1",
            features=["feature1", "feature2", "feature3"],
        )

        feature_result = result["results"]["feature1"]
        assert "current_mean" in feature_result
        assert np.isclose(
            feature_result["current_mean"], sample_data["feature1"].mean(), rtol=0.01
        )


class TestVisualizationFunctions:
    """Tests for visualization creation functions."""

    def test_create_feature_importance_figure_with_data(self, sample_data):
        """Test feature importance figure creation with valid data."""
        from drilldown.pages.analyze.algorithms import (
            compute_feature_importance,
            create_feature_importance_figure,
        )

        result = compute_feature_importance(
            sample_data,
            target="target",
            features=["feature1", "feature2", "feature3"],
        )

        fig = create_feature_importance_figure(result)

        assert fig is not None
        assert hasattr(fig, "to_dict")

    def test_create_feature_importance_figure_with_error(self):
        """Test feature importance figure creation with error result."""
        from drilldown.pages.analyze.algorithms import create_feature_importance_figure

        result = {"error": "Test error message"}
        fig = create_feature_importance_figure(result)

        assert fig is not None
        # Should have error annotation
        fig_dict = fig.to_dict()
        assert "layout" in fig_dict

    def test_create_correlation_figure_with_data(self, sample_data):
        """Test correlation figure creation with valid data."""
        from drilldown.pages.analyze.algorithms import (
            compute_correlation_analysis,
            create_correlation_figure,
        )

        result = compute_correlation_analysis(
            sample_data,
            target="target",
            features=["feature1", "feature2", "feature3"],
        )

        fig = create_correlation_figure(result)

        assert fig is not None
        assert hasattr(fig, "to_dict")

    def test_create_what_if_figure_with_data(self, sample_data):
        """Test what-if figure creation with valid data."""
        from drilldown.pages.analyze.algorithms import (
            compute_what_if_analysis,
            create_what_if_figure,
        )

        result = compute_what_if_analysis(
            sample_data,
            target="target",
            feature="feature1",
            features=["feature1", "feature2", "feature3"],
        )

        output = create_what_if_figure(result)

        assert output is not None
        assert isinstance(output, dict)
        assert output["type"] == "tabs"
        assert "figures" in output
        assert "feature1" in output["figures"]


class TestAnalyzeTypes:
    """Tests for analyze type definitions."""

    def test_analyze_types_defined(self):
        """Test that analyze types are defined."""
        from drilldown.pages.analyze.algorithms import ANALYZE_TYPES

        assert len(ANALYZE_TYPES) > 0
        for analyze_type in ANALYZE_TYPES:
            assert "value" in analyze_type
            assert "label" in analyze_type

    def test_analyze_types_include_expected(self):
        """Test that expected analyze types are included."""
        from drilldown.pages.analyze.algorithms import ANALYZE_TYPES

        values = [t["value"] for t in ANALYZE_TYPES]
        assert "feature_importance" in values
        assert "what_if" in values
        assert "correlation" in values


class TestThemeSupport:
    """Tests for theme support in analyze visualizations."""

    def test_feature_importance_figure_with_dark_theme(self, sample_data):
        """Test feature importance figure creation with dark theme."""
        from drilldown.pages.analyze.algorithms import (
            compute_feature_importance,
            create_feature_importance_figure,
        )

        result = compute_feature_importance(
            sample_data,
            target="target",
            features=["feature1", "feature2", "feature3"],
        )

        fig = create_feature_importance_figure(result, theme="dark")

        assert fig is not None
        fig_dict = fig.to_dict()
        assert "layout" in fig_dict
        # Check that dark theme is applied to the layout (not just the template)
        assert fig_dict["layout"]["paper_bgcolor"] == "#171b1f"
        assert fig_dict["layout"]["plot_bgcolor"] == "#171b1f"

    def test_feature_importance_figure_with_light_theme(self, sample_data):
        """Test feature importance figure creation with light theme."""
        from drilldown.pages.analyze.algorithms import (
            compute_feature_importance,
            create_feature_importance_figure,
        )

        result = compute_feature_importance(
            sample_data,
            target="target",
            features=["feature1", "feature2", "feature3"],
        )

        fig = create_feature_importance_figure(result, theme="light")

        assert fig is not None
        fig_dict = fig.to_dict()
        assert "layout" in fig_dict
        # Check that light theme is applied (should not have dark background)
        assert fig_dict["layout"].get("paper_bgcolor") != "#171b1f"
        assert fig_dict["layout"].get("plot_bgcolor") != "#171b1f"

    def test_what_if_figure_with_dark_theme(self, sample_data):
        """Test what-if figure creation with dark theme."""
        from drilldown.pages.analyze.algorithms import (
            compute_what_if_analysis,
            create_what_if_figure,
        )

        result = compute_what_if_analysis(
            sample_data,
            target="target",
            feature="feature1",
            features=["feature1", "feature2", "feature3"],
        )

        output = create_what_if_figure(result, theme="dark")

        assert output is not None
        assert output["type"] == "tabs"
        fig = output["figures"]["feature1"]
        fig_dict = fig.to_dict() if hasattr(fig, "to_dict") else fig
        assert "layout" in fig_dict
        assert fig_dict["layout"]["paper_bgcolor"] == "#171b1f"
        assert fig_dict["layout"]["plot_bgcolor"] == "#171b1f"

    def test_what_if_figure_with_light_theme(self, sample_data):
        """Test what-if figure creation with light theme."""
        from drilldown.pages.analyze.algorithms import (
            compute_what_if_analysis,
            create_what_if_figure,
        )

        result = compute_what_if_analysis(
            sample_data,
            target="target",
            feature="feature1",
            features=["feature1", "feature2", "feature3"],
        )

        output = create_what_if_figure(result, theme="light")

        assert output is not None
        assert output["type"] == "tabs"
        fig = output["figures"]["feature1"]
        fig_dict = fig.to_dict() if hasattr(fig, "to_dict") else fig
        assert "layout" in fig_dict
        assert fig_dict["layout"].get("paper_bgcolor") != "#171b1f"
        assert fig_dict["layout"].get("plot_bgcolor") != "#171b1f"

    def test_correlation_figure_with_dark_theme(self, sample_data):
        """Test correlation figure creation with dark theme."""
        from drilldown.pages.analyze.algorithms import (
            compute_correlation_analysis,
            create_correlation_figure,
        )

        result = compute_correlation_analysis(
            sample_data,
            target="target",
            features=["feature1", "feature2", "feature3"],
        )

        fig = create_correlation_figure(result, theme="dark")

        assert fig is not None
        fig_dict = fig.to_dict()
        assert "layout" in fig_dict
        # Check that dark theme is applied
        assert fig_dict["layout"]["paper_bgcolor"] == "#171b1f"
        assert fig_dict["layout"]["plot_bgcolor"] == "#171b1f"

    def test_correlation_figure_with_light_theme(self, sample_data):
        """Test correlation figure creation with light theme."""
        from drilldown.pages.analyze.algorithms import (
            compute_correlation_analysis,
            create_correlation_figure,
        )

        result = compute_correlation_analysis(
            sample_data,
            target="target",
            features=["feature1", "feature2", "feature3"],
        )

        fig = create_correlation_figure(result, theme="light")

        assert fig is not None
        fig_dict = fig.to_dict()
        assert "layout" in fig_dict
        # Check that light theme is applied (should not have dark background)
        assert fig_dict["layout"].get("paper_bgcolor") != "#171b1f"
        assert fig_dict["layout"].get("plot_bgcolor") != "#171b1f"

    def test_feature_importance_figure_with_none_theme(self, sample_data):
        """Test feature importance figure creation with None theme defaults to light."""
        from drilldown.pages.analyze.algorithms import (
            compute_feature_importance,
            create_feature_importance_figure,
        )

        result = compute_feature_importance(
            sample_data,
            target="target",
            features=["feature1", "feature2", "feature3"],
        )

        fig = create_feature_importance_figure(result, theme=None)

        assert fig is not None
        fig_dict = fig.to_dict()
        assert "layout" in fig_dict
        # Check that when theme is None, it defaults to light (no dark background)
        assert fig_dict["layout"].get("paper_bgcolor") != "#171b1f"
        assert fig_dict["layout"].get("plot_bgcolor") != "#171b1f"


class TestMultipleDimensionsSupport:
    """Tests for multi-dimension analysis support."""

    def test_what_if_analysis_with_multiple_features(self, sample_data):
        """Test what-if analysis with multiple features."""
        from drilldown.pages.analyze.algorithms import compute_what_if_analysis

        result = compute_what_if_analysis(
            sample_data,
            target="target",
            feature=["feature1", "feature2"],
            features=["feature1", "feature2", "feature3"],
        )

        assert "results" in result
        assert "target" in result
        assert result["target"] == "target"
        assert "feature1" in result["results"]
        assert "feature2" in result["results"]

    def test_what_if_analysis_with_single_feature(self, sample_data):
        """Test what-if analysis with single feature also returns results dict."""
        from drilldown.pages.analyze.algorithms import compute_what_if_analysis

        result = compute_what_if_analysis(
            sample_data,
            target="target",
            feature="feature1",
            features=["feature1", "feature2", "feature3"],
        )

        assert "results" in result
        assert "target" in result
        assert "feature1" in result["results"]

    def test_what_if_figure_with_multiple_features(self, sample_data):
        """Test what-if figure with multiple features."""
        from drilldown.pages.analyze.algorithms import (
            compute_what_if_analysis,
            create_what_if_figure,
        )

        result = compute_what_if_analysis(
            sample_data,
            target="target",
            feature=["feature1", "feature2"],
            features=["feature1", "feature2", "feature3"],
        )

        output = create_what_if_figure(result)

        assert output is not None
        assert isinstance(output, dict)
        assert output["type"] == "tabs"
        assert "figures" in output
        assert "feature1" in output["figures"]
        assert "feature2" in output["figures"]

    def test_what_if_with_multiple_features_individual_results(self, sample_data):
        """Test that what-if analysis with multiple features contains valid individual results."""
        from drilldown.pages.analyze.algorithms import compute_what_if_analysis

        result = compute_what_if_analysis(
            sample_data,
            target="target",
            feature=["feature1", "feature2"],
            features=["feature1", "feature2", "feature3"],
        )

        for feature_name, feature_result in result["results"].items():
            if "error" not in feature_result:
                assert "feature" in feature_result
                assert "feature_range" in feature_result
                assert "predictions" in feature_result
                assert len(feature_result["feature_range"]) == len(
                    feature_result["predictions"]
                )


class TestEBMAnalysis:
    """Tests for Explainable Boosting Machine analysis."""

    def test_compute_ebm_analysis_basic(self, sample_data):
        """Test basic EBM analysis computation."""
        from drilldown.pages.analyze.algorithms import compute_ebm_analysis

        result = compute_ebm_analysis(
            sample_data,
            target="target",
            features=["feature1", "feature2", "feature3"],
        )

        assert "error" not in result
        assert "feature_names" in result
        assert "feature_scores" in result
        assert "model_score" in result
        assert "X_all" in result
        assert "X_all_indices" in result
        assert len(result["feature_names"]) > 0
        assert len(result["feature_scores"]) == len(result["feature_names"])

    def test_compute_ebm_analysis_model_score(self, sample_data):
        """Test that EBM analysis returns a valid model score."""
        from drilldown.pages.analyze.algorithms import compute_ebm_analysis

        result = compute_ebm_analysis(
            sample_data,
            target="target",
            features=["feature1", "feature2", "feature3"],
        )

        assert "model_score" in result
        # R² can be negative for very poor models, but should be reasonable for our test data
        assert result["model_score"] > -1.0

    def test_compute_ebm_analysis_with_insufficient_data(self):
        """Test EBM analysis with insufficient data."""
        from drilldown.pages.analyze.algorithms import compute_ebm_analysis

        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3],
                "target": [1, 2, 3],
            }
        )

        result = compute_ebm_analysis(
            df,
            target="target",
            features=["feature1"],
        )

        # Should return error due to insufficient data
        assert "error" in result

    def test_compute_ebm_local_explanation(self, sample_data):
        """Test on-demand local explanation computation."""
        from drilldown.pages.analyze.algorithms import (
            compute_ebm_analysis,
            compute_ebm_local_explanation,
        )

        # First compute the global analysis
        result = compute_ebm_analysis(
            sample_data,
            target="target",
            features=["feature1", "feature2", "feature3"],
        )

        # Then compute local explanation on demand
        local_result = compute_ebm_local_explanation(result, 0)

        assert "error" not in local_result
        assert "sample_idx" in local_result
        assert "feature_names" in local_result
        assert "feature_scores" in local_result
        assert local_result["sample_idx"] == 0

    def test_create_ebm_global_figure_with_data(self, sample_data):
        """Test EBM global figure creation with valid data."""
        from drilldown.pages.analyze.algorithms import (
            compute_ebm_analysis,
            create_ebm_global_figure,
        )

        result = compute_ebm_analysis(
            sample_data,
            target="target",
            features=["feature1", "feature2", "feature3"],
        )

        output = create_ebm_global_figure(result)

        assert output is not None
        assert "type" in output
        assert output["type"] == "tabs"
        assert "figures" in output
        assert "Summary" in output["figures"]

    def test_create_ebm_global_figure_with_error(self):
        """Test EBM global figure creation with error result."""
        from drilldown.pages.analyze.algorithms import create_ebm_global_figure

        result = {"error": "Test error message"}
        output = create_ebm_global_figure(result)

        assert output is not None
        assert "type" in output
        assert output["type"] == "single"
        assert "figure" in output

    def test_create_ebm_local_figure_with_data(self, sample_data):
        """Test EBM local figure creation with valid data."""
        from drilldown.pages.analyze.algorithms import (
            compute_ebm_analysis,
            compute_ebm_local_explanation,
            create_ebm_local_figure,
        )

        result = compute_ebm_analysis(
            sample_data,
            target="target",
            features=["feature1", "feature2", "feature3"],
        )

        # Compute local explanation on demand
        local_result = compute_ebm_local_explanation(result, 0)
        fig = create_ebm_local_figure(local_result)

        assert fig is not None
        assert hasattr(fig, "to_dict")

    def test_create_ebm_local_figure_with_error(self):
        """Test EBM local figure creation with error result."""
        from drilldown.pages.analyze.algorithms import create_ebm_local_figure

        local_result = {"error": "Test error message"}
        fig = create_ebm_local_figure(local_result)

        assert fig is not None
        fig_dict = fig.to_dict()
        assert "layout" in fig_dict

    def test_ebm_analysis_with_dark_theme(self, sample_data):
        """Test EBM figure creation with dark theme."""
        from drilldown.pages.analyze.algorithms import (
            compute_ebm_analysis,
            create_ebm_global_figure,
        )

        result = compute_ebm_analysis(
            sample_data,
            target="target",
            features=["feature1", "feature2", "feature3"],
        )

        output = create_ebm_global_figure(result, theme="dark")

        assert output is not None
        assert output["type"] == "tabs"
        summary_fig = output["figures"]["Summary"]
        # Convert to dict for assertion checks
        fig_dict = (
            summary_fig.to_dict() if hasattr(summary_fig, "to_dict") else summary_fig
        )
        assert "layout" in fig_dict
        assert fig_dict["layout"]["paper_bgcolor"] == "#171b1f"
        assert fig_dict["layout"]["plot_bgcolor"] == "#171b1f"

    def test_ebm_analysis_with_light_theme(self, sample_data):
        """Test EBM figure creation with light theme."""
        from drilldown.pages.analyze.algorithms import (
            compute_ebm_analysis,
            create_ebm_global_figure,
        )

        result = compute_ebm_analysis(
            sample_data,
            target="target",
            features=["feature1", "feature2", "feature3"],
        )

        output = create_ebm_global_figure(result, theme="light")

        assert output is not None
        assert output["type"] == "tabs"
        summary_fig = output["figures"]["Summary"]
        # Convert to dict for assertion checks
        fig_dict = (
            summary_fig.to_dict() if hasattr(summary_fig, "to_dict") else summary_fig
        )
        assert "layout" in fig_dict
        assert fig_dict["layout"].get("paper_bgcolor") != "#171b1f"
        assert fig_dict["layout"].get("plot_bgcolor") != "#171b1f"


class TestAnalyzeTypesEBM:
    """Tests for EBM analyze type definition."""

    def test_ebm_in_analyze_types(self):
        """Test that EBM modules are included in analyze types."""
        from drilldown.pages.analyze.algorithms import ANALYZE_TYPES

        values = [t["value"] for t in ANALYZE_TYPES]
        assert "ebm_global" in values
        assert "ebm_local" in values

    def test_ebm_analyze_types_have_labels(self):
        """Test that EBM analyze types have proper labels."""
        from drilldown.pages.analyze.algorithms import ANALYZE_TYPES

        ebm_global = [t for t in ANALYZE_TYPES if t["value"] == "ebm_global"]
        assert len(ebm_global) == 1
        assert "label" in ebm_global[0]

        ebm_local = [t for t in ANALYZE_TYPES if t["value"] == "ebm_local"]
        assert len(ebm_local) == 1
        assert "label" in ebm_local[0]


class TestEBMTabsStructure:
    """Tests for EBM tabs structure and functionality."""

    def test_ebm_global_figure_returns_tabs(self, sample_data):
        """Test that EBM global figure returns tabs structure."""
        from drilldown.pages.analyze.algorithms import (
            compute_ebm_analysis,
            create_ebm_global_figure,
        )

        result = compute_ebm_analysis(
            sample_data,
            target="target",
            features=["feature1", "feature2", "feature3"],
        )

        output = create_ebm_global_figure(result)

        # Check that it returns a tabs structure
        assert output["type"] == "tabs"
        assert "figures" in output
        assert len(output["figures"]) > 0

    def test_ebm_global_figure_has_summary_tab(self, sample_data):
        """Test that summary tab is included."""
        from drilldown.pages.analyze.algorithms import (
            compute_ebm_analysis,
            create_ebm_global_figure,
        )

        result = compute_ebm_analysis(
            sample_data,
            target="target",
            features=["feature1", "feature2", "feature3"],
        )

        output = create_ebm_global_figure(result)

        # Check that summary tab exists
        assert "Summary" in output["figures"]

    def test_ebm_global_figure_excludes_interaction_terms(self, sample_data):
        """Test that interaction terms are excluded from individual tabs."""
        from drilldown.pages.analyze.algorithms import (
            compute_ebm_analysis,
            create_ebm_global_figure,
        )

        result = compute_ebm_analysis(
            sample_data,
            target="target",
            features=["feature1", "feature2", "feature3"],
        )

        output = create_ebm_global_figure(result)

        # Check that interaction terms are not in the figures (except summary)
        for key in output["figures"].keys():
            if key != "Summary":
                assert "&" not in key, f"Interaction term {key} should be excluded"

    def test_ebm_global_figure_has_single_variable_tabs(self, sample_data):
        """Test that single variables have their own tabs."""
        from drilldown.pages.analyze.algorithms import (
            compute_ebm_analysis,
            create_ebm_global_figure,
        )

        result = compute_ebm_analysis(
            sample_data,
            target="target",
            features=["feature1", "feature2", "feature3"],
        )

        output = create_ebm_global_figure(result)

        # Check that single variables are included (at least some of them)
        single_var_count = sum(
            1 for key in output["figures"].keys() if "&" not in key and key != "Summary"
        )
        assert single_var_count > 0, "Should have tabs for single variables"

    def test_ebm_global_individual_figures_have_dual_yaxis(self, sample_data):
        """Test that individual feature figures have stacked subplots for score and density."""
        from drilldown.pages.analyze.algorithms import (
            compute_ebm_analysis,
            create_ebm_global_figure,
        )

        result = compute_ebm_analysis(
            sample_data,
            target="target",
            features=["feature1", "feature2", "feature3"],
        )

        output = create_ebm_global_figure(result)

        single_var_figures = {
            k: v
            for k, v in output["figures"].items()
            if k != "Summary" and "&" not in k
        }

        if single_var_figures:
            first_fig = list(single_var_figures.values())[0]
            # Convert to dict for assertion checks
            fig_dict = (
                first_fig.to_dict() if hasattr(first_fig, "to_dict") else first_fig
            )
            layout = fig_dict["layout"]
            # Individual figures use stacked subplots (rows=2), not overlaying axes
            assert "yaxis" in layout
            assert "yaxis2" in layout

    def test_ebm_analysis_includes_density_data(self, sample_data):
        """Test that EBM analysis includes density data in feature_data."""
        from drilldown.pages.analyze.algorithms import compute_ebm_analysis

        result = compute_ebm_analysis(
            sample_data,
            target="target",
            features=["feature1", "feature2", "feature3"],
        )

        # Check that feature_data exists
        assert "feature_data" in result

        # Check that at least one feature has density data
        feature_data = result["feature_data"]
        assert len(feature_data) > 0

        # Check first feature for density
        first_feature = list(feature_data.keys())[0]
        assert "density" in feature_data[first_feature]
        density = feature_data[first_feature]["density"]
        # Density should be a dict that is either empty or contains 'scores'
        assert isinstance(density, dict) and (not density or "scores" in density)
