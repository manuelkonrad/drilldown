# SPDX-FileCopyrightText: 2026 Manuel Konrad
#
# SPDX-License-Identifier: MIT

"""Root Cause Analysis algorithms and visualization utilities."""

import logging
import warnings
from contextlib import contextmanager

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
from interpret.glassbox import (
    ExplainableBoostingClassifier,
    ExplainableBoostingRegressor,
)
from plotly.subplots import make_subplots
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from drilldown.constants import (
    MAX_SHAP_SAMPLES,
    N_ESTIMATORS,
    RANDOM_STATE,
    WHAT_IF_N_POINTS,
    WHAT_IF_PERCENTILE_HIGH,
    WHAT_IF_PERCENTILE_LOW,
)
from drilldown.utils import apply_theme

logger = logging.getLogger(__name__)


@contextmanager
def suppress_warnings():
    """Context manager for targeted warning suppression."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        yield


# Analysis type options
ANALYZE_TYPES = [
    {"value": "correlation", "label": "Correlation Analysis"},
    {"value": "feature_importance", "label": "Feature Importance (SHAP)"},
    {"value": "ebm_global", "label": "Global Explanations (EBM)"},
    {"value": "ebm_local", "label": "Local Explanations (EBM)"},
    {"value": "what_if", "label": "What-If Analysis"},
]


def hex_to_rgba(hex_color: str, alpha: float = 0.2) -> str:
    """Convert hex color to rgba string (e.g., '#636EFA' -> 'rgba(99, 110, 250, 0.2)')."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def _prepare_data_for_modeling(
    df: pd.DataFrame, target: str, features: list[str]
) -> tuple[pd.DataFrame, pd.Series, dict[str, LabelEncoder], LabelEncoder | None, bool]:
    """Prepare data for modeling by encoding categorical variables."""
    X = df[features].copy()
    y = df[target].copy()

    is_categorical_target = y.dtype == "object" or y.dtype.name == "category"

    # Encode categorical features
    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    encoders = {}
    for col in categorical_features:
        encoders[col] = LabelEncoder()
        X[col] = encoders[col].fit_transform(X[col].astype(str))

    # Encode categorical target
    target_encoder = None
    if is_categorical_target:
        target_encoder = LabelEncoder()
        y = pd.Series(target_encoder.fit_transform(y.astype(str)), index=y.index)

    return X, y, encoders, target_encoder, is_categorical_target


def _validate_and_clean_data(
    X: pd.DataFrame, y: pd.Series, min_samples: int = 10
) -> tuple[pd.DataFrame, pd.Series, dict | None]:
    """Validate and clean data for modeling.

    Returns:
        Tuple containing:
        - X: Cleaned feature DataFrame
        - y: Cleaned target series
        - error_dict: Dictionary with error message if validation fails, None otherwise
    """
    if len(X) < min_samples:
        return (
            X,
            y,
            {
                "error": f"Not enough data points for analysis (minimum {min_samples} required)"
            },
        )

    if X.isnull().all().any() or y.isnull().all():
        return X, y, {"error": "One or more columns contain only missing values"}

    # Fill missing values
    X = X.fillna(X.median())
    y = y.fillna(y.median())

    return X, y, None


def _create_model_for_task(is_categorical: bool, model_type: str = "random_forest"):
    """Create an appropriate model based on the problem type.

    Args:
        is_categorical: Whether the target is categorical (classification) or continuous (regression)
        model_type: Type of model to create ('random_forest', 'gradient_boosting', or 'ebm')

    Returns:
        Instantiated model object
    """
    if model_type == "random_forest":
        if is_categorical:
            return RandomForestClassifier(
                n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1
            )
        return RandomForestRegressor(
            n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1
        )
    elif model_type == "gradient_boosting":
        if is_categorical:
            return GradientBoostingClassifier(
                n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE
            )
        return GradientBoostingRegressor(
            n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE
        )
    elif model_type == "ebm":
        if is_categorical:
            return ExplainableBoostingClassifier(random_state=RANDOM_STATE)
        return ExplainableBoostingRegressor(random_state=RANDOM_STATE)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def compute_feature_importance(
    df: pd.DataFrame, target: str, features: list[str]
) -> dict:
    """Compute feature importance using SHAP with a Random Forest model."""
    try:
        X, y, encoders, target_encoder, is_categorical_target = (
            _prepare_data_for_modeling(df, target, features)
        )

        X, y, error = _validate_and_clean_data(X, y)
        if error:
            return error

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=RANDOM_STATE
        )

        model = _create_model_for_task(is_categorical_target, "random_forest")
        model.fit(X_train, y_train)

        importances = model.feature_importances_

        try:
            with suppress_warnings():
                explainer = shap.TreeExplainer(model)
                shap_sample_size = min(MAX_SHAP_SAMPLES, len(X_test))
                shap_values = explainer.shap_values(X_test[:shap_sample_size])

            shap_importance = np.abs(shap_values).mean(axis=0)

            return {
                "features": features,
                "importances": importances.tolist(),
                "shap_importances": shap_importance.tolist(),
                "shap_values": shap_values.tolist(),
                "feature_values": X_test[:shap_sample_size].tolist(),
                "model_score": model.score(
                    X_test[:shap_sample_size], y_test[:shap_sample_size]
                ),
            }
        except Exception as e:
            logger.debug(
                "SHAP computation failed, falling back to RF importances: %s", e
            )
            return {
                "features": features,
                "importances": importances.tolist(),
                "shap_importances": importances.tolist(),
                "model_score": model.score(X_test, y_test),
            }
    except Exception as e:
        return {"error": str(e)}


def _compute_single_what_if(
    df: pd.DataFrame, target: str, feature: str, features: list[str]
) -> dict:
    """Compute what-if analysis for a single feature."""
    try:
        X, y, encoders, target_encoder, is_categorical_target = (
            _prepare_data_for_modeling(df, target, features)
        )

        X, y, error = _validate_and_clean_data(X, y)
        if error:
            return error

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        with suppress_warnings():
            model = _create_model_for_task(is_categorical_target, "gradient_boosting")
            model.fit(X_scaled, y)

        feature_idx = features.index(feature)
        feature_values = X[feature].values

        is_categorical_feature = (
            df[feature].dtype == "object" or df[feature].dtype.name == "category"
        )

        if is_categorical_feature:
            feature_range = np.unique(feature_values)
        else:
            feature_min = float(np.percentile(feature_values, WHAT_IF_PERCENTILE_LOW))
            feature_max = float(np.percentile(feature_values, WHAT_IF_PERCENTILE_HIGH))
            feature_range = np.linspace(feature_min, feature_max, WHAT_IF_N_POINTS)

        X_median = X.median(axis=0).values.copy()

        predictions = []
        for val in feature_range:
            X_scenario = X_median.copy()
            X_scenario[feature_idx] = val
            X_scenario_df = pd.DataFrame([X_scenario], columns=X.columns)
            X_scenario_scaled = scaler.transform(X_scenario_df)[0]
            pred = model.predict([X_scenario_scaled])[0]
            predictions.append(float(pred))

        return {
            "feature": feature,
            "feature_range": feature_range.tolist(),
            "predictions": predictions,
            "current_mean": float(feature_values.mean()),
            "current_std": float(feature_values.std()),
            "target_mean": float(y.mean()),
            "model_score": model.score(X_scaled, y),
            "is_categorical": is_categorical_feature,
        }
    except Exception as e:
        return {"error": str(e)}


def compute_what_if_analysis(
    df: pd.DataFrame, target: str, feature: str | list[str], features: list[str]
) -> dict:
    """Compute what-if analysis by varying a feature and predicting outcomes."""
    feature_list = [feature] if isinstance(feature, str) else feature
    results = {}
    for f in feature_list:
        results[f] = _compute_single_what_if(df, target, f, features)
    return {"results": results, "target": target}


def compute_correlation_analysis(
    df: pd.DataFrame, target: str, features: list[str]
) -> dict:
    """Compute correlation analysis between target and features."""
    try:
        analysis_df = df[[target] + features].copy()

        for col in [target] + features:
            if (
                analysis_df[col].dtype == "object"
                or analysis_df[col].dtype.name == "category"
            ):
                encoder = LabelEncoder()
                analysis_df[col] = encoder.fit_transform(analysis_df[col].astype(str))

        analysis_df = analysis_df.dropna()

        if len(analysis_df) < 3:
            return {"error": "Not enough data points for correlation analysis"}

        correlations = {}
        for feature in features:
            corr = analysis_df[target].corr(analysis_df[feature])
            correlations[feature] = float(corr) if not np.isnan(corr) else 0.0

        sorted_features = sorted(
            correlations.keys(), key=lambda x: abs(correlations[x]), reverse=True
        )

        return {
            "target": target,
            "correlations": correlations,
            "sorted_features": sorted_features,
            "correlation_matrix": analysis_df.corr().to_dict(),
            "n_samples": len(analysis_df),
        }
    except Exception as e:
        return {"error": str(e)}


def create_feature_importance_figure(
    result: dict, theme: str | None = None
) -> go.Figure:
    """Create a bar chart figure for feature importance visualization."""
    if "error" in result:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error: {result['error']}",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="red"),
        )
        return apply_theme(fig, theme)

    features = result["features"]
    importances = result.get("shap_importances", result["importances"])

    # Sort by importance
    sorted_idx = np.argsort(importances)[::-1]
    sorted_features = [features[i] for i in sorted_idx]
    sorted_importances = [importances[i] for i in sorted_idx]

    # Create horizontal bar chart
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=sorted_features[::-1],  # Reverse to show highest at top
            x=sorted_importances[::-1],
            orientation="h",
            marker=dict(
                color=sorted_importances[::-1],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(
                    title="Score",
                    len=1.0,
                    y=0.0,
                    yanchor="bottom",
                ),
            ),
            text=[f"{v:.3f}" for v in sorted_importances[::-1]],
            textposition="auto",
        )
    )

    fig.update_layout(
        xaxis_title=f"Score (R² = {result.get('model_score', 0):.3f})",
        margin=dict(l=50, r=50, t=50, b=50),
    )

    return apply_theme(fig, theme)


def create_what_if_figure(result: dict, theme: str | None = None) -> dict | go.Figure:
    """Create figures for what-if analysis, returning tabs for multiple features."""
    if "error" in result:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error: {result['error']}",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="red"),
        )
        return apply_theme(fig, theme)

    features_results = result["results"]
    target = result["target"]

    valid_results = {f: r for f, r in features_results.items() if "error" not in r}

    if not valid_results:
        fig = go.Figure()
        fig.add_annotation(
            text="Error: No valid what-if analysis results",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="red"),
        )
        return apply_theme(fig, theme)

    figures = {}

    for feature_name, feature_result in valid_results.items():
        feature_range = feature_result["feature_range"]
        predictions = feature_result["predictions"]
        current_mean = feature_result["current_mean"]
        target_mean = feature_result["target_mean"]
        model_score = feature_result.get("model_score", 0)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=feature_range,
                y=predictions,
                mode="lines",
                name="Predicted Outcome",
                line=dict(color="#636EFA", width=3),
                fill="tozeroy",
                fillcolor="rgba(99, 110, 250, 0.2)",
            )
        )

        fig.add_vline(
            x=current_mean,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"{feature_name} mean: {current_mean:.2f}",
            annotation_position="top",
        )

        fig.add_hline(
            y=target_mean,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"Target mean: {target_mean:.2f}",
        )

        fig.update_layout(
            xaxis_title=f"{feature_name}",
            yaxis_title=f"Prediction of {target} (R² = {model_score:.3f})",
        )

        figures[feature_name] = apply_theme(fig, theme)

    return {"type": "tabs", "figures": figures}


def create_correlation_figure(result: dict, theme: str | None = None) -> go.Figure:
    """Create a figure with bar chart and heatmap for correlation visualization."""
    if "error" in result:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error: {result['error']}",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="red"),
        )
        return apply_theme(fig, theme)

    correlations = result["correlations"]
    sorted_features = result["sorted_features"]
    target = result["target"]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"Correlation with {target}",
            f"Correlation Matrix (n={result.get('n_samples', 0)})",
        ),
        column_widths=[0.4, 0.6],
        horizontal_spacing=0.1,
    )

    colors = [
        px.colors.sequential.Viridis[
            int((correlations[f] + 1) / 2 * (len(px.colors.sequential.Viridis) - 1))
        ]
        for f in sorted_features
    ]

    fig.add_trace(
        go.Bar(
            y=sorted_features[::-1],
            x=[correlations[f] for f in sorted_features[::-1]],
            orientation="h",
            marker=dict(color=colors[::-1]),
            text=[f"{correlations[f]:.3f}" for f in sorted_features[::-1]],
            textposition="auto",
        ),
        row=1,
        col=1,
    )

    # Heatmap of correlation matrix
    corr_matrix = result["correlation_matrix"]
    all_cols = [target] + sorted_features
    matrix_data = []
    for col1 in all_cols:
        row = []
        for col2 in all_cols:
            if col1 in corr_matrix and col2 in corr_matrix[col1]:
                row.append(corr_matrix[col1][col2])
            else:
                row.append(0)
        matrix_data.append(row)

    fig.add_trace(
        go.Heatmap(
            z=matrix_data,
            x=all_cols,
            y=all_cols,
            colorscale="Viridis",
            zmid=0,
            text=[[f"{v:.2f}" for v in row] for row in matrix_data],
            texttemplate="%{text}",
            textfont={"size": 12},
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Correlation Coefficient", row=1, col=1)

    return apply_theme(fig, theme)


def _extract_density_data(feature_data: dict) -> dict:
    """Extract density data from EBM feature data, returning names and scores."""
    if "density" not in feature_data or not isinstance(feature_data["density"], dict):
        return {}

    density = feature_data["density"]
    return {
        "names": list(density["names"]) if "names" in density else [],
        "scores": list(density["scores"]) if "scores" in density else [],
    }


def compute_ebm_analysis(df: pd.DataFrame, target: str, features: list[str]) -> dict:
    """Compute Explainable Boosting Machine (EBM) global analysis."""
    try:
        X, y, encoders, target_encoder, is_categorical_target = (
            _prepare_data_for_modeling(df, target, features)
        )

        # Basic validation
        if len(X) < 10:
            return {
                "error": "Not enough data points for EBM analysis (minimum 10 required)"
            }

        if X.isnull().all().any() or y.isnull().all():
            return {"error": "One or more columns contain only missing values"}

        # Fill missing values (EBM-specific: use mode for categorical targets)
        for col in X.columns:
            if X[col].isnull().any():
                X[col] = X[col].fillna(X[col].median())

        if y.isnull().any():
            if is_categorical_target:
                y = y.fillna(y.mode()[0] if not y.mode().empty else 0)
            else:
                y = y.fillna(y.median())

        X_all = X.copy()
        y_all = y.copy()

        X_all = X_all.astype("float64")
        if is_categorical_target:
            y_all = y_all.astype("int64")
        else:
            y_all = y_all.astype("float64")

        with suppress_warnings():
            model = _create_model_for_task(is_categorical_target, "ebm")
            model.fit(X_all, y_all)

        ebm_global = model.explain_global()

        feature_names = list(ebm_global.data()["names"])
        feature_scores = list(ebm_global.data()["scores"])

        feature_data = {}
        for i, feature_name in enumerate(feature_names):
            f_data = ebm_global.data(i)
            feature_data[feature_name] = {
                "type": f_data["type"],
                "scores": list(f_data["scores"]),
                "names": (list(f_data["names"]) if "names" in f_data else None),
                "values": (list(f_data["values"]) if "values" in f_data else None),
                "density": _extract_density_data(f_data),
            }

        model_score = model.score(X_all, y_all)

        X_all_list = X_all.to_dict("records")
        y_all_list = y_all.tolist() if hasattr(y_all, "tolist") else list(y_all)
        X_all_indices = X_all.index.tolist()

        return {
            "feature_names": feature_names,
            "feature_scores": feature_scores,
            "feature_data": feature_data,
            "model_score": model_score,
            "is_categorical": is_categorical_target,
            "n_samples": len(X_all),
            # Data for on-demand local explanation computation
            "X_all": X_all_list,
            "y_all": y_all_list,
            "X_all_indices": X_all_indices,  # Original DataFrame indices of all samples
            "features": features,
        }
    except Exception as e:
        logger.exception("EBM analysis failed: %s", e)
        return {"error": str(e)}


def compute_ebm_local_explanation(result: dict, sample_index: int) -> dict:
    """Compute local explanation for a specific sample on demand."""
    try:
        X_all_list = result.get("X_all", [])
        y_all_list = result.get("y_all", [])
        X_all_indices = result.get("X_all_indices", [])
        is_categorical = result.get("is_categorical", False)

        if sample_index not in X_all_indices:
            return {"error": f"Sample index {sample_index} not found in dataset"}

        position = X_all_indices.index(sample_index)

        if position < 0 or position >= len(X_all_list):
            return {
                "error": f"Invalid sample position: {position}. "
                f"Must be between 0 and {len(X_all_list) - 1}"
            }

        X_all = pd.DataFrame(X_all_list)
        y_all = pd.Series(y_all_list)

        with suppress_warnings():
            model = _create_model_for_task(is_categorical, "ebm")
            model.fit(X_all, y_all)

        sample = X_all.iloc[[position]]
        ebm_local = model.explain_local(sample, y=None)

        feature_names = list(ebm_local.data(0)["names"])
        feature_scores = list(ebm_local.data(0)["scores"])

        return {
            "sample_idx": sample_index,
            "feature_names": feature_names,
            "feature_scores": feature_scores,
            "sample_values": X_all.iloc[position].to_dict(),
        }
    except Exception as e:
        logger.exception("EBM local explanation failed: %s", e)
        return {"error": str(e)}


def create_ebm_global_figure(result: dict, theme: str | None = None) -> dict:
    """Create EBM global explanation with summary and individual feature tabs."""
    if "error" in result:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error: {result['error']}",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="red"),
        )
        return {"type": "single", "figure": apply_theme(fig, theme)}

    feature_names = result["feature_names"]
    feature_scores = result["feature_scores"]
    feature_data = result.get("feature_data", {})
    model_score = result.get("model_score", 0)
    score_type = "R²" if not result.get("is_categorical", False) else "Accuracy"

    sorted_idx = np.argsort(feature_scores)[::-1]
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_scores = [feature_scores[i] for i in sorted_idx]

    summary_fig = go.Figure()
    summary_fig.add_trace(
        go.Bar(
            y=sorted_features[::-1],
            x=sorted_scores[::-1],
            orientation="h",
            marker=dict(
                color=sorted_scores[::-1],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(
                    title="Score",
                    len=1.0,
                    y=0.0,
                    yanchor="bottom",
                ),
            ),
            text=[f"{v:.3f}" for v in sorted_scores[::-1]],
            textposition="auto",
        )
    )
    summary_fig.update_layout(
        xaxis_title=f"Score ({score_type} = {model_score:.3f})",
        margin=dict(l=50, r=50, t=50, b=50),
    )
    summary_fig = apply_theme(summary_fig, theme)

    figures = {"Summary": summary_fig}

    for idx in sorted_idx:
        feature_name = feature_names[idx]

        # Skip interaction terms
        if "&" in feature_name:
            continue

        f_data = feature_data.get(feature_name, {})
        if not f_data:
            continue

        # Get feature data
        scores = f_data.get("scores", [])
        x_values = f_data.get("names") or f_data.get("values", [])

        # Ensure x_values is a list and not None
        if x_values is None:
            x_values = []

        # Ensure x_values and scores have compatible lengths
        if len(x_values) > len(scores):
            x_values = x_values[: len(scores)]

        if not scores or not x_values:
            continue

        # Create individual figure with stacked subplots
        # Handle density data
        density_data = f_data.get("density", {})
        density_scores = (
            density_data.get("scores", []) if isinstance(density_data, dict) else []
        )
        density_names = (
            density_data.get("names", []) if isinstance(density_data, dict) else []
        )

        # Create stacked subplots: scores on top, density on bottom
        fig = make_subplots(
            rows=2,
            cols=1,
            row_heights=[0.6, 0.4],
            vertical_spacing=0.05,
            shared_xaxes=True,
        )

        # Calculate symmetric y-axis range for scores (equal negative and positive)
        max_abs_score = max(abs(min(scores)), abs(max(scores))) if scores else 1
        y_range = [-max_abs_score * 1.1, max_abs_score * 1.1]

        # Add score trace with color coding based on y value
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=scores,
                mode="lines+markers",
                name="Score",
                line=dict(color="#8aa3b8", width=2),
                marker=dict(
                    size=8,
                    color=scores,
                    colorscale="Viridis",  # Red for negative, Blue for positive
                    cmin=-max_abs_score,
                    cmax=max_abs_score,
                    showscale=True,
                    colorbar=dict(
                        title="Score",
                        len=0.5,
                        y=1.0,
                        yanchor="top",
                    ),
                ),
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        if density_scores and density_names:
            fig.add_trace(
                go.Bar(
                    x=density_names,
                    y=density_scores,
                    name="Density",
                    marker=dict(color="#8aa3b8", opacity=0.6),
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

        fig.update_layout(
            margin=dict(l=50, r=50, t=50, b=50),
        )

        fig.update_xaxes(title_text=f"{feature_name}", row=2, col=1)
        fig.update_yaxes(title_text="Score", range=y_range, row=1, col=1)
        fig.update_yaxes(title_text="Density", row=2, col=1)

        figures[feature_name] = apply_theme(fig, theme)

    return {"type": "tabs", "figures": figures}


def create_ebm_local_figure(local_result: dict, theme: str | None = None) -> go.Figure:
    """Create a bar chart showing feature contributions for a specific sample."""
    if "error" in local_result:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error: {local_result['error']}",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="red"),
        )
        return apply_theme(fig, theme)

    feature_names = local_result.get("feature_names", [])
    feature_scores = local_result.get("feature_scores", [])
    sample_values = local_result.get("sample_values", {})
    sample_id = local_result.get("sample_id", None)
    target_field = local_result.get("target_field", "target")

    if not feature_names:
        fig = go.Figure()
        fig.add_annotation(
            text="No explanation data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="red"),
        )
        return apply_theme(fig, theme)

    # Sort by absolute importance
    sorted_idx = np.argsort(np.abs(feature_scores))[::-1]
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_scores = [feature_scores[i] for i in sorted_idx]

    # Get sample values for the sorted features
    sorted_values = [sample_values.get(f, "N/A") for f in sorted_features]

    # Create bar chart showing top features
    top_n = min(15, len(sorted_features))  # Show top 15 features
    fig = go.Figure()

    # color according to viridis scale based on score
    colors = [
        px.colors.sequential.Viridis[
            int(
                (score - min(sorted_scores))
                / (max(sorted_scores) - min(sorted_scores))
                * (len(px.colors.sequential.Viridis) - 1)
            )
        ]
        for score in sorted_scores[:top_n]
    ]

    fig.add_trace(
        go.Bar(
            y=sorted_features[:top_n][::-1],  # Reverse to show highest at top
            x=sorted_scores[:top_n][::-1],
            orientation="h",
            marker=dict(color=colors[::-1]),
            text=[f"{v:.3f}" for v in sorted_scores[:top_n][::-1]],
            textposition="auto",
            hovertemplate="<b>%{y}</b><br>"
            + "Contribution: %{x:.3f}<br>"
            + "Value: %{customdata}<br>"
            + "<extra></extra>",
            customdata=[
                f"{v:.3f}" if isinstance(v, (int, float)) else str(v)
                for v in sorted_values[:top_n][::-1]
            ],
        )
    )

    fig.update_layout(
        xaxis_title=f"Contribution to {target_field} for sample {sample_id}",
        margin=dict(l=50, r=50, t=50, b=50),
        showlegend=False,
    )

    return apply_theme(fig, theme)
