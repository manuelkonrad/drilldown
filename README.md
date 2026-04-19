# drilldown

[![CI - Tests](https://github.com/manuelkonrad/drilldown/actions/workflows/tests.yml/badge.svg)](https://github.com/manuelkonrad/drilldown/actions/workflows/tests.yml)
[![CI - Bandit](https://github.com/manuelkonrad/drilldown/actions/workflows/bandit.yml/badge.svg)](https://github.com/manuelkonrad/drilldown/actions/workflows/bandit.yml)
[![CI - Build](https://github.com/manuelkonrad/drilldown/actions/workflows/build.yml/badge.svg)](https://github.com/manuelkonrad/drilldown/actions/workflows/build.yml)

[![License - MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://spdx.org/licenses/MIT.html)
[![PyPI - Version](https://img.shields.io/pypi/v/drilldown.svg)](https://pypi.org/project/drilldown)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/drilldown.svg)](https://pypi.org/project/drilldown)
[![Python Project Management - Hatch](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg)](https://github.com/pypa/hatch)
[![Linting - Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Types - Mypy](https://img.shields.io/badge/types-Mypy-blue.svg)](https://github.com/python/mypy)
[![Security - Bandit](https://img.shields.io/badge/security-Bandit-yellow.svg)](https://github.com/PyCQA/bandit)

Explore and analyze multimodal data.

- Built with `dash`, `dash-mantine-components`, and `plotly`
- Lightweight feature store based on `obstore` and `deltalake`
- Light and dark themes

## Getting Started

Installation:

```console
pip install drilldown
```

Demo mode:

```console
drilldown --demo=true
```

This starts the application on a local development server with an auto-generated dataset for exploring its features.

![Screenshot of Home page](docs/assets/screenshot_home.png)

### Explore

Features:

- Table view of the selected dataset
- Chart view with the following chart types:
  - Scatter Plot
  - Time Series
  - Box Plot
  - Histogram
  - Parallel Coordinates
  - Cycle Plot
  - t-SNE Plot (including K-Means and PCA)
- Resizable sidebar with additional visualizations, such as images or curves for samples selected in the table or chart view

![Screenshot of Explore page](docs/assets/screenshot_explore_tsne.png)

### Root Cause Analysis

Available root cause analysis methods include:

- Correlation Analysis
- Feature Importance (SHAP)
- Global Explanations via Explainable Boosting Machine (EBM)
- Local Explanations via EBM
- What-If Analysis

### Drift Monitoring

Drift monitoring based on the Kolmogorov-Smirnov statistic.

## License

`drilldown` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
