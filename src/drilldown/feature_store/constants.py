# SPDX-FileCopyrightText: 2026 Manuel Konrad
#
# SPDX-License-Identifier: MIT

"""Constants for the feature store module."""

from enum import Enum


class ColumnType(str, Enum):
    """Column types for feature store."""

    PRIMARY_ID = "primary_id"
    PRIMARY_TIMESTAMP = "primary_timestamp"
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    CURVE = "curve"
    URI_IMG = "uri_img"
    URI_CURVE = "uri_curve"
    TIMESTAMP = "timestamp"
    DATE = "date"
    YEAR_WEEK = "year_week"
    YEAR_MONTH = "year_month"
    OBJECT = "object"
    IDENTIFIER = "identifier"


class TypeGroups:
    """Groupings of column types for easier access."""

    DATETIME_VARS = ["date", "timestamp", "year_week", "year_month"]
    IDENTIFIER_VARS = ["identifier", "primary_id"]
    NUMERICAL_VARS = ["numerical"]
    CATEGORICAL_VARS = ["categorical"]
    CURVE_VARS = ["curve", "uri_curve"]
    IMAGE_VARS = ["uri_img"]
    OBJECT_VARS = ["object"]
