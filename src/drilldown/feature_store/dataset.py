# SPDX-FileCopyrightText: 2026 Manuel Konrad
#
# SPDX-License-Identifier: MIT

"""Dataset handling with Delta Lake backend."""

import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from deltalake import DeltaTable
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from drilldown.feature_store.column import Column
from drilldown.feature_store.constants import ColumnType


class Dataset(BaseModel):
    """A dataset backed by Delta Lake."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    path: str
    name: str
    description: str = ""
    columns: dict[str, Column] = Field(default_factory=dict)
    partitioning_columns: list[str] = Field(default_factory=list)
    datetime_columns: list[str] = Field(default_factory=list)
    date_columns: list[str] = Field(default_factory=list)
    timestamp_columns: list[str] = Field(default_factory=list)
    year_week_columns: list[str] = Field(default_factory=list)
    year_month_columns: list[str] = Field(default_factory=list)
    _delta_table: DeltaTable | None = PrivateAttr(default=None)

    def _get_delta_table(self) -> DeltaTable:
        """Get or create the Delta table instance."""
        if self._delta_table is None:
            self._delta_table = DeltaTable(self.path)
        return self._delta_table

    def _load_metadata(self):
        """Load metadata from Delta table schema and partitioning info."""
        import pyarrow as pa

        dt = self._get_delta_table()
        self.description = dt.metadata().description or ""

        try:
            delta_schema = dt.schema().to_pyarrow()
        except AttributeError:
            delta_schema = dt.schema().to_arrow()

        if not isinstance(delta_schema, pa.Schema):
            schema = pa.schema(delta_schema)
        else:
            schema = delta_schema

        self.columns = {}
        for field in schema:
            col = Column.from_arrow_field(field)
            self.columns[field.name] = col

        self.partitioning_columns = dt.metadata().partition_columns or []

        self.datetime_columns = [
            name
            for name, col in self.columns.items()
            if col.column_type
            in [
                ColumnType.TIMESTAMP,
                ColumnType.DATE,
                ColumnType.YEAR_WEEK,
                ColumnType.YEAR_MONTH,
            ]
        ]

        self.date_columns = [
            name
            for name, col in self.columns.items()
            if col.column_type == ColumnType.DATE
        ]

        self.timestamp_columns = [
            name
            for name, col in self.columns.items()
            if col.column_type == ColumnType.TIMESTAMP
        ]

        self.year_week_columns = [
            name
            for name, col in self.columns.items()
            if col.column_type == ColumnType.YEAR_WEEK
        ]

        self.year_month_columns = [
            name
            for name, col in self.columns.items()
            if col.column_type == ColumnType.YEAR_MONTH
        ]

    def get_dataframe_date_range(
        self,
        start: datetime.datetime,
        end: datetime.datetime,
        partitions: list[str] | None = None,
    ) -> tuple[pd.DataFrame, list[dict[str, str]] | None]:
        """Get a dataframe filtered by date range and optional partitions."""
        dt = self._get_delta_table()

        filters: list[tuple[str, str, Any]] = []

        # Find the primary timestamp column
        timestamp_col = None
        for name, col in self.columns.items():
            if col.column_type == ColumnType.PRIMARY_TIMESTAMP:
                timestamp_col = name
                break

        if not timestamp_col and self.datetime_columns:
            timestamp_col = self.datetime_columns[0]

        # Separate partition columns by type
        date_partition_cols = [
            col for col in self.partitioning_columns if col in self.date_columns
        ]
        timestamp_partition_cols = [
            col for col in self.partitioning_columns if col in self.timestamp_columns
        ]
        year_week_partition_cols = [
            col for col in self.partitioning_columns if col in self.year_week_columns
        ]
        year_month_partition_cols = [
            col for col in self.partitioning_columns if col in self.year_month_columns
        ]

        # Add filters based on timestamp range
        if timestamp_col:
            # For timestamp columns, use datetime directly
            if timestamp_col in timestamp_partition_cols:
                filters.append((timestamp_col, ">=", start))
                filters.append((timestamp_col, "<=", end))
            elif timestamp_col in date_partition_cols:
                # For date columns, extract date portion
                filters.append((timestamp_col, ">=", start.date()))
                filters.append((timestamp_col, "<=", end.date()))
            else:
                # If not partitioned, still filter (timestamp or date)
                if timestamp_col in self.date_columns:
                    filters.append((timestamp_col, ">=", start.date()))
                    filters.append((timestamp_col, "<=", end.date()))
                else:
                    filters.append((timestamp_col, ">=", start))
                    filters.append((timestamp_col, "<=", end))

        # Add filters for other datetime partition columns (for efficiency)
        for dt_col in timestamp_partition_cols:
            if dt_col != timestamp_col:
                filters.append((dt_col, ">=", start))
                filters.append((dt_col, "<=", end))

        # Add filters for date partition columns (use date portion)
        for date_col in date_partition_cols:
            if date_col != timestamp_col:
                filters.append((date_col, ">=", start.date()))
                filters.append((date_col, "<=", end.date()))

        # Add filters for year_week partition columns (format: YYYY-cwWW)
        for yw_col in year_week_partition_cols:
            start_week = f"{start.year}-cw{start.isocalendar()[1]:02d}"
            end_week = f"{end.year}-cw{end.isocalendar()[1]:02d}"
            filters.append((yw_col, ">=", start_week))
            filters.append((yw_col, "<=", end_week))

        # Add filters for year_month partition columns (format: YYYY-MM)
        for ym_col in year_month_partition_cols:
            start_month = f"{start.year}-{start.month:02d}"
            end_month = f"{end.year}-{end.month:02d}"
            filters.append((ym_col, ">=", start_month))
            filters.append((ym_col, "<=", end_month))

        # Add partition filters if provided
        if partitions and self.partitioning_columns:
            # partitions is a list of selected partition values
            # Need to match with partitioning_columns
            additional_partitions = [
                col
                for col in self.partitioning_columns
                if col not in self.datetime_columns
            ]
            if additional_partitions and partitions:
                for i, part_col in enumerate(additional_partitions):
                    if i < len(partitions):
                        filters.append((part_col, "=", partitions[i]))

        if filters:
            df = dt.to_pandas(filters=filters)
        else:
            df = dt.to_pandas()

        partition_options = None
        additional_partition_cols = [
            col for col in self.partitioning_columns if col not in self.datetime_columns
        ]
        if additional_partition_cols:
            partition_values = df[additional_partition_cols].drop_duplicates()
            partition_options = [
                {col: str(row[col]) for col in additional_partition_cols}
                for _, row in partition_values.iterrows()
            ]

        return df, partition_options

    def get_column_names_by_type(self) -> list[Any]:
        """Get column names grouped by type: [primary_id, primary_timestamp, column_types_dict]."""
        primary_id: str | None = None
        primary_timestamp: str | None = None
        column_types: dict[str, list[str]] = {
            "identifier": [],
            "timestamp": [],
            "date": [],
            "year_week": [],
            "year_month": [],
            "categorical": [],
            "numerical": [],
            "curve": [],
            "object": [],
        }

        for name, col in self.columns.items():
            if col.column_type == ColumnType.PRIMARY_ID:
                primary_id = name
            elif col.column_type == ColumnType.PRIMARY_TIMESTAMP:
                primary_timestamp = name
            elif col.column_type == ColumnType.IDENTIFIER:
                column_types["identifier"].append(name)
            elif col.column_type == ColumnType.TIMESTAMP:
                column_types["timestamp"].append(name)
            elif col.column_type == ColumnType.DATE:
                column_types["date"].append(name)
            elif col.column_type == ColumnType.YEAR_WEEK:
                column_types["year_week"].append(name)
            elif col.column_type == ColumnType.YEAR_MONTH:
                column_types["year_month"].append(name)
            elif col.column_type == ColumnType.CATEGORICAL:
                column_types["categorical"].append(name)
            elif col.column_type == ColumnType.NUMERICAL:
                column_types["numerical"].append(name)
            elif (
                col.column_type == ColumnType.CURVE
                or col.column_type == ColumnType.URI_CURVE
            ):
                column_types["curve"].append(name)
            elif (
                col.column_type == ColumnType.OBJECT
                or col.column_type == ColumnType.URI_IMG
            ):
                column_types["object"].append(name)

        return [primary_id, primary_timestamp, column_types]

    @classmethod
    def from_path(cls, path: str, name: str | None = None) -> "Dataset":
        """Create a Dataset from a Delta table path."""
        if name is None:
            name = Path(path).name

        dataset = cls(path=path, name=name)
        dataset._load_metadata()
        return dataset
