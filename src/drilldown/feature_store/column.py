# SPDX-FileCopyrightText: 2026 Manuel Konrad
#
# SPDX-License-Identifier: MIT

"""Column metadata and type handling."""

from typing import Any

import pyarrow as pa
from pydantic import BaseModel, ConfigDict

from drilldown.feature_store.constants import ColumnType


class Column(BaseModel):
    """Metadata for a single column."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    column_type: str
    description: str = ""
    object_format: str | None = None

    def read_uri_object(self, uri: str) -> Any:
        """Read and deserialize an object from a URI based on column type."""
        from drilldown.feature_store.uri_handler import URIHandler

        uri_handler = URIHandler()
        return uri_handler.read_uri(uri, self.column_type, self.object_format)

    @staticmethod
    def infer_column_type(
        arrow_type: pa.DataType, metadata: dict[str, Any] | None = None
    ) -> str:
        """Infer column type from Arrow data type, checking metadata first."""
        if metadata:
            column_type = metadata.get("column_type")
            if column_type:
                return column_type

        if pa.types.is_integer(arrow_type) or pa.types.is_floating(arrow_type):
            return ColumnType.NUMERICAL
        elif pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
            return ColumnType.CATEGORICAL
        elif pa.types.is_timestamp(arrow_type):
            return ColumnType.TIMESTAMP
        elif pa.types.is_date(arrow_type):
            return ColumnType.DATE
        elif pa.types.is_boolean(arrow_type):
            return ColumnType.CATEGORICAL
        elif pa.types.is_list(arrow_type) or pa.types.is_struct(arrow_type):
            return ColumnType.CURVE
        else:
            return ColumnType.OBJECT

    @classmethod
    def from_arrow_field(cls, field: pa.Field) -> "Column":
        """Create a Column from an Arrow field, extracting metadata."""
        metadata = {}
        if field.metadata:
            metadata = {
                k.decode() if isinstance(k, bytes) else k: v.decode()
                if isinstance(v, bytes)
                else v
                for k, v in field.metadata.items()
            }

        column_type = cls.infer_column_type(field.type, metadata)
        description = metadata.get("description", "")
        object_format = metadata.get("object_format")

        return cls(
            name=field.name,
            column_type=column_type,
            description=description,
            object_format=object_format,
        )
