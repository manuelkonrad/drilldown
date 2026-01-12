# SPDX-FileCopyrightText: 2026 Manuel Konrad
#
# SPDX-License-Identifier: MIT

"""Tests for the feature store module."""

import datetime
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from deltalake import write_deltalake

from drilldown.feature_store import Column, ColumnType, Dataset, FeatureStore


@pytest.fixture
def sample_delta_table():
    """Create a sample Delta table for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create sample data
        df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "timestamp": pd.to_datetime(
                    [
                        "2024-01-01",
                        "2024-01-02",
                        "2024-01-03",
                        "2024-01-04",
                        "2024-01-05",
                    ]
                ),
                "value": [10.5, 20.3, 15.7, 25.1, 30.2],
                "category": ["A", "B", "A", "B", "A"],
            }
        )

        # Write to Delta table
        table_path = Path(tmpdir) / "test_table"
        write_deltalake(str(table_path), df)

        yield str(table_path)


def test_column_type_inference():
    """Test column type inference."""
    import pyarrow as pa

    # Test numerical inference
    int_type = pa.int64()
    assert Column.infer_column_type(int_type) == ColumnType.NUMERICAL

    float_type = pa.float64()
    assert Column.infer_column_type(float_type) == ColumnType.NUMERICAL

    # Test categorical inference
    string_type = pa.string()
    assert Column.infer_column_type(string_type) == ColumnType.CATEGORICAL

    # Test timestamp inference
    timestamp_type = pa.timestamp("us")
    assert Column.infer_column_type(timestamp_type) == ColumnType.TIMESTAMP

    # Test date inference
    date_type = pa.date32()
    assert Column.infer_column_type(date_type) == ColumnType.DATE


def test_column_from_metadata():
    """Test column creation from metadata."""
    import pyarrow as pa

    field = pa.field(
        "test_col",
        pa.int64(),
        metadata={"column_type": "primary_id", "description": "Test column"},
    )

    col = Column.from_arrow_field(field)
    assert col.name == "test_col"
    assert col.column_type == ColumnType.PRIMARY_ID
    assert col.description == "Test column"


def test_dataset_from_path(sample_delta_table):
    """Test creating a dataset from a Delta table path."""
    dataset = Dataset.from_path(sample_delta_table, name="test_dataset")

    assert dataset.name == "test_dataset"
    assert dataset.path == sample_delta_table
    assert len(dataset.columns) > 0
    assert "id" in dataset.columns
    assert "timestamp" in dataset.columns
    assert "value" in dataset.columns
    assert "category" in dataset.columns


def test_dataset_get_column_names_by_type(sample_delta_table):
    """Test getting column names by type."""
    dataset = Dataset.from_path(sample_delta_table)
    columns_by_type = dataset.get_column_names_by_type()

    assert len(columns_by_type) == 3
    primary_id, primary_timestamp, column_types = columns_by_type

    # Check that column_types has expected keys
    assert "numerical" in column_types
    assert "categorical" in column_types
    assert "timestamp" in column_types


def test_dataset_get_dataframe_date_range(sample_delta_table):
    """Test querying data by date range."""
    dataset = Dataset.from_path(sample_delta_table)

    start = datetime.datetime(2024, 1, 2)
    end = datetime.datetime(2024, 1, 4)

    df, partition_options = dataset.get_dataframe_date_range(start, end)

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    # Should include dates from Jan 2-4
    assert len(df) <= 3


def test_feature_store_initialization():
    """Test feature store initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a collection directory
        collection_path = Path(tmpdir) / "collection1"
        collection_path.mkdir()

        # Create a simple Delta table
        df = pd.DataFrame({"col1": [1, 2, 3]})
        table_path = collection_path / "dataset1"
        write_deltalake(str(table_path), df)

        # Initialize feature store
        fs = FeatureStore(collection_paths=[str(collection_path)])

        assert "collection1" in fs.collections
        assert "dataset1" in fs.collections["collection1"]


def test_feature_store_json_serialization():
    """Test feature store JSON serialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        collection_path = Path(tmpdir) / "collection1"
        collection_path.mkdir()

        df = pd.DataFrame({"col1": [1, 2, 3]})
        table_path = collection_path / "dataset1"
        write_deltalake(str(table_path), df)

        fs = FeatureStore(collection_paths=[str(collection_path)])
        json_str = fs.to_json()

        assert isinstance(json_str, str)
        assert len(json_str) > 0

        # Test deserialization
        fs2 = FeatureStore.model_validate_json(json_str)
        assert isinstance(fs2, FeatureStore)
        assert fs2.collection_paths == fs.collection_paths


# =====================================================
# URIHandler Tests
# =====================================================


class TestURIHandler:
    """Tests for URIHandler functionality."""

    def test_uri_handler_read_local_json_file(self):
        """Test reading a local JSON file."""
        import json

        from drilldown.feature_store.uri_handler import URIHandler

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test JSON file
            json_path = Path(tmpdir) / "test_curve.json"
            test_data = {"x": [1, 2, 3], "y": [4, 5, 6]}
            with open(json_path, "w") as f:
                json.dump(test_data, f)

            # Read with URIHandler
            handler = URIHandler()
            result = handler.read_json(str(json_path))

            assert result == test_data
            assert result["x"] == [1, 2, 3]
            assert result["y"] == [4, 5, 6]

    def test_uri_handler_read_local_image_file(self):
        """Test reading a local image file."""
        import numpy as np
        from PIL import Image

        from drilldown.feature_store.uri_handler import URIHandler

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test image
            img_path = Path(tmpdir) / "test_image.png"
            img_array = np.zeros((100, 100, 3), dtype=np.uint8)
            img_array[:, :] = [255, 0, 0]  # Red image
            Image.fromarray(img_array).save(img_path)

            # Read with URIHandler
            handler = URIHandler()
            result = handler.read_image(str(img_path))

            assert isinstance(result, np.ndarray)
            assert result.shape == (100, 100, 3)
            # Check it's predominantly red
            assert result[50, 50, 0] == 255  # Red channel
            assert result[50, 50, 1] == 0  # Green channel
            assert result[50, 50, 2] == 0  # Blue channel

    def test_uri_handler_read_curve_json(self):
        """Test reading curve data in JSON format."""
        import json

        from drilldown.feature_store.uri_handler import URIHandler

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test JSON curve file
            json_path = Path(tmpdir) / "curve.json"
            curve_data = {"x": [0, 1, 2, 3, 4], "y": [0, 1, 4, 9, 16]}
            with open(json_path, "w") as f:
                json.dump(curve_data, f)

            # Read with URIHandler
            handler = URIHandler()
            result = handler.read_curve(str(json_path), curve_format="json")

            assert result == curve_data

    def test_uri_handler_read_uri_with_column_type_uri_img(self):
        """Test read_uri with column_type=URI_IMG."""
        import numpy as np
        from PIL import Image

        from drilldown.feature_store.constants import ColumnType
        from drilldown.feature_store.uri_handler import URIHandler

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test image
            img_path = Path(tmpdir) / "test.png"
            img_array = np.zeros((50, 50, 3), dtype=np.uint8)
            img_array[:, :] = [0, 255, 0]  # Green image
            Image.fromarray(img_array).save(img_path)

            # Read with URIHandler using column_type
            handler = URIHandler()
            result = handler.read_uri(str(img_path), column_type=ColumnType.URI_IMG)

            assert isinstance(result, np.ndarray)
            assert result.shape == (50, 50, 3)

    def test_uri_handler_read_uri_with_column_type_uri_curve(self):
        """Test read_uri with column_type=URI_CURVE."""
        import json

        from drilldown.feature_store.constants import ColumnType
        from drilldown.feature_store.uri_handler import URIHandler

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test JSON curve file
            json_path = Path(tmpdir) / "curve.json"
            curve_data = {"points": [[0, 0], [1, 1], [2, 4]]}
            with open(json_path, "w") as f:
                json.dump(curve_data, f)

            # Read with URIHandler using column_type
            handler = URIHandler()
            result = handler.read_uri(str(json_path), column_type=ColumnType.URI_CURVE)

            assert result == curve_data

    def test_uri_handler_read_uri_with_object_format_image(self):
        """Test read_uri with column_type=OBJECT and object_format=png."""
        import numpy as np
        from PIL import Image

        from drilldown.feature_store.constants import ColumnType
        from drilldown.feature_store.uri_handler import URIHandler

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test image
            img_path = Path(tmpdir) / "test.png"
            img_array = np.zeros((30, 30, 3), dtype=np.uint8)
            img_array[:, :] = [0, 0, 255]  # Blue image
            Image.fromarray(img_array).save(img_path)

            # Read with URIHandler using OBJECT column type with image format
            handler = URIHandler()
            result = handler.read_uri(
                str(img_path), column_type=ColumnType.OBJECT, object_format="png"
            )

            assert isinstance(result, np.ndarray)
            assert result.shape == (30, 30, 3)

    def test_uri_handler_read_uri_with_object_format_json(self):
        """Test read_uri with column_type=OBJECT and object_format=json."""
        import json

        from drilldown.feature_store.constants import ColumnType
        from drilldown.feature_store.uri_handler import URIHandler

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test JSON file
            json_path = Path(tmpdir) / "data.json"
            test_data = {"key": "value", "number": 42}
            with open(json_path, "w") as f:
                json.dump(test_data, f)

            # Read with URIHandler using OBJECT column type with json format
            handler = URIHandler()
            result = handler.read_uri(
                str(json_path), column_type=ColumnType.OBJECT, object_format="json"
            )

            assert result == test_data

    def test_uri_handler_read_uri_raw_bytes(self):
        """Test read_uri returns raw bytes when no column_type is specified."""
        from drilldown.feature_store.uri_handler import URIHandler

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test file
            file_path = Path(tmpdir) / "test.txt"
            file_path.write_text("Hello, World!")

            # Read with URIHandler without column_type
            handler = URIHandler()
            result = handler.read_uri(str(file_path))

            assert isinstance(result, bytes)
            assert result == b"Hello, World!"

    def test_uri_handler_missing_file_error(self):
        """Test that reading a missing file raises an error."""
        from drilldown.feature_store.uri_handler import URIHandler

        handler = URIHandler()

        with pytest.raises(Exception):  # obstore raises various exceptions
            handler.read_json("/nonexistent/path/file.json")

    def test_uri_handler_unsupported_curve_format(self):
        """Test that unsupported curve format raises ValueError."""
        from drilldown.feature_store.uri_handler import URIHandler

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test file
            file_path = Path(tmpdir) / "test.xml"
            file_path.write_text("<data>test</data>")

            handler = URIHandler()

            with pytest.raises(ValueError, match="Unsupported curve format"):
                handler.read_curve(str(file_path), curve_format="xml")

    def test_uri_handler_read_multiple_files_same_directory(self):
        """Test reading multiple files from the same directory."""
        import json

        from drilldown.feature_store.uri_handler import URIHandler

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create two files in the same directory
            dir1 = Path(tmpdir) / "dir1"
            dir1.mkdir()
            file1 = dir1 / "file1.json"
            file2 = dir1 / "file2.json"

            file1.write_text(json.dumps({"file": 1}))
            file2.write_text(json.dumps({"file": 2}))

            handler = URIHandler()

            # Read both files
            result1 = handler.read_json(str(file1))
            result2 = handler.read_json(str(file2))

            assert result1 == {"file": 1}
            assert result2 == {"file": 2}


# =====================================================
# Year/Week and Year/Month Column Type Tests
# =====================================================


class TestYearWeekYearMonthColumnTypes:
    """Tests for year_week and year_month column types."""

    def test_column_type_inference_year_week_from_metadata(self):
        """Test that year_week column type is inferred from metadata."""
        import pyarrow as pa

        field = pa.field(
            "week_col",
            pa.string(),
            metadata={"column_type": "year_week", "description": "Week column"},
        )

        col = Column.from_arrow_field(field)
        assert col.name == "week_col"
        assert col.column_type == ColumnType.YEAR_WEEK
        assert col.description == "Week column"

    def test_column_type_inference_year_month_from_metadata(self):
        """Test that year_month column type is inferred from metadata."""
        import pyarrow as pa

        field = pa.field(
            "month_col",
            pa.string(),
            metadata={"column_type": "year_month", "description": "Month column"},
        )

        col = Column.from_arrow_field(field)
        assert col.name == "month_col"
        assert col.column_type == ColumnType.YEAR_MONTH
        assert col.description == "Month column"

    def test_dataset_with_year_week_partition(self):
        """Test dataset with year_week partition column."""
        import pyarrow as pa

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create data with year_week column
            df = pd.DataFrame(
                {
                    "id": [1, 2, 3, 4, 5],
                    "timestamp": pd.to_datetime(
                        [
                            "2024-01-01",
                            "2024-01-08",
                            "2024-01-15",
                            "2024-01-22",
                            "2024-01-29",
                        ]
                    ),
                    "year_week": [
                        "2024-cw01",
                        "2024-cw02",
                        "2024-cw03",
                        "2024-cw04",
                        "2024-cw05",
                    ],
                    "value": [10, 20, 30, 40, 50],
                }
            )

            # Create schema with metadata
            schema = pa.schema(
                [
                    pa.field("id", pa.int64()),
                    pa.field("timestamp", pa.timestamp("us")),
                    pa.field(
                        "year_week",
                        pa.string(),
                        metadata={b"column_type": b"year_week"},
                    ),
                    pa.field("value", pa.int64()),
                ]
            )

            # Write to Delta table with partitioning
            table_path = Path(tmpdir) / "test_table"
            table = pa.Table.from_pandas(df, schema=schema)
            write_deltalake(str(table_path), table, partition_by=["year_week"])

            # Load dataset and check
            dataset = Dataset.from_path(str(table_path))

            assert "year_week" in dataset.columns
            assert dataset.columns["year_week"].column_type == ColumnType.YEAR_WEEK
            assert "year_week" in dataset.year_week_columns

    def test_dataset_with_year_month_partition(self):
        """Test dataset with year_month partition column."""
        import pyarrow as pa

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create data with year_month column
            df = pd.DataFrame(
                {
                    "id": [1, 2, 3, 4, 5],
                    "timestamp": pd.to_datetime(
                        [
                            "2024-01-15",
                            "2024-02-15",
                            "2024-03-15",
                            "2024-04-15",
                            "2024-05-15",
                        ]
                    ),
                    "year_month": [
                        "2024-01",
                        "2024-02",
                        "2024-03",
                        "2024-04",
                        "2024-05",
                    ],
                    "value": [100, 200, 300, 400, 500],
                }
            )

            # Create schema with metadata
            schema = pa.schema(
                [
                    pa.field("id", pa.int64()),
                    pa.field("timestamp", pa.timestamp("us")),
                    pa.field(
                        "year_month",
                        pa.string(),
                        metadata={b"column_type": b"year_month"},
                    ),
                    pa.field("value", pa.int64()),
                ]
            )

            # Write to Delta table with partitioning
            table_path = Path(tmpdir) / "test_table"
            table = pa.Table.from_pandas(df, schema=schema)
            write_deltalake(str(table_path), table, partition_by=["year_month"])

            # Load dataset and check
            dataset = Dataset.from_path(str(table_path))

            assert "year_month" in dataset.columns
            assert dataset.columns["year_month"].column_type == ColumnType.YEAR_MONTH
            assert "year_month" in dataset.year_month_columns

    def test_date_range_query_with_year_week_partition(self):
        """Test querying with date range on year_week partitioned data."""
        import pyarrow as pa

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create data with year_week column
            df = pd.DataFrame(
                {
                    "id": [1, 2, 3, 4, 5, 6],
                    "timestamp": pd.to_datetime(
                        [
                            "2024-01-01",
                            "2024-01-08",
                            "2024-01-15",
                            "2024-01-22",
                            "2024-01-29",
                            "2024-02-05",
                        ]
                    ),
                    "year_week": [
                        "2024-cw01",
                        "2024-cw02",
                        "2024-cw03",
                        "2024-cw04",
                        "2024-cw05",
                        "2024-cw06",
                    ],
                    "value": [10, 20, 30, 40, 50, 60],
                }
            )

            # Create schema with metadata
            schema = pa.schema(
                [
                    pa.field("id", pa.int64()),
                    pa.field(
                        "timestamp",
                        pa.timestamp("us"),
                        metadata={b"column_type": b"primary_timestamp"},
                    ),
                    pa.field(
                        "year_week",
                        pa.string(),
                        metadata={b"column_type": b"year_week"},
                    ),
                    pa.field("value", pa.int64()),
                ]
            )

            # Write to Delta table with partitioning
            table_path = Path(tmpdir) / "test_table"
            table = pa.Table.from_pandas(df, schema=schema)
            write_deltalake(str(table_path), table, partition_by=["year_week"])

            # Load dataset
            dataset = Dataset.from_path(str(table_path))

            # Query for Jan 8-22 (weeks 2-4)
            start = datetime.datetime(2024, 1, 8)
            end = datetime.datetime(2024, 1, 22)

            result_df, _ = dataset.get_dataframe_date_range(start, end)

            assert len(result_df) >= 1  # Should get at least some rows
            # The filtering should include weeks 2, 3, 4

    def test_date_range_query_with_year_month_partition(self):
        """Test querying with date range on year_month partitioned data."""
        import pyarrow as pa

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create data with year_month column
            df = pd.DataFrame(
                {
                    "id": [1, 2, 3, 4],
                    "timestamp": pd.to_datetime(
                        ["2024-01-15", "2024-02-15", "2024-03-15", "2024-04-15"]
                    ),
                    "year_month": ["2024-01", "2024-02", "2024-03", "2024-04"],
                    "value": [100, 200, 300, 400],
                }
            )

            # Create schema with metadata
            schema = pa.schema(
                [
                    pa.field("id", pa.int64()),
                    pa.field(
                        "timestamp",
                        pa.timestamp("us"),
                        metadata={b"column_type": b"primary_timestamp"},
                    ),
                    pa.field(
                        "year_month",
                        pa.string(),
                        metadata={b"column_type": b"year_month"},
                    ),
                    pa.field("value", pa.int64()),
                ]
            )

            # Write to Delta table with partitioning
            table_path = Path(tmpdir) / "test_table"
            table = pa.Table.from_pandas(df, schema=schema)
            write_deltalake(str(table_path), table, partition_by=["year_month"])

            # Load dataset
            dataset = Dataset.from_path(str(table_path))

            # Query for Feb-Mar
            start = datetime.datetime(2024, 2, 1)
            end = datetime.datetime(2024, 3, 31)

            result_df, _ = dataset.get_dataframe_date_range(start, end)

            assert len(result_df) >= 1  # Should get at least some rows

    def test_year_week_string_comparison_across_years(self):
        """Test that year_week string comparison works correctly across year boundaries."""
        # This tests that "2024-cw52" < "2025-cw01" (lexicographic ordering)
        assert "2024-cw52" < "2025-cw01"
        assert "2023-cw53" < "2024-cw01"
        assert "2024-cw01" < "2024-cw52"

    def test_year_month_string_comparison_across_years(self):
        """Test that year_month string comparison works correctly across year boundaries."""
        # This tests that "2024-12" < "2025-01" (lexicographic ordering)
        assert "2024-12" < "2025-01"
        assert "2023-12" < "2024-01"
        assert "2024-01" < "2024-12"


# =====================================================
# Column.read_uri_object Tests
# =====================================================


class TestColumnReadUriObject:
    """Tests for Column.read_uri_object method."""

    def test_column_read_uri_object_image(self):
        """Test reading an image through Column.read_uri_object."""
        import numpy as np
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test image
            img_path = Path(tmpdir) / "test.png"
            img_array = np.zeros((40, 40, 3), dtype=np.uint8)
            img_array[:, :] = [128, 128, 128]  # Gray image
            Image.fromarray(img_array).save(img_path)

            # Create a Column with uri_img type
            col = Column(
                name="image_col",
                column_type=ColumnType.URI_IMG,
                description="Image column",
            )

            # Read through Column
            result = col.read_uri_object(str(img_path))

            assert isinstance(result, np.ndarray)
            assert result.shape == (40, 40, 3)

    def test_column_read_uri_object_curve(self):
        """Test reading a curve through Column.read_uri_object."""
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test JSON curve file
            json_path = Path(tmpdir) / "curve.json"
            curve_data = {"x": [0, 1, 2], "y": [0, 2, 4]}
            with open(json_path, "w") as f:
                json.dump(curve_data, f)

            # Create a Column with uri_curve type
            col = Column(
                name="curve_col",
                column_type=ColumnType.URI_CURVE,
                description="Curve column",
                object_format="json",
            )

            # Read through Column
            result = col.read_uri_object(str(json_path))

            assert result == curve_data
