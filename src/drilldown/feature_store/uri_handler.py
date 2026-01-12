# SPDX-FileCopyrightText: 2026 Manuel Konrad
#
# SPDX-License-Identifier: MIT

"""URI handler for reading objects from local paths and cloud storage."""

import json
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import yaml
from obstore.store import LocalStore, from_url
from PIL import Image

from drilldown.constants import CURVE_FORMATS, IMAGE_FORMATS
from drilldown.feature_store.constants import ColumnType


class URIHandler:
    """Handler for reading objects from URIs using obstore (local or cloud).

    Create a new instance directly before each use rather than reusing instances.
    """

    def _get_store(self, uri: str) -> tuple[Any, str]:
        """Create an obstore for the given URI, returning (store, object_path)."""
        parsed = urlparse(uri)

        if not parsed.scheme or parsed.scheme == "file":
            path = Path(uri.replace("file://", ""))
            directory = str(path.parent.absolute())
            file_name = path.name

            store = LocalStore(prefix=directory)
            return store, file_name

        store = from_url(uri)
        object_path = parsed.path.lstrip("/")
        return store, object_path

    def read_uri(
        self, uri: str, column_type: str | None = None, object_format: str | None = None
    ) -> Any:
        """Read and deserialize from a URI based on column type and format."""
        if column_type:
            if column_type == ColumnType.URI_IMG:
                return self.read_image(uri)
            elif column_type == ColumnType.URI_CURVE:
                return self.read_curve(uri, object_format or "json")
            elif column_type == ColumnType.OBJECT:
                if object_format in IMAGE_FORMATS:
                    return self.read_image(uri)
                elif object_format in CURVE_FORMATS:
                    return self.read_curve(uri, object_format)

        store, object_path = self._get_store(uri)
        result = store.get(object_path)
        return bytes(result.bytes())

    def read_image(self, uri: str) -> np.ndarray:
        """Read an image from a URI and return as numpy array."""
        store, object_path = self._get_store(uri)
        result = store.get(object_path)
        data = bytes(result.bytes())
        img = Image.open(BytesIO(data))
        return np.array(img)

    def read_json(self, uri: str) -> Any:
        """Read and parse JSON from a URI."""
        store, object_path = self._get_store(uri)
        result = store.get(object_path)
        data = bytes(result.bytes())
        return json.loads(data.decode("utf-8"))

    def read_curve(self, uri: str, curve_format: str = "json") -> Any:
        """Read curve data from a URI in the specified format (json or yaml)."""
        if curve_format == "json":
            return self.read_json(uri)
        elif curve_format == "yaml":
            store, object_path = self._get_store(uri)
            result = store.get(object_path)
            data = bytes(result.bytes())

            try:
                return yaml.safe_load(data.decode("utf-8"))
            except ImportError:
                raise ImportError("PyYAML is required to read YAML curve data")
        else:
            raise ValueError(f"Unsupported curve format: {curve_format}")
