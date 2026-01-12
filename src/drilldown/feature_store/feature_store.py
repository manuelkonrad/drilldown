# SPDX-FileCopyrightText: 2026 Manuel Konrad
#
# SPDX-License-Identifier: MIT

"""Feature store with Delta Lake backend."""

import logging
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from drilldown.feature_store.dataset import Dataset

logger = logging.getLogger(__name__)


class FeatureStore(BaseModel):
    """Feature store managing collections and datasets."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    collection_paths: list[str] = Field(default_factory=list)
    collections: dict[str, dict[str, Dataset]] = Field(default_factory=dict)

    def __init__(self, **data):
        """Initialize the feature store and load collections."""
        super().__init__(**data)
        self._load_collections()

    def _load_collections(self):
        """Load all collections from the collection paths."""
        for collection_path in self.collection_paths:
            collection_path_obj = Path(collection_path)
            if not collection_path_obj.exists():
                continue

            collection_name = collection_path_obj.name
            if collection_name not in self.collections:
                self.collections[collection_name] = {}

            # Find all Delta tables in the collection directory
            for dataset_path in collection_path_obj.iterdir():
                if dataset_path.is_dir():
                    # Check if it's a Delta table by looking for _delta_log
                    delta_log_path = dataset_path / "_delta_log"
                    if delta_log_path.exists():
                        dataset_name = dataset_path.name
                        try:
                            dataset = Dataset.from_path(
                                str(dataset_path),
                                name=dataset_name,
                            )
                            self.collections[collection_name][dataset_name] = dataset
                        except Exception as e:
                            # Skip datasets that fail to load
                            logger.warning(
                                f"Failed to load dataset {dataset_name}: {e}"
                            )
                            continue

    def to_json(self, **kwargs) -> str:
        """Serialize to JSON string."""
        return self.model_dump_json(**kwargs)
