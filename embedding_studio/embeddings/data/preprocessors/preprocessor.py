from abc import ABC, abstractmethod
from typing import Any

from datasets import DatasetDict


class ItemsDatasetDictPreprocessor(ABC):
    """Interface for preprocessing dataset dictionaries containing items.

    This abstract base class defines the interface for preprocessors that convert
    and transform datasets into standardized formats for embedding models.
    """

    @abstractmethod
    def convert(self, dataset: DatasetDict) -> DatasetDict:
        """Convert a dataset dictionary into a processed format.

        :param dataset: The original dataset dictionary to be preprocessed
        :return: Processed DatasetDict with normalized fields and transformed items

        Example implementation:
        ```
        def convert(self, dataset: DatasetDict) -> DatasetDict:
            # Normalize fields
            dataset = self._field_normalizer(dataset)

            # Create ItemsSets
            items_sets = {}
            for key in dataset.keys():
                items_sets[key] = ItemsSet(
                    dataset[key],
                    "feature_name",
                    self._field_normalizer.id_field_name
                )

            # Apply transformations
            result = DatasetDict(items_sets)
            result = result.with_transform(
                lambda examples: some_transform_function(examples)
            )

            return result
        ```
        """
        raise NotImplemented()

    @abstractmethod
    def __call__(self, item: Any) -> Any:
        """Transform a single item using the preprocessor.

        :param item: The input item to be processed (type depends on implementation)
        :return: Processed item (type depends on implementation)

        Example implementation:
        ```
        def __call__(self, item: str) -> str:
            return self._transform(item)
        ```
        """
        raise NotImplemented()

    @abstractmethod
    def get_id_field_name(self) -> str:
        """Get the field name used for item identification.

        :return: The name of the field used as the item identifier

        Example implementation:
        ```
        def get_id_field_name(self) -> str:
            return self._field_normalizer.id_field_name
        ```
        """
        raise NotImplemented()
