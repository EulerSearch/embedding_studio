from abc import ABC, abstractmethod
from typing import Generator, List, Optional, Type

from datasets import Dataset

from embedding_studio.data_storage.loaders.downloaded_item import (
    DownloadedItem,
)
from embedding_studio.data_storage.loaders.item_meta import ItemMeta


class DataLoader(ABC):
    """
    Abstract base class defining the interface for data loaders.

    DataLoader provides methods to load and manage datasets from various sources.
    It handles both batch-wise loading and complete dataset loading, allowing for
    efficient management of potentially large datasets.

    :param kwargs: Additional keyword arguments for configuring the loader
    """

    def __init__(self, **kwargs):
        pass

    @property
    @abstractmethod
    def item_meta_cls(self) -> Type[ItemMeta]:
        """
        Abstract property that returns the ItemMeta class used by this loader.

        :return: The ItemMeta class type used for metadata in this loader

        Example implementation:
        ```python
        @property
        def item_meta_cls(self):
            return FileItemMeta
        ```
        """
        raise NotImplemented

    @abstractmethod
    def load(self, items_data: List[ItemMeta]) -> Dataset:
        """
        Abstract method to load a full dataset from the provided item metadata.

        :param items_data: List of ItemMeta objects identifying the items to load
        :return: A Dataset object containing the loaded data

        Example implementation:
        ```python
        def load(self, items_data: List[ItemMeta]) -> Dataset:
            downloaded_items = self.load_items(items_data)
            data_dict = {
                "id": [item.id for item in downloaded_items],
                "text": [item.data for item in downloaded_items],
                "metadata": [item.meta.dict() for item in downloaded_items]
            }
            return Dataset.from_dict(data_dict)
        ```
        """
        raise NotImplemented

    @abstractmethod
    def load_items(self, items_data: List[ItemMeta]) -> List[DownloadedItem]:
        """
        Abstract method to load individual items from their metadata.

        :param items_data: List of ItemMeta objects identifying the items to load
        :return: List of DownloadedItem objects containing the loaded data and metadata

        Example implementation:
        ```python
        def load_items(self, items_data: List[ItemMeta]) -> List[DownloadedItem]:
            result = []
            for item_meta in items_data:
                data = self._fetch_data(item_meta.id)
                result.append(DownloadedItem(
                    id=item_meta.id,
                    data=data,
                    meta=item_meta
                ))
            return result
        ```
        """
        raise NotImplemented

    @abstractmethod
    def _load_batch_with_offset(
        self, offset: int, batch_size: int, **kwargs
    ) -> List[DownloadedItem]:
        """
        Abstract method to load a batch of data items, starting from the given offset.

        :param offset: The offset from where to start loading items.
        :param batch_size: The number of items to load in a single batch.
        :return: A list of tuples with each tuple containing an item's ID, data and metadata.

        Example implementation:
        ```python
        def _load_batch_with_offset(
            self, offset: int, batch_size: int, **kwargs
        ) -> List[DownloadedItem]:
            # Get item metadata for this batch
            item_metas = self._get_metadata_slice(offset, batch_size, **kwargs)
            # Load the actual items using the metadata
            return self.load_items(item_metas)
        ```
        """
        raise NotImplemented

    @abstractmethod
    def total_count(self, **kwargs) -> Optional[int]:
        """Abstract method to retrieve total count of items.

        :return: int if count is accessible, or None if is not.

        Example implementation:
        ```python
        def total_count(self, **kwargs) -> Optional[int]:
            try:
                return len(self._get_all_metadata(**kwargs))
            except Exception:
                return None
        ```
        """
        return None

    def load_all(
        self, batch_size: int, **kwargs
    ) -> Generator[List[DownloadedItem], None, None]:
        """
        A generator that iteratively loads batches using the `load_batch` method.
        This allows for managing large datasets by processing them in manageable chunks.
        Each batch is yielded to the caller, which can handle or process the batch as needed.

        :param batch_size: The size of each batch to load.
        :yield: Each batch as a list of tuples (id, data, item_info).
        """
        offset = 0
        while True:
            current_batch = self._load_batch_with_offset(offset, batch_size)
            if not current_batch:
                break  # Stop yielding if no more data is returned.
            yield current_batch
            offset += batch_size
