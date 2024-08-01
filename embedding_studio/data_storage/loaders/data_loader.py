from abc import ABC, abstractmethod
from typing import Generator, List, Optional, Type

from datasets import Dataset

from embedding_studio.data_storage.loaders.downloaded_item import (
    DownloadedItem,
)
from embedding_studio.data_storage.loaders.item_meta import ItemMeta


class DataLoader(ABC):
    def __init__(self, **kwargs):
        pass

    @property
    @abstractmethod
    def item_meta_cls(self) -> Type[ItemMeta]:
        raise NotImplemented

    @abstractmethod
    def load(self, items_data: List[ItemMeta]) -> Dataset:
        raise NotImplemented

    @abstractmethod
    def load_items(self, items_data: List[ItemMeta]) -> List[DownloadedItem]:
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
        """
        raise NotImplemented

    @abstractmethod
    def total_count(self, **kwargs) -> Optional[int]:
        """Abstract method to retrieve total count of items.

        :return: int if count is accessible, or None if is not.
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
