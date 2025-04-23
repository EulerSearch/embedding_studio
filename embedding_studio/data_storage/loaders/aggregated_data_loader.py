from collections import defaultdict
from typing import Dict, Generator, List, Optional, Type

from datasets import Dataset

from embedding_studio.data_storage.loaders.data_loader import DataLoader
from embedding_studio.data_storage.loaders.downloaded_item import (
    DownloadedItem,
)
from embedding_studio.data_storage.loaders.item_meta import (
    ItemMetaWithSourceInfo,
)


class AggregatedDataLoader(DataLoader):
    """
    A DataLoader implementation that aggregates multiple data loaders.

    AggregatedDataLoader combines multiple DataLoader instances, allowing data to be
    loaded from different sources through a single interface. It routes load requests
    to the appropriate loader based on the source name in the item metadata.

    :param loaders: Dictionary mapping source names to their respective DataLoader instances
    :param item_meta_cls: The ItemMetaWithSourceInfo class type to use for metadata
    """

    def __init__(
        self,
        loaders: Dict[str, DataLoader],
        item_meta_cls: Type[ItemMetaWithSourceInfo],
    ) -> None:
        """
        Initialize an AggregatedDataLoader with multiple data loaders.

        :param loaders: Dictionary mapping source names to their respective DataLoader instances
        :param item_meta_cls: The ItemMetaWithSourceInfo class type to use for metadata
        """
        self.loaders = loaders
        self._item_meta_cls = item_meta_cls

    @property
    def item_meta_cls(self):
        """
        Returns the ItemMeta class used by this loader.

        :return: The ItemMetaWithSourceInfo class type used for metadata in this loader
        """
        return self._item_meta_cls

    def load(self, items_data: List[ItemMetaWithSourceInfo]) -> Dataset:
        """
        Load data items from multiple sources and combine them into a single dataset.

        This method groups items by source, delegates loading to the appropriate loader
        for each source, and then concatenates the results into a single dataset.

        :param items_data: List of ItemMetaWithSourceInfo objects identifying the items to load
        :return: A unified Dataset containing data from all sources
        """
        grouped_items_data = defaultdict(list)
        for item in items_data:
            grouped_items_data[item.source_name].append(item)

        results = []
        for key, items in grouped_items_data.items():
            results += self.loaders[key].load(items)

        return (
            results[0].concatenate(results[1:])
            if results
            else Dataset.from_dict({})
        )

    def load_items(
        self, items_data: List[ItemMetaWithSourceInfo]
    ) -> List[DownloadedItem]:
        """
        Load individual items from multiple sources.

        This method groups items by source and delegates loading to the
        appropriate loader for each source, then combines the results.

        :param items_data: List of ItemMetaWithSourceInfo objects identifying the items to load
        :return: Combined list of DownloadedItem objects from all sources
        """
        grouped_items_data = defaultdict(list)
        for item in items_data:
            grouped_items_data[item.source_name].append(item)

        results = []
        for key, items in grouped_items_data.items():
            results += self.loaders[key].load_items(items)

        return results

    def _load_batch_with_offset(
        self, offset: int, batch_size: int, **kwargs
    ) -> List[DownloadedItem]:
        """
        Load a batch of data items from all sources starting from the given offset.

        This method gets batches from all loaders and combines them into a single batch.

        :param offset: The offset from where to start loading items
        :param batch_size: The number of items to load in a single batch
        :param kwargs: Additional parameters for customizing the batch loading process
        :return: A combined list of DownloadedItem objects from all sources
        """
        all_batches = []
        for loader in self.loaders:
            batch = loader._load_batch_with_offset(
                offset, batch_size, **kwargs
            )
            all_batches.extend(batch)
        return all_batches

    def total_count(self, **kwargs) -> Optional[int]:
        """
        Calculate the total count of items across all loaders.

        :param kwargs: Additional parameters passed to each loader's total_count method
        :return: Sum of all available counts, or None if no counts are available
        """
        counts = [loader.total_count(**kwargs) for loader in self.loaders]
        return sum(filter(None, counts)) if any(counts) else None

    def load_all(
        self, batch_size: int, **kwargs
    ) -> Generator[List[DownloadedItem], None, None]:
        """
        A generator that iteratively loads all data in batches from all sources.

        This method overrides the base implementation to get batches from all
        loaders for each offset.

        :param batch_size: The size of each batch to load
        :param kwargs: Additional parameters for customizing the batch loading process
        :yield: Each combined batch as a list of DownloadedItem objects
        """
        offset = 0
        while True:
            all_batches = []
            for loader in self.loaders:
                batch = loader._load_batch_with_offset(
                    offset, batch_size, **kwargs
                )
                all_batches.extend(batch)
            if not all_batches:
                break
            yield all_batches
            offset += batch_size
