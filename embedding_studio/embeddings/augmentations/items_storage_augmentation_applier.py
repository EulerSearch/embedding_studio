from typing import Dict, Iterable

from datasets import Dataset, concatenate_datasets

from embedding_studio.embeddings.augmentations.augmentation_with_random_selection import (
    AugmentationWithRandomSelection,
)
from embedding_studio.embeddings.data.storages.storage import ItemsStorage


class ItemsStorageAugmentationApplier:
    """Class to apply an augmentation to ItemStorage, and generate more augmented examples

    The ItemsStorageAugmentationApplier class is designed to augment the items stored in an ItemsStorage instance.
    It uses a specified augmentation strategy to transform the original items and generate additional
    augmented examples. The augmented items are then stored alongside the original items in a new
    ItemsStorage instance.

    :param augmentation: An instance of an AugmentationWithRandomSelection class, which defines the transformation to be
                         applied to the items in the storage.
    """

    def __init__(self, augmentation: AugmentationWithRandomSelection):
        self.augmentation = augmentation

    def _augment(self, storage: ItemsStorage) -> Iterable[Dict]:
        """Private method to apply the augmentation transformation to each item in the storage.

        :param storage: An instance of ItemsStorage containing the original items.
        :return: An iterable of dictionaries, where each dictionary represents an augmented item
                 along with its corresponding ID.
        """
        for row in storage:
            for object in self.augmentation.transform(
                row[storage.item_field_name]
            ):
                yield {
                    storage.id_field_name: row[storage.id_field_name],
                    storage.item_field_name: object,
                }

    def apply_augmentation(self, storage: ItemsStorage) -> ItemsStorage:
        """Method to apply the augmentation and return a new ItemsStorage with both original and augmented items.

        :param storage: An instance of ItemsStorage containing the original items.
        :return: A new instance of ItemsStorage containing both the original and the augmented items.
        """
        return ItemsStorage(
            concatenate_datasets(
                [
                    storage.data,
                    Dataset.from_generator(lambda: self._augment(storage)),
                ]
            ),
            item_field_name=storage.item_field_name,
            id_field_name=storage.id_field_name,
        )
