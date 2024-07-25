from typing import Dict, Iterable

from datasets import Dataset, concatenate_datasets

from embedding_studio.embeddings.augmentations.augmentation_with_random_selection import (
    AugmentationWithRandomSelection,
)
from embedding_studio.embeddings.data.items.items_set import ItemsSet


class ItemsSetAugmentationApplier:
    """Class to apply an augmentation to ItemsSet, and generate more augmented examples

    The ItemsSetAugmentationApplier class is designed to augment the items stored in an ItemsSet instance.
    It uses a specified augmentation strategy to transform the original items and generate additional
    augmented examples. The augmented items are then stored alongside the original items in a new
    ItemsSet instance.

    :param augmentation: An instance of an AugmentationWithRandomSelection class, which defines the transformation to be
                         applied to the items in the items_set.
    """

    def __init__(self, augmentation: AugmentationWithRandomSelection):
        self.augmentation = augmentation

    def _augment(self, items_set: ItemsSet) -> Iterable[Dict]:
        """Private method to apply the augmentation transformation to each item in the items_set.

        :param items_set: An instance of ItemsSet containing the original items.
        :return: An iterable of dictionaries, where each dictionary represents an augmented item
                 along with its corresponding ID.
        """
        for row in items_set:
            for object in self.augmentation.transform(
                row[items_set.item_field_name]
            ):
                yield {
                    items_set.id_field_name: row[items_set.id_field_name],
                    items_set.item_field_name: object,
                }

    def apply_augmentation(self, items_set: ItemsSet) -> ItemsSet:
        """Method to apply the augmentation and return a new ItemsSet with both original and augmented items.

        :param items_set: An instance of ItemsSet containing the original items.
        :return: A new instance of ItemsSet containing both the original and the augmented items.
        """
        return ItemsSet(
            concatenate_datasets(
                [
                    items_set.data,
                    Dataset.from_generator(lambda: self._augment(items_set)),
                ]
            ),
            item_field_name=items_set.item_field_name,
            id_field_name=items_set.id_field_name,
        )
