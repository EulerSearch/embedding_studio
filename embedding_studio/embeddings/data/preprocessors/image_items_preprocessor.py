from typing import Callable, Dict, Optional

import numpy as np
from datasets import DatasetDict
from PIL.Image import Image

from embedding_studio.embeddings.data.items.items_set import ItemsSet
from embedding_studio.embeddings.data.preprocessors.preprocessor import (
    ItemsDatasetDictPreprocessor,
)
from embedding_studio.embeddings.data.transforms.image.center_padded import (
    resize_by_longest_and_pad_transform,
)
from embedding_studio.embeddings.data.transforms.image.transforms import (
    image_transforms,
)
from embedding_studio.embeddings.data.utils.fields_normalizer import (
    DatasetFieldsNormalizer,
)


class ImageItemsDatasetDictPreprocessor(ItemsDatasetDictPreprocessor):
    IMAGES_FEATURE_PIXEL_NAME = "pixel_values"

    def __init__(
        self,
        field_normalizer: DatasetFieldsNormalizer,
        n_pixels: int = 224,
        transform: Optional[Callable] = resize_by_longest_and_pad_transform,
    ):
        """Preprocessor for image data items.

        :param field_normalizer: how to normalize field names
        :param n_pixels: side size
        :param transform: function to get pixels (np.array) out of images
        """
        self._field_normalizer = field_normalizer

        if not isinstance(n_pixels, int) or n_pixels <= 0:
            raise ValueError(
                f"Num of pixels {n_pixels} should be a positive integer"
            )
        self._n_pixels = n_pixels

        self._transform = transform

    def __call__(self, item: Image) -> np.ndarray:
        return self._transform(self._n_pixels)(item)

    def get_id_field_name(self) -> str:
        return self._field_normalizer.id_field_name

    def convert(self, dataset: DatasetDict) -> DatasetDict:
        """Normalize fields, apply image transforms and create items sets.

        :param dataset: dataset dict to be preprocessed
        :return: train/test DatasetDict with ItemsSet as values
        """
        dataset: DatasetDict = self._field_normalizer(dataset)

        sets: Dict[ItemsSet] = {}
        # TODO: use more optimal way to iterate over DatasetDict
        for key in dataset.keys():
            sets[key] = ItemsSet(
                dataset[key],
                ImageItemsDatasetDictPreprocessor.IMAGES_FEATURE_PIXEL_NAME,
                self._field_normalizer.id_field_name,
            )

        sets: DatasetDict = DatasetDict(sets)
        sets = sets.with_transform(
            lambda examples: image_transforms(
                examples,
                transform=self._transform,
                n_pixels=self._n_pixels,
                image_field_name=self._field_normalizer.item_field_name,
                pixel_values_name=ImageItemsDatasetDictPreprocessor.IMAGES_FEATURE_PIXEL_NAME,
            )
        )

        return sets
