from typing import Callable, Dict, Optional

from datasets import DatasetDict

from embedding_studio.embeddings.data.preprocessors.preprocessor import (
    ItemsDatasetDictPreprocessor,
)
from embedding_studio.embeddings.data.storages.storage import ItemsStorage
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
        """Preprocessor for image data storages.

        :param field_normalizer: how to normalize field names
        :type field_normalizer: DatasetFieldsNormalizer
        :param n_pixels: side size
        :type n_pixels: int
        :param transform: function to get pixels (np.array) out of images
        :type transform: Optional[Callable]
        """
        self.field_normalizer = field_normalizer
        self.n_pixels = n_pixels
        self.transform = transform

    def get_id_field_name(self) -> str:
        return self.field_normalizer.id_field_name

    def convert(self, dataset: DatasetDict) -> DatasetDict:
        """Normalize fields, apply image transforms and create items storages.

        :param dataset: dataset dict to be preprocessed
        :type dataset: DatasetDict
        :return: train/test DatasetDict with ItemsStorage as values
        :rtype: DatasetDict
        """
        dataset: DatasetDict = self.field_normalizer(dataset)

        storages: Dict[ItemsStorage] = {}
        # TODO: use more optimal way to iterate over DatasetDict
        for key in dataset.keys():
            storages[key] = ItemsStorage(
                dataset[key],
                ImageItemsDatasetDictPreprocessor.IMAGES_FEATURE_PIXEL_NAME,
                self.field_normalizer.id_field_name,
            )

        storages: DatasetDict = DatasetDict(storages)
        storages = storages.with_transform(
            lambda examples: image_transforms(
                examples,
                transform=self.transform,
                n_pixels=self.n_pixels,
                image_field_name=self.field_normalizer.item_field_name,
                pixel_values_name=ImageItemsDatasetDictPreprocessor.IMAGES_FEATURE_PIXEL_NAME,
            )
        )

        return storages
