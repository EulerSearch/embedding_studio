from typing import Callable, Dict, Optional

from datasets import DatasetDict

from embedding_studio.embeddings.data.preprocessors.preprocessor import (
    ItemsDatasetDictPreprocessor,
)
from embedding_studio.embeddings.data.storages.storage import ItemsStorage
from embedding_studio.embeddings.data.transforms.text.dummy import do_nothing
from embedding_studio.embeddings.data.transforms.text.transforms import (
    text_transforms,
)
from embedding_studio.embeddings.data.utils.fields_normalizer import (
    DatasetFieldsNormalizer,
)


class TextItemsDatasetDictPreprocessor(ItemsDatasetDictPreprocessor):
    TEXTS_FEATURE_NAME = "text"

    def __init__(
        self,
        field_normalizer: DatasetFieldsNormalizer,
        transform: Optional[Callable] = do_nothing,
    ):
        """Preprocessor for dict data storages.

        :param field_normalizer: how to normalize field names
        :param transform: function to get dict line out of dict object
        """
        self._field_normalizer = field_normalizer
        self._transform = transform

    def get_id_field_name(self) -> str:
        return self._field_normalizer.id_field_name

    def __call__(self, item: str) -> str:
        return self._transform(item)

    def convert(self, dataset: DatasetDict) -> DatasetDict:
        """Normalize fields, apply text transforms and create text items storages.

        :param dataset: dataset dict to be preprocessed
        :return: train/test DatasetDict with ItemsStorage as values
        """
        dataset: DatasetDict = self._field_normalizer(dataset)

        storages: Dict[ItemsStorage] = {}
        # TODO: use more optimal way to iterate over DatasetDict
        for key in dataset.keys():
            storages[key] = ItemsStorage(
                dataset[key],
                TextItemsDatasetDictPreprocessor.TEXTS_FEATURE_NAME,
                self._field_normalizer.id_field_name,
            )

        storages: DatasetDict = DatasetDict(storages)
        storages = storages.with_transform(
            lambda examples: text_transforms(
                examples,
                transform=self._transform,
                text_values_name=TextItemsDatasetDictPreprocessor.TEXTS_FEATURE_NAME,
            )
        )

        return storages
