from typing import Callable, Dict, Optional

from datasets import DatasetDict

from embedding_studio.embeddings.data.preprocessors.preprocessor import (
    ItemsDatasetDictPreprocessor,
)
from embedding_studio.embeddings.data.storages.storage import ItemsStorage
from embedding_studio.embeddings.data.transforms.dict.line_from_dict import (
    get_text_line_from_dict,
)
from embedding_studio.embeddings.data.transforms.dict.transforms import (
    dict_transforms,
)
from embedding_studio.embeddings.data.utils.fields_normalizer import (
    DatasetFieldsNormalizer,
)


class DictItemsDatasetDictPreprocessor(ItemsDatasetDictPreprocessor):
    TEXTS_FEATURE_NAME = "text"

    def __init__(
        self,
        field_normalizer: DatasetFieldsNormalizer,
        transform: Optional[Callable] = None,
    ):
        """Preprocessor for dict data storages.

        :param field_normalizer: how to normalize field names
        :param transform: function to get dict line out of dict object (default: None)
        """
        self._field_normalizer = field_normalizer
        self._transform = transform if transform else get_text_line_from_dict

    def get_id_field_name(self) -> str:
        return self._field_normalizer.id_field_name

    def convert(self, dataset: DatasetDict) -> DatasetDict:
        """Normalize fields, apply dict transforms and create text items storages.

        :param dataset: dataset dict to be preprocessed
        :return: train/test DatasetDict with ItemsStorage as values
        """
        dataset: DatasetDict = self._field_normalizer(dataset)

        storages: Dict[ItemsStorage] = {}
        # TODO: use more optimal way to iterate over DatasetDict
        for key in dataset.keys():
            storages[key] = ItemsStorage(
                dataset[key],
                DictItemsDatasetDictPreprocessor.TEXTS_FEATURE_NAME,
                self._field_normalizer.id_field_name,
            )

        storages: DatasetDict = DatasetDict(storages)
        storages = storages.with_transform(
            lambda examples: dict_transforms(
                examples,
                transform=self._transform,
                text_values_name=DictItemsDatasetDictPreprocessor.TEXTS_FEATURE_NAME,
            )
        )

        return storages
