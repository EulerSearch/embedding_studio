import logging

from datasets import DatasetDict
from torch import FloatTensor, Tensor

logger = logging.getLogger(__name__)


class DatasetFieldsNormalizer:
    ID_FIELD_NAME = "item_id"
    ITEM_FIELD_NAME = "item"

    def __init__(self, item_field_name: str, id_field_name: str):
        """Unify column names in DatasetDict, so it can be used in fine-tuning script.
        A dataset should have ID column, related to ID in clickstream.

        :param item_field_name: name of column with items.
        :type item_field_name: str
        :param id_field_name: name of ID column
        :type id_field_name: str
        """
        if not id_field_name:
            raise ValueError("id_field_name should be non-empty string")
        self.id_field_name = id_field_name

        if not item_field_name:
            raise ValueError("item_field_name should be non-empty string")
        self.item_field_name = item_field_name

    def __call__(self, dataset: DatasetDict) -> DatasetDict:
        id_normalizer = (
            lambda id_value: str(id_value.item())
            if (
                isinstance(id_value, Tensor)
                or isinstance(id_value, FloatTensor)
            )
            else str(id_value)
        )
        for key in dataset.keys():

            if (
                DatasetFieldsNormalizer.ID_FIELD_NAME
                not in dataset.column_names[key]
            ):
                dataset = dataset.rename_column(
                    self.id_field_name, DatasetFieldsNormalizer.ID_FIELD_NAME
                )
            else:
                logger.warning(
                    f"Dataset {key} split already has {DatasetFieldsNormalizer.ID_FIELD_NAME} field"
                )

            if (
                DatasetFieldsNormalizer.ITEM_FIELD_NAME
                not in dataset.column_names[key]
            ):
                dataset = dataset.rename_column(
                    self.item_field_name,
                    DatasetFieldsNormalizer.ITEM_FIELD_NAME,
                )
            else:
                logger.warning(
                    f"Dataset {key} split already has {DatasetFieldsNormalizer.ITEM_FIELD_NAME} field"
                )

        return dataset.map(
            lambda example: {
                DatasetFieldsNormalizer.ID_FIELD_NAME: id_normalizer(
                    example[DatasetFieldsNormalizer.ID_FIELD_NAME]
                )
            }
        )
