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
        :param id_field_name: name of ID column
        """
        if not id_field_name:
            raise ValueError("id_field_name should be non-empty string")
        self.id_field_name = id_field_name

        if not item_field_name:
            raise ValueError("item_field_name should be non-empty string")
        self.item_field_name = item_field_name

    def __call__(self, dataset: DatasetDict) -> DatasetDict:
        """Normalize field names in a dataset and convert IDs to string format.

        This method renames the specified item and ID fields to standardized names
        and ensures IDs are converted to string format for consistency. If the
        standardized field names already exist, a warning is logged.

        :param dataset: DatasetDict to normalize
        :return: DatasetDict with normalized field names and ID values as strings
        """
        # Define a helper function to normalize ID values
        # Converts PyTorch tensor values to scalar strings using .item()
        # Any other value types are directly converted to strings
        id_normalizer = lambda id_value: (
            str(
                id_value.item()
            )  # Extract scalar value from tensor and convert to string
            if (
                isinstance(id_value, Tensor)
                or isinstance(id_value, FloatTensor)
            )
            else str(id_value)  # Convert non-tensor values directly to string
        )

        # Process each split in the dataset (e.g., train, test, validation)
        for key in dataset.keys():
            # Check if the standardized ID field name already exists in this split
            if (
                DatasetFieldsNormalizer.ID_FIELD_NAME
                not in dataset.column_names[key]
            ):
                # Rename the user-specified ID field to the standardized name
                dataset = dataset.rename_column(
                    self.id_field_name, DatasetFieldsNormalizer.ID_FIELD_NAME
                )
            else:
                # Log a warning if the standardized ID field already exists
                logger.warning(
                    f"Dataset {key} split already has {DatasetFieldsNormalizer.ID_FIELD_NAME} field"
                )

            # Perform the same check and renaming for the item field
            if (
                DatasetFieldsNormalizer.ITEM_FIELD_NAME
                not in dataset.column_names[key]
            ):
                # Rename the user-specified item field to the standardized name
                dataset = dataset.rename_column(
                    self.item_field_name,
                    DatasetFieldsNormalizer.ITEM_FIELD_NAME,
                )
            else:
                # Log a warning if the standardized item field already exists
                logger.warning(
                    f"Dataset {key} split already has {DatasetFieldsNormalizer.ITEM_FIELD_NAME} field"
                )

        # Apply the ID normalization function to each example in the dataset
        # This ensures all ID values are consistently represented as strings
        return dataset.map(
            lambda example: {
                DatasetFieldsNormalizer.ID_FIELD_NAME: id_normalizer(
                    example[DatasetFieldsNormalizer.ID_FIELD_NAME]
                )
            }
        )
