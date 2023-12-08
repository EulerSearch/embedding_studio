import logging
from typing import Optional, Set, Union

from datasets import Dataset, DatasetDict

from embedding_studio.embeddings.data.clickstream.paired_session import (
    PairedClickstreamDataset,
)
from embedding_studio.embeddings.data.preprocessors.preprocessor import (
    ItemsDatasetDictPreprocessor,
)

logger = logging.getLogger(__name__)


class ItemStorageProducer:
    def __init__(
        self,
        preprocessor: ItemsDatasetDictPreprocessor,
        id_field_name: Optional[str] = None,
    ):
        """Preprocess and split dataset with train/test clickstream sessions.

        :param preprocessor: items dataset dict preprocessing
        :type preprocessor: ItemsDatasetDictPreprocessor
        :param id_field_name: specified field name ID (default: None)
        :type id_field_name: Optional[str]
        """
        self.preprocessor = preprocessor
        self._id_field_name = (
            id_field_name
            if id_field_name is not None
            else preprocessor.get_id_field_name()
        )

    @property
    def id_field_name(self) -> str:
        return self._id_field_name

    def _preprocess(self, dataset: DatasetDict) -> DatasetDict:
        logger.debug("Prerprocess a dataset")
        return self.preprocessor.convert(dataset)

    def __call__(
        self,
        dataset: Union[Dataset, DatasetDict],
        clickstream_dataset: DatasetDict,
    ) -> DatasetDict:
        """Split dataset with train_clickstream / test_clickstream

        :param dataset: dataset to be split
        :type dataset: Union[Dataset, DatasetDict]
        :param clickstream_dataset: train /test clickstream sessions (PairedClickstreamDataset)
        :type clickstream_dataset: DatasetDict
        :return: split dataset
        :rtype: DatasetDict
        """

        if not (
            isinstance(clickstream_dataset["train"], PairedClickstreamDataset)
            and isinstance(
                clickstream_dataset["test"], PairedClickstreamDataset
            )
        ):
            raise ValueError(
                "clickstream_dataset values should be instances of PairedClickstreamDataset"
            )

        if isinstance(dataset, Dataset):
            train_ids: Set[str] = clickstream_dataset[
                "train"
            ].irrelevant_ids.union(
                clickstream_dataset["train"].not_irrelevant_ids
            )

            if len(train_ids) == 0:
                raise ValueError("Train clickstream is empty")

            test_ids: Set[str] = clickstream_dataset[
                "test"
            ].irrelevant_ids.union(
                clickstream_dataset["test"].not_irrelevant_ids
            )

            if len(test_ids) == 0:
                raise ValueError("Train clickstream is empty")

            split_dataset: DatasetDict = DatasetDict(
                {
                    "train": dataset.filter(
                        lambda example: example[self.id_field_name]
                        in train_ids
                    ),
                    "test": dataset.filter(
                        lambda example: example[self.id_field_name] in test_ids
                    ),
                }
            )

        else:
            logger.warning(f"Provided dataset is already split")
            split_dataset: DatasetDict = dataset

        return self._preprocess(split_dataset)
