import logging
from typing import Callable, Optional, Set, Tuple, Union

from datasets import Dataset, DatasetDict

from embedding_studio.embeddings.augmentations.items_storage_augmentation_applier import (
    ItemsStorageAugmentationApplier,
)
from embedding_studio.embeddings.data.clickstream.paired_session import (
    PairedClickstreamDataset,
)
from embedding_studio.embeddings.data.preprocessors.preprocessor import (
    ItemsDatasetDictPreprocessor,
)
from embedding_studio.embeddings.data.storages.storage import ItemsStorage
from embedding_studio.embeddings.splitters.dataset_splitter import (
    ItemsStorageSplitter,
)

logger = logging.getLogger(__name__)


class ItemStorageProducer:
    def __init__(
        self,
        preprocessor: ItemsDatasetDictPreprocessor,
        id_field_name: Optional[str] = None,
        items_storage_splitter: Optional[ItemsStorageSplitter] = None,
        augmenter: Optional[
            Union[
                ItemsStorageAugmentationApplier,
                Callable[[ItemsStorage], ItemsStorage],
            ]
        ] = None,
        do_augment_test: bool = False,
        do_augmentation_before_preprocess: bool = True,
    ):
        """Preprocess and split dataset with train/test clickstream inputs.

        :param preprocessor: items dataset dict preprocessing
        :param id_field_name: specified field name ID (default: None)
        :param items_storage_splitter: class to split the items in subparts(default: None)
        :param augmenter: function that add additional augmented rows to an item storage (default: None)
        :param do_augment_test: do test split augmentation (default: False)
        :param do_augmentation_before_preprocess: do augmentation process before preprocess (default: True)
        """
        self.preprocessor = preprocessor
        self._id_field_name = (
            id_field_name
            if id_field_name is not None
            else preprocessor.get_id_field_name()
        )
        self.splitter = items_storage_splitter
        self._augmenter = augmenter
        self.do_augment_test = do_augment_test
        self.do_augmentation_before_preprocess = (
            do_augmentation_before_preprocess
        )

    def _augment_items_storage(self, storage: ItemsStorage) -> ItemsStorage:
        if self._augmenter is None:
            return storage

        return self._augmenter.apply_augmentation(storage)

    def _augment_test_items_storage(
        self, storage: ItemsStorage, before_preprocess: bool = True
    ) -> ItemsStorage:
        if before_preprocess == self.do_augmentation_before_preprocess:
            if self.do_augment_test:
                return self._augment_items_storage(storage)

        return storage

    def _augment_train_items_storage(
        self, storage: ItemsStorage, before_preprocess: bool = True
    ) -> ItemsStorage:
        if before_preprocess == self.do_augmentation_before_preprocess:
            return self._augment_items_storage(storage)

        return storage

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
    ) -> Tuple[DatasetDict, DatasetDict]:
        """Split dataset with train_clickstream / test_clickstream

        :param dataset: dataset to be split
        :param clickstream_dataset: train /test clickstream inputs (PairedClickstreamDataset)
        :return: split dataset and postprocessed clickstream_dataset
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

            train_dataset = dataset.filter(
                function=lambda example: example[self.id_field_name]
                in train_ids
            )
            test_dataset = dataset.filter(
                function=lambda example: example[self.id_field_name]
                in test_ids
            )

            split_dataset: DatasetDict = DatasetDict(
                {
                    "train": self._augment_train_items_storage(
                        train_dataset, True
                    ),
                    "test": self._augment_test_items_storage(
                        test_dataset, True
                    ),
                }
            )

        else:
            logger.warning(f"Provided dataset is already split")
            split_dataset: DatasetDict = dataset

        processed: DatasetDict = self._preprocess(split_dataset)
        if not self.do_augmentation_before_preprocess:
            processed: DatasetDict = DatasetDict(
                {
                    "train": self._augment_train_items_storage(
                        processed["train"], False
                    ),
                    "test": self._augment_test_items_storage(
                        processed["test"], False
                    ),
                }
            )

        if self.splitter:
            processed, clickstream_dataset = self.splitter(
                processed, clickstream_dataset
            )

        return processed, clickstream_dataset
