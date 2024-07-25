import logging
from typing import Callable, Optional, Set, Tuple, Union

from datasets import Dataset, DatasetDict

from embedding_studio.embeddings.augmentations.items_set_augmentation_applier import (
    ItemsSetAugmentationApplier,
)
from embedding_studio.embeddings.data.clickstream.paired_fine_tuning_inputs import (
    PairedFineTuningInputsDataset,
)
from embedding_studio.embeddings.data.items.items_set import ItemsSet
from embedding_studio.embeddings.data.preprocessors.preprocessor import (
    ItemsDatasetDictPreprocessor,
)
from embedding_studio.embeddings.splitters.dataset_splitter import (
    ItemsSetSplitter,
)

logger = logging.getLogger(__name__)


class ItemSetManager:
    def __init__(
        self,
        preprocessor: ItemsDatasetDictPreprocessor,
        id_field_name: Optional[str] = None,
        items_set_splitter: Optional[ItemsSetSplitter] = None,
        augmenter: Optional[
            Union[
                ItemsSetAugmentationApplier,
                Callable[[ItemsSet], ItemsSet],
            ]
        ] = None,
        do_augment_test: bool = False,
        do_augmentation_before_preprocess: bool = True,
    ):
        """Initialize ItemSetManager with preprocessing, splitting, and augmentation configurations.

        :param preprocessor: items dataset dict preprocessing
        :param id_field_name: specified field name ID (default: None)
        :param items_set_splitter: class to split the items in subparts (default: None)
        :param augmenter: function that adds additional augmented rows to an item items_set (default: None)
        :param do_augment_test: do test split augmentation (default: False)
        :param do_augmentation_before_preprocess: do augmentation process before preprocess (default: True)
        """
        self.preprocessor = preprocessor
        self._id_field_name = id_field_name or preprocessor.get_id_field_name()
        self.splitter = items_set_splitter
        self._augmenter = augmenter
        self.do_augment_test = do_augment_test
        self.do_augmentation_before_preprocess = (
            do_augmentation_before_preprocess
        )

    def _augment_items_set(self, items_set: ItemsSet) -> ItemsSet:
        """Apply augmentation to items set if an augmenter is defined.

        :param items_set: Items set to be augmented
        :return: Augmented items set
        """
        if self._augmenter is None:
            return items_set

        return self._augmenter.apply_augmentation(items_set)

    def _augment_test_items_set(
        self, items_set: ItemsSet, before_preprocess: bool = True
    ) -> ItemsSet:
        """Apply augmentation to test items set if configured.

        :param items_set: Items set to be augmented
        :param before_preprocess: Whether to augment before preprocessing
        :return: Augmented test items set
        """
        if (
            before_preprocess == self.do_augmentation_before_preprocess
            and self.do_augment_test
        ):
            return self._augment_items_set(items_set)

        return items_set

    def _augment_train_items_set(
        self, items_set: ItemsSet, before_preprocess: bool = True
    ) -> ItemsSet:
        """Apply augmentation to train items set if configured.

        :param items_set: Items set to be augmented
        :param before_preprocess: Whether to augment before preprocessing
        :return: Augmented train items set
        """
        if before_preprocess == self.do_augmentation_before_preprocess:
            return self._augment_items_set(items_set)

        return items_set

    @property
    def id_field_name(self) -> str:
        """Get the ID field name.

        :return: ID field name
        """
        return self._id_field_name

    def _preprocess(self, dataset: DatasetDict) -> DatasetDict:
        """Preprocess the dataset.

        :param dataset: Dataset to preprocess
        :return: Preprocessed dataset
        """
        logger.debug("Preprocess the dataset")
        return self.preprocessor.convert(dataset)

    def _split_dataset(
        self, dataset: Dataset, train_ids: Set[str], test_ids: Set[str]
    ) -> DatasetDict:
        """Split the dataset into train and test sets based on IDs.

        :param dataset: Dataset to split
        :param train_ids: Set of train IDs
        :param test_ids: Set of test IDs
        :return: Split dataset with train and test sets
        """
        logger.info("Splitting the dataset into train and test sets")
        train_dataset = dataset.filter(
            lambda example: example[self.id_field_name] in train_ids
        )
        test_dataset = dataset.filter(
            lambda example: example[self.id_field_name] in test_ids
        )

        return DatasetDict(
            {
                "train": self._augment_train_items_set(train_dataset, True),
                "test": self._augment_test_items_set(test_dataset, True),
            }
        )

    def _check_clickstream_dataset(
        self, clickstream_dataset: DatasetDict
    ) -> None:
        """Check if the clickstream dataset is valid.

        :param clickstream_dataset: Clickstream dataset to check
        :raises ValueError: If clickstream dataset is invalid
        """
        if not (
            isinstance(
                clickstream_dataset["train"], PairedFineTuningInputsDataset
            )
            and isinstance(
                clickstream_dataset["test"], PairedFineTuningInputsDataset
            )
        ):
            raise ValueError(
                "clickstream_dataset values should be instances of PairedClickstreamDataset"
            )

    def __call__(
        self,
        dataset: Union[Dataset, DatasetDict],
        clickstream_dataset: DatasetDict,
    ) -> Tuple[DatasetDict, DatasetDict]:
        """Split dataset with train_clickstream / test_clickstream.

        :param dataset: Dataset to be split
        :param clickstream_dataset: Train/test clickstream inputs (PairedClickstreamDataset)
        :return: Split dataset and postprocessed clickstream_dataset
        """
        self._check_clickstream_dataset(clickstream_dataset)

        if isinstance(dataset, Dataset):
            train_ids: Set[str] = clickstream_dataset[
                "train"
            ].irrelevant_ids.union(
                clickstream_dataset["train"].not_irrelevant_ids
            )

            if not train_ids:
                raise ValueError("Train clickstream is empty")

            test_ids: Set[str] = clickstream_dataset[
                "test"
            ].irrelevant_ids.union(
                clickstream_dataset["test"].not_irrelevant_ids
            )

            if not test_ids:
                raise ValueError("Test clickstream is empty")

            split_dataset = self._split_dataset(dataset, train_ids, test_ids)
        else:
            logger.warning("Provided dataset is already split")
            split_dataset = dataset

        processed = self._preprocess(split_dataset)

        if not self.do_augmentation_before_preprocess:
            processed = DatasetDict(
                {
                    "train": self._augment_train_items_set(
                        processed["train"], False
                    ),
                    "test": self._augment_test_items_set(
                        processed["test"], False
                    ),
                }
            )

        if self.splitter:
            processed, clickstream_dataset = self.splitter(
                processed, clickstream_dataset
            )

        return processed, clickstream_dataset
