import logging
import random
from typing import Callable, List, Optional, Set, Union

from datasets import DatasetDict
from sklearn.model_selection import train_test_split

from embedding_studio.embeddings.augmentations.clickstream_augmentation_applier import (
    ClickstreamQueryAugmentationApplier,
)
from embedding_studio.embeddings.data.clickstream.paired_fine_tuning_inputs import (
    PairedFineTuningInputsDataset,
)
from embedding_studio.embeddings.features.fine_tuning_input import (
    FineTuningInput,
)

logger = logging.getLogger(__name__)


class TrainTestSplitter:
    def __init__(
        self,
        test_size_ratio: float = 0.2,
        shuffle: bool = True,
        random_state: Optional[int] = None,
        augmenter: Optional[
            Union[
                ClickstreamQueryAugmentationApplier,
                Callable[[List[FineTuningInput]], List[FineTuningInput]],
            ]
        ] = None,
        do_augment_test: bool = False,
    ):
        """Generate train / test fine-tuning inputs split.

        :param test_size_ratio: ratio of test split size (default: 0.2)
        :param shuffle: to shuffle or not paired clickstream inputs (default: True)
        :param random_state: random state to sklearn items_set_splitter (default: None)
        :param augmenter: function to augment clickstream inputs by augmenting queries (default: None)
        :param do_augment_test: do test split augmentation (default: False)
        """
        if (
            not isinstance(test_size_ratio, float)
            or test_size_ratio <= 0
            or test_size_ratio >= 1.0
        ):
            raise ValueError(
                f"test_size_ration is a numeric value in range (0.0, 1.0)"
            )

        if test_size_ratio >= 0.5:
            logger.warning(
                "test_size_ration is larger than 0.5. It's unusual for ML to have test size > train size."
            )

        self._test_size_ratio = test_size_ratio

        if not isinstance(shuffle, bool):
            raise ValueError("shuffle should be boolean")
        self._shuffle = shuffle
        self._random_state = random_state
        self._augmenter = augmenter
        self._do_augment_test = do_augment_test

    def _augment_clickstream(
        self, inputs: List[FineTuningInput]
    ) -> List[FineTuningInput]:
        """Apply augmentation to the clickstream inputs if an augmenter is provided.

        :param inputs: List of fine-tuning inputs to augment
        :return: Augmented list of fine-tuning inputs
        """
        if self._augmenter is None:
            return inputs

        return self._augmenter.apply_augmentation(inputs)

    @property
    def shuffle(self) -> bool:
        """Get the shuffle setting for this splitter.

        :return: Boolean indicating whether shuffling is enabled
        """
        return self._shuffle

    def split(self, inputs: List[FineTuningInput]) -> DatasetDict:
        """Split fine-tuning inputs into train and test sets.

        Splits the inputs based on result IDs to ensure related inputs stay together.
        When inputs have overlapping result IDs, they are assigned to the set where
        the majority of their results belong.

        :param inputs: List of fine-tuning inputs to split
        :return: DatasetDict containing 'train' and 'test' PairedFineTuningInputsDataset instances
        """
        # Get all IDs
        all_result_ids: Set[str] = set()
        for input in inputs:
            all_result_ids.update(input.results)

        if len(all_result_ids) == 0:
            raise ValueError("Inputs list is empty")

        # Ensure a minimum number of unique result IDs in each set
        min_unique_test_inputs: int = int(self._test_size_ratio * len(inputs))

        # Split the result IDs into train and test sets
        train_result_ids, test_result_ids = train_test_split(
            list(all_result_ids),
            test_size=self._test_size_ratio,
            random_state=self._random_state,
        )
        test_result_ids: Set[str] = set(test_result_ids)

        # Split inputs into train and test based on result IDs
        train_inputs: List[FineTuningInput] = []
        test_inputs: List[FineTuningInput] = []

        for input in inputs:
            if len(input.results) == 0:
                continue

            if (
                len(set(input.results) & test_result_ids) / len(input.results)
                <= 0.5
            ):
                # If less than 50% of result IDs intersect with the test set, add to the train set
                train_inputs.append(input)
            else:
                test_inputs.append(input)

        if len(test_inputs) < min_unique_test_inputs:
            logger.warning(
                f"Clickstream inputs intersects highly, so they are not split well"
            )
            random_train_input_indexess: List[int] = random.choices(
                list(range(len(train_inputs))),
                k=min_unique_test_inputs - len(test_inputs),
            )
            for i in reversed(sorted(random_train_input_indexess)):
                test_inputs.append(train_inputs.pop(i))

        if len(test_inputs) + len(train_inputs) < len(inputs):
            missed_inputs_count = len(inputs) - (
                len(test_inputs) + len(train_inputs)
            )
            logger.warning(
                f"Fine-tuning inputs weren't split correctly, add {missed_inputs_count} more inputs to the train split."
            )

            for input in inputs:
                if input not in train_inputs and input not in test_inputs:
                    train_inputs.append(input)

        return DatasetDict(
            {
                "train": PairedFineTuningInputsDataset(
                    self._augment_clickstream(train_inputs), self.shuffle
                ),
                "test": PairedFineTuningInputsDataset(
                    self._augment_clickstream(test_inputs),
                    self.shuffle,
                ),
            }
        )
