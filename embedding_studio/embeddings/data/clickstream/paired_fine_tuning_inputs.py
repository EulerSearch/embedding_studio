import logging
import random
from itertools import cycle, islice
from typing import Any, Dict, List, Set, Tuple, Union

from torch.utils.data import Dataset

from embedding_studio.embeddings.features.fine_tuning_input import (
    FineTuningInput,
)

logger = logging.getLogger(__name__)


def _make_lists_equal_size(
    list1: List[Any], list2: List[Any]
) -> Tuple[List[Any], List[Any]]:
    # Calculate the maximum size of the two lists
    max_size = max(len(list1), len(list2))

    # Use itertools.cycle to repeat elements of the smaller list
    equal_size_list1 = list(islice(cycle(list1), max_size))
    equal_size_list2 = list(islice(cycle(list2), max_size))

    return equal_size_list1, equal_size_list2


class PairedFineTuningInputsDataset(Dataset):
    def __init__(
        self,
        inputs: List[FineTuningInput],
        randomize: bool = False,
        inputs_count: int = -1,
    ):
        """ "Combines relevant and irrelevant fine-tuning inputs to create paired training data.

        This dataset pairs relevant and irrelevant inputs together to extract useful information
        for model training. It handles cases where the counts of relevant and irrelevant inputs
        differ by making them equal size through cycling.

        :param inputs: List of fine-tuning inputs to group into relevant and irrelevant sets
        :param randomize: Whether to shuffle the inputs randomly (default: False)
        :param inputs_count: Maximum number of input pairs to use, -1 for unlimited (default: -1)
        """
        self.randomize = randomize
        self.inputs_count = inputs_count

        self.irrelevant: List[FineTuningInput] = []
        self.not_irrelevant: List[FineTuningInput] = []
        for fine_tuning_input in inputs:
            if fine_tuning_input.is_irrelevant:
                self.irrelevant.append(fine_tuning_input)
            else:
                self.not_irrelevant.append(fine_tuning_input)

        self.irrelevant_indexes: List[int] = list(range(len(self.irrelevant)))
        self.not_irrelevant_indexes: List[int] = list(
            range(len(self.not_irrelevant))
        )

        # Count of inputs should be different, so we need to aligned them
        if len(self.irrelevant) > 0 and len(self.not_irrelevant) > 0:
            if len(self.irrelevant) != len(self.not_irrelevant):
                logger.debug(
                    "Lists of irrelevant and not irrelevant inputs has different sizes. Make them equal."
                )
                (
                    self.irrelevant_indexes,
                    self.not_irrelevant_indexes,
                ) = _make_lists_equal_size(
                    self.irrelevant_indexes, self.not_irrelevant_indexes
                )

            if randomize:
                random.shuffle(self.irrelevant_indexes)
                random.shuffle(self.not_irrelevant_indexes)

            if inputs_count > 0:
                self.irrelevant_indexes = self.irrelevant_indexes[
                    :inputs_count
                ]
                self.not_irrelevant_indexes = self.not_irrelevant_indexes[
                    :inputs_count
                ]

        elif len(self.irrelevant) == 0:
            logger.warning("List of irrelevant inputs is empty")

        else:
            raise ValueError("List of not irrelevant inputs is empty")

        self.irrelevant_ids: Set[str] = set()
        for fine_tuning_input in self.irrelevant:
            self.irrelevant_ids.update(fine_tuning_input.results)

        self.not_irrelevant_ids: Set[str] = set()
        for fine_tuning_input in self.not_irrelevant:
            self.not_irrelevant_ids.update(fine_tuning_input.results)

    def __len__(self) -> int:
        """Return the number of pairs in the dataset.

        :return: Number of paired inputs in the dataset
        """
        if len(self.irrelevant) == 0:
            return len(self.not_irrelevant)

        elif len(self.not_irrelevant) == 0:
            return len(self.irrelevant)

        return min(
            len(self.irrelevant_indexes), len(self.not_irrelevant_indexes)
        )

    def __getitem__(
        self, idx
    ) -> Tuple[
        Union[FineTuningInput, Dict, None],
        Union[FineTuningInput, Dict, None],
    ]:
        """Get a pair of relevant and irrelevant inputs at the specified index.

        Returns a tuple containing a relevant input (or None) and an irrelevant input (or None).
        If either category is empty, the corresponding tuple element will be None.

        :param idx: Index of the paired inputs to retrieve
        :return: Tuple of (relevant_input, irrelevant_input) where either could be None if the
                 corresponding category is empty
        """
        if len(self.irrelevant) == 0:
            return self.not_irrelevant[self.not_irrelevant_indexes[idx]], None

        elif len(self.not_irrelevant) == 0:
            return None, self.irrelevant[self.irrelevant_indexes[idx]]

        return (
            self.not_irrelevant[self.not_irrelevant_indexes[idx]],
            self.irrelevant[self.irrelevant_indexes[idx]],
        )
