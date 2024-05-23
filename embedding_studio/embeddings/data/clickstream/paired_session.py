import logging
import random
from itertools import cycle, islice
from typing import Any, Dict, List, Set, Tuple, Union

from torch.utils.data import Dataset

from embedding_studio.clickstream_storage.raw_session import FineTuningInput

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


class PairedClickstreamDataset(Dataset):
    def __init__(
        self,
        sessions: List[FineTuningInput],
        randomize: bool = False,
        session_count: int = -1,
    ):
        """Irrelevant clickstream inputs are quite unuseful, combine them with
        usual inputs to extract useful information for the future.

        :param sessions: clickstream inputs to group
        :param randomize: shuffle inputs or not (default: False)
        :param session_count: maximum session pairs to use (default: -1)
        """
        self.randomize = randomize
        self.session_count = session_count

        self.irrelevant: List[FineTuningInput] = []
        self.not_irrelevant: List[FineTuningInput] = []
        for session in sessions:
            if session.is_irrelevant:
                self.irrelevant.append(session)
            else:
                self.not_irrelevant.append(session)

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

            if session_count > 0:
                self.irrelevant_indexes = self.irrelevant_indexes[
                    :session_count
                ]
                self.not_irrelevant_indexes = self.not_irrelevant_indexes[
                    :session_count
                ]

        elif len(self.irrelevant) == 0:
            logger.warning("List of irrelevant inputs is empty")

        else:
            raise ValueError("List of not irrelevant inputs is empty")

        self.irrelevant_ids: Set[str] = set()
        for session in self.irrelevant:
            self.irrelevant_ids.update(session.results)

        self.not_irrelevant_ids: Set[str] = set()
        for session in self.not_irrelevant:
            self.not_irrelevant_ids.update(session.results)

    def __len__(self) -> int:
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
        if len(self.irrelevant) == 0:
            return self.not_irrelevant[self.not_irrelevant_indexes[idx]], None

        elif len(self.not_irrelevant) == 0:
            return None, self.irrelevant[self.irrelevant_indexes[idx]]

        return (
            self.not_irrelevant[self.not_irrelevant_indexes[idx]],
            self.irrelevant[self.irrelevant_indexes[idx]],
        )
