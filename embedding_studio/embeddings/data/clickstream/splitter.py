import random
from typing import List, Optional, Set

from datasets import DatasetDict
from sklearn.model_selection import train_test_split

from embedding_studio.embeddings.data.clickstream.paired_session import (
    PairedClickstreamDataset,
)
from embedding_studio.embeddings.data.clickstream.raw_session import (
    ClickstreamSession,
)


class ClickstreamSessionsSplitter:
    def __init__(
        self,
        test_size_ratio: float = 0.2,
        shuffle: bool = True,
        random_state: Optional[int] = None,
    ):
        """Generate train / test clickstream sessions split.

        :param test_size_ratio: ratio of test split size (default: 0.2)
        :type test_size_ratio: float
        :param shuffle: to shuffle or not paired clickstream sessions (default: True)
        :type shuffle:  bool
        :param random_state: random state to sklearn splitter (default: None)
        :type random_state: Optional[int]
        """
        self.test_size_ratio = test_size_ratio
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, sessions: List[ClickstreamSession]) -> DatasetDict:
        """Split clickstream sessions.

        :param sessions: sessions to be split
        :type sessions: List[ClickstreamSession]
        :return: train / test splits accordingly (PairedClickstreamDataset)
        :rtype: DatasetDict
        """
        # Get all IDs
        all_result_ids: Set[str] = set()
        for session in sessions:
            all_result_ids.update(session.results)

        # Ensure a minimum number of unique result IDs in each set
        min_unique_test_sessions: int = int(
            self.test_size_ratio * len(sessions)
        )

        # Split the result IDs into train and test sets
        train_result_ids, test_result_ids = train_test_split(
            list(all_result_ids),
            test_size=self.test_size_ratio,
            random_state=self.random_state,
        )
        test_result_ids: Set[str] = set(test_result_ids)

        # Split sessions into train and test based on result IDs
        train_sessions: List[ClickstreamSession] = []
        test_sessions: List[ClickstreamSession] = []

        for session in sessions:
            if len(session.results) == 0:
                continue

            if (
                len(set(session.results) & test_result_ids)
                / len(session.results)
                <= 0.5
            ):
                # If less than 50% of result IDs intersect with the test set, add to the train set
                train_sessions.append(session)
            else:
                test_sessions.append(session)

        if len(test_sessions) < min_unique_test_sessions:
            random_train_session_indexess: List[int] = random.choices(
                list(range(len(train_sessions))),
                k=min_unique_test_sessions - len(test_sessions),
            )
            for i in reversed(sorted(random_train_session_indexess)):
                test_sessions.append(train_sessions.pop(i))

        return DatasetDict(
            {
                "train": PairedClickstreamDataset(
                    train_sessions, self.shuffle
                ),
                "test": PairedClickstreamDataset(test_sessions, self.shuffle),
            }
        )
