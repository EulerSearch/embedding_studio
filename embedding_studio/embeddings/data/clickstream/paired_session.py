import random
from typing import Dict, List, Set, Tuple, Union

from torch.utils.data import Dataset

from embedding_studio.embeddings.data.clickstream.raw_session import (
    ClickstreamSession,
)


class PairedClickstreamDataset(Dataset):
    def __init__(
        self,
        sessions: List[ClickstreamSession],
        randomize: bool = False,
        session_count: int = -1,
    ):
        """Irrelevant clickstream sessions are quite unuseful, combine them with
        usual sessions to extract useful information for the future.

        :param sessions: clickstream sessions to group
        :type sessions: List[ClickstreamSession]
        :param randomize: shuffle sessions or not (default: False)
        :type randomize: bool
        :param session_count: maximum session pairs to use (default: -1)
        :type session_count: int
        """
        self.irrelevant: List[ClickstreamSession] = []
        self.not_irrelevant: List[ClickstreamSession] = []
        for session in sessions:
            if session.is_irrelevant:
                self.irrelevant.append(session)
            else:
                self.not_irrelevant.append(session)

        self.irrelevant_indexes: List[int] = list(range(len(self.irrelevant)))
        self.not_irrelevant_indexes: List[int] = list(
            range(len(self.not_irrelevant))
        )

        # Count of sessions should be different, so we need to aligned them
        if len(self.irrelevant) > 0 and len(self.not_irrelevant) > 0:
            relevant_multiplier: float = len(self.irrelevant) // len(
                self.not_irrelevant
            )

            if relevant_multiplier > 1:
                self.not_irrelevant_indexes: List[
                    int
                ] = self.not_irrelevant_indexes * (relevant_multiplier + 1)

            elif relevant_multiplier == 0:
                relevant_multiplier: float = len(self.not_irrelevant) // len(
                    self.irrelevant
                )
                self.irrelevant_indexes: List[
                    int
                ] = self.irrelevant_indexes * (relevant_multiplier + 1)

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

        self.irrelevant_ids: Set[str] = set()
        for session in self.irrelevant:
            self.irrelevant_ids.update(session.results)

        self.not_irrelevant_ids: Set[str] = set()
        for session in self.not_irrelevant:
            self.not_irrelevant_ids.update(session.results)

    def __len__(self) -> int:
        return min(
            len(self.irrelevant_indexes), len(self.not_irrelevant_indexes)
        )

    def __getitem__(
        self, idx
    ) -> Tuple[ClickstreamSession, Union[ClickstreamSession, Dict]]:
        if len(self.irrelevant) > 0:
            return (
                self.not_irrelevant[self.not_irrelevant_indexes[idx]],
                self.irrelevant[self.irrelevant_indexes[idx]],
            )
        else:
            return self.not_irrelevant[self.not_irrelevant_indexes[idx]], {}
