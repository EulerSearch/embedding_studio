from typing import Optional

import torch
from torch import FloatTensor, Tensor


class SessionFeatures:
    def __init__(
        self,
        positive_ranks: Optional[FloatTensor] = None,
        negative_ranks: Optional[FloatTensor] = None,
        target: Optional[Tensor] = None,
        positive_confidences: Optional[FloatTensor] = None,
        negative_confidences: Optional[FloatTensor] = None,
    ):
        """Extracted features of clickstream session using embeddings.

        :param positive_ranks: ranks of positive events
        :type positive_ranks: Optional[FloatTensor]
        :param negative_ranks: ranks of negative events
        :type negative_ranks: Optional[FloatTensor]
        :param target: if target == 1 ranks are similarities, if target == -1 ranks are distances
        :type target: Optional[FloatTensor]
        :param positive_confidences: confidences of positive events (like clicks)
        :type positive_confidences: Optional[FloatTensor]
        :param negative_confidences: confidences of not positive events
        :type negative_confidences: Optional[FloatTensor]
        """
        self.positive_ranks = positive_ranks
        self.negative_ranks = negative_ranks

        self.target = target

        self.positive_confidences = positive_confidences
        self.negative_confidences = negative_confidences

    @staticmethod
    def __accumulate(self_var: Tensor, other_var: Tensor):
        if self_var is not None and other_var is not None:
            return torch.cat([self_var, other_var])

        elif other_var is not None:
            return other_var

    def __iadd__(self, other):
        """Accumulate features from another session

        :param other: other session
        :type other: SessionFeatures
        :return: aggregates features
        :rtype: SessionFeatures
        """
        self.positive_ranks = SessionFeatures.__accumulate(
            self.positive_ranks, other.positive_ranks
        )
        self.negative_ranks = SessionFeatures.__accumulate(
            self.negative_ranks, other.negative_ranks
        )

        self.target = SessionFeatures.__accumulate(self.target, other.target)

        self.positive_confidences = SessionFeatures.__accumulate(
            self.positive_confidences, other.positive_confidences
        )
        self.negative_confidences = SessionFeatures.__accumulate(
            self.negative_confidences, other.negative_confidences
        )

        return self

    def clamp_diff_in(self, min: float, max: float):
        """Filter min < |positive_ranks - negative_ranks| < max examples.

        :param min: minimal difference between pos and neg ranks
        :type min: float
        :param max: maximal difference between pos and neg ranks
        :type max: float
        """
        if self.positive_ranks is not None and self.negative_ranks is not None:
            hard_examples: Tensor = (
                torch.abs(self.positive_ranks - self.negative_ranks) > min
            )
            smooth_examples: Tensor = (
                torch.abs(self.positive_ranks - self.negative_ranks) < max
            )

            examples: Tensor = torch.logical_and(
                hard_examples, smooth_examples
            )

            self.positive_ranks = self.positive_ranks[examples]
            self.negative_ranks = self.negative_ranks[examples]
            self.target = self.target[examples]
            self.positive_confidences = self.positive_confidences[examples]
            self.negative_confidences = self.negative_confidences[examples]

    def use_positive_from(self, other):
        """If session is fully irrelevant, to use positive pairs from another session.
        This way "triple loss" becomes "contrastive"

        :param other: not irrelevant session with positive evidences
        :type other: SessionFeatures
        """
        if self.negative_ranks.shape[0] < other.positive_ranks.shape[0]:
            positive_ranks_: FloatTensor = other.positive_ranks[
                : self.negative_ranks.shape[0]
            ]

        elif self.negative_ranks.shape[0] > other.positive_ranks.shape[0]:
            self.negative_ranks = self.negative_ranks[
                : other.positive_ranks.shape[0]
            ]
            self.target = self.target[: other.positive_ranks.shape[0]]

            self.negative_confidences = self.negative_confidences[
                : other.positive_ranks.shape[0]
            ]
            positive_ranks_: FloatTensor = other.positive_ranks
        else:
            positive_ranks_: FloatTensor = other.positive_ranks

        self.positive_confidences = other.positive_ranks
        self.positive_ranks = positive_ranks_
