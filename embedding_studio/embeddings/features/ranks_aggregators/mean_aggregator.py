from typing import List, Union

import torch

from embedding_studio.embeddings.features.ranks_aggregators.ranks_aggregator import (
    RanksAggregator,
)


class MeanAggregator(RanksAggregator):
    def __init__(
        self, if_empty_value: float = 0.0, if_zeroes_value: float = 0.0
    ):
        """Aggregator that returns the mean value of passed ranks.

        This aggregator calculates the average of all ranks. It has special handling
        for empty lists and when all ranks are zeros.

        :param if_empty_value: the value to return when ranks is an empty list
        :param if_zeroes_value: the value to return when all ranks are zeros
        """
        self.if_empty_value = if_empty_value
        self.if_zeroes_value = if_zeroes_value

    def _aggregate(self, ranks: Union[List[float], torch.Tensor]) -> float:
        """Calculates the mean value from the provided ranks.

        :param ranks: list of floats or a tensor of subitems ranks
        :return: the mean value of ranks, if_empty_value if ranks is empty,
                 or if_zeroes_value if sum of ranks is zero
        """
        if len(ranks) == 0:
            return self.if_empty_value

        if sum(ranks) == 0:
            return self.if_zeroes_value

        return sum(ranks) / len(ranks)

    def _aggregate_differentiable(self, ranks: torch.Tensor) -> torch.Tensor:
        """Calculates the differentiable mean value from the provided ranks tensor.

        :param ranks: a tensor of subitems ranks
        :return: a tensor containing the mean value that supports gradient computation
        """
        if len(ranks) == 0:
            return torch.Value(self.if_empty_value)

        if sum(ranks) == 0:
            return torch.Value(self.if_zeroes_value)

        return torch.mean(ranks, dim=-1)
