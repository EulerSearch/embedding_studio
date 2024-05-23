from typing import List, Union

import torch

from embedding_studio.embeddings.features.ranks_aggregators.ranks_aggregator import (
    RanksAggregator,
)


class MeanAggregator(RanksAggregator):
    def __init__(
        self, if_empty_value: float = 0.0, if_zeroes_value: float = 0.0
    ):
        """Aggregator, that returns a mean value of passed ranks.

        :param if_empty_value: the value being returned in the situation if ranks is an empty list (default: 0.0)
        :param if_zeroes_value: the value being returned in the situation if ranks are zeroes (default: 0.0)
        """
        self.if_empty_value = if_empty_value
        self.if_zeroes_value = if_zeroes_value

    def _aggregate(self, ranks: Union[List[float], torch.Tensor]) -> float:
        if len(ranks) == 0:
            return self.if_empty_value

        if sum(ranks) == 0:
            return self.if_zeroes_value

        return sum(ranks) / len(ranks)

    def _aggregate_differentiable(self, ranks: torch.Tensor) -> torch.Tensor:
        if len(ranks) == 0:
            return torch.Value(self.if_empty_value)

        if sum(ranks) == 0:
            return torch.Value(self.if_zeroes_value)

        return torch.mean(ranks, dim=-1)
