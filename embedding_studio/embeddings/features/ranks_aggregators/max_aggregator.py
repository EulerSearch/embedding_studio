from typing import List, Union

import torch

from embedding_studio.embeddings.features.ranks_aggregators.ranks_aggregator import (
    RanksAggregator,
)
from embedding_studio.embeddings.models.utils.differentiable_extreme import (
    differentiable_extreme,
)


class MaxAggregator(RanksAggregator):
    def __init__(self, if_empty_value: float = 0.0):
        """Aggregator, that returns a max value.

        :param if_empty_value: the value being returned in the situation if ranks is an empty list (default: 0.0)
        """
        self.if_empty_value = if_empty_value

    def _aggregate(self, ranks: Union[List[float], torch.Tensor]) -> float:
        if len(ranks) == 0:
            return self.if_empty_value

        return max(ranks)

    def _aggregate_differentiable(self, ranks: torch.Tensor) -> torch.Tensor:
        if len(ranks) == 0:
            return torch.Value(self.if_empty_value)

        return differentiable_extreme(ranks, "max")
