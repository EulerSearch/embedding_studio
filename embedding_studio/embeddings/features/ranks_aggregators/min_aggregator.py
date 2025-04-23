from typing import List, Union

import torch

from embedding_studio.embeddings.features.ranks_aggregators.ranks_aggregator import (
    RanksAggregator,
)
from embedding_studio.embeddings.models.utils.differentiable_extreme import (
    differentiable_extreme,
)


class MinAggregator(RanksAggregator):
    def __init__(self, if_empty_value: float = 0.0):
        """Aggregator that returns the minimum value from a list of ranks.

        This aggregator selects the lowest rank value from the provided ranks. If the ranks
        list is empty, it returns a configurable default value.

        :param if_empty_value: the value to return when ranks is an empty list
        """
        self.if_empty_value = if_empty_value

    def _aggregate(self, ranks: Union[List[float], torch.Tensor]) -> float:
        """Calculates the minimum value from the provided ranks.

        :param ranks: list of floats or a tensor of subitems ranks
        :return: the minimum value from ranks, or if_empty_value if ranks is empty
        """
        if len(ranks) == 0:
            return self.if_empty_value

        return min(ranks)

    def _aggregate_differentiable(self, ranks: torch.Tensor) -> torch.Tensor:
        """Calculates the differentiable minimum value from the provided ranks tensor.

        Uses differentiable_extreme to ensure gradient flow during training.

        :param ranks: a tensor of subitems ranks
        :return: a tensor containing the minimum value that supports gradient computation
        """
        if len(ranks) == 0:
            return torch.Value(self.if_empty_value)

        return differentiable_extreme(ranks, "min")
