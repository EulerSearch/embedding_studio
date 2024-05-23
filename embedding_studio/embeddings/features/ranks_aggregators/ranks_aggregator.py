from abc import ABC, abstractmethod
from typing import List, Union

import torch


class RanksAggregator(ABC):
    """Interface for a method of ranks aggregation in the situation if an item is split into subitems."""

    @abstractmethod
    def _aggregate(self, ranks: Union[List[float], torch.Tensor]) -> float:
        """A way to aggregate.

        :param ranks: list of floats or a tensor of subitems ranks
        :return:
            a solid rank
        """
        raise NotImplemented

    @abstractmethod
    def _aggregate_differentiable(self, ranks: torch.Tensor) -> torch.Tensor:
        """A way to aggregate, that should be differentiable in case of using in training process.

        :param ranks: a tensor of subitems ranks
        :return:
            A tensor that can be used for making a gradient step.
        """
        raise NotImplemented

    def __call__(
        self,
        ranks: Union[List[float], torch.Tensor],
        differentiable: bool = False,
    ) -> Union[float, torch.Tensor]:
        """Aggregate subitem ranks.

        :param ranks: list of floats or a tensor of subitems ranks
        :param differentiable: should use differentiable version of aggregation function in case of training (default: False)
        :return:
            Aggregated rank - a solid value or a tensor that can be used for making a gradient step.
        """
        if isinstance(ranks, list):
            return self._aggregate(ranks)

        if differentiable:
            return self._aggregate_differentiable(ranks)

        return self._aggregate(ranks)
