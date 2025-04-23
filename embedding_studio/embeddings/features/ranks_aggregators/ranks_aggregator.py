from abc import ABC, abstractmethod
from typing import List, Union

import torch


class RanksAggregator(ABC):
    """Interface for a method of ranks aggregation in the situation if an item is split into subitems.

    This abstract base class defines the interface for aggregating ranks of subitems into a single rank.
    Implementations should provide concrete aggregation methods for both non-differentiable and
    differentiable contexts.
    """

    @abstractmethod
    def _aggregate(self, ranks: Union[List[float], torch.Tensor]) -> float:
        """A way to aggregate ranks into a single value.

        :param ranks: list of floats or a tensor of subitems ranks
        :return: a single aggregated rank value

        Example implementation:
        ```
        def _aggregate(self, ranks: Union[List[float], torch.Tensor]) -> float:
            if len(ranks) == 0:
                return 0.0

            return sum(ranks) / len(ranks)  # Simple mean implementation
        ```
        """
        raise NotImplemented

    @abstractmethod
    def _aggregate_differentiable(self, ranks: torch.Tensor) -> torch.Tensor:
        """A way to aggregate ranks that is differentiable for use in training processes.

        :param ranks: a tensor of subitems ranks
        :return: a tensor containing the aggregated rank that supports gradient computation

        Example implementation:
        ```
        def _aggregate_differentiable(self, ranks: torch.Tensor) -> torch.Tensor:
            if len(ranks) == 0:
                return torch.tensor(0.0)

            return torch.mean(ranks, dim=-1)  # Differentiable mean implementation
        ```
        """
        raise NotImplemented

    def __call__(
        self,
        ranks: Union[List[float], torch.Tensor],
        differentiable: bool = False,
    ) -> Union[float, torch.Tensor]:
        """Aggregate subitem ranks into a single value.

        :param ranks: list of floats or a tensor of subitems ranks
        :param differentiable: should use differentiable version of aggregation function in case of training
        :return: aggregated rank - either a scalar value or a tensor that supports gradient computation
        """
        if isinstance(ranks, list):
            return self._aggregate(ranks)

        if differentiable:
            return self._aggregate_differentiable(ranks)

        return self._aggregate(ranks)
