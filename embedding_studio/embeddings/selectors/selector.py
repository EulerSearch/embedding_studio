from abc import ABC, abstractmethod
from typing import List

import torch


class AbstractSelector(ABC):
    @abstractmethod
    def select(
        self, query_vector: torch.Tensor, item_vectors: torch.Tensor
    ) -> List[int]:
        pass
