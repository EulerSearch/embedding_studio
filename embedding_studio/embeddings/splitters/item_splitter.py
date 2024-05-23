from abc import ABC, abstractmethod
from typing import Any, List


class ItemSplitter(ABC):
    """Interface for a method of splitting an item into subitems."""

    @abstractmethod
    def __call__(self, item: Any) -> List[Any]:
        raise NotImplementedError
