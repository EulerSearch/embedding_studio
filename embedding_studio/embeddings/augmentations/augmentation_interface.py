from abc import ABC, abstractmethod
from typing import Any, List


class AugmentationInterface(ABC):
    """Abstract single augmentation method."""

    @abstractmethod
    def transform(self, object: Any) -> List[Any]:
        raise NotImplementedError
