from abc import ABC, abstractmethod
from typing import Any, List


class AugmentationInterface(ABC):
    """Abstract single augmentation method."""

    @abstractmethod
    def transform(self, object: Any) -> List[Any]:
        """Transform the input object into a list of augmented objects.

        :param object: The input object to be augmented.
        :return: A list of augmented objects after transformation.

        Example implementation:
            def transform(self, object: Any) -> List[Any]:
                # Simple example of duplicating the input object
                return [object, object]
        """
        raise NotImplementedError
