import random
from abc import abstractmethod
from typing import Any, List

from embedding_studio.embeddings.augmentations.augmentation_interface import (
    AugmentationInterface,
)


class AugmentationWithRandomSelection(AugmentationInterface):
    def __init__(self, selection_size: float = 1.0):
        """Abstract single augmentation method with random selection of results.

        :param selection_size: number of items to be selected from original augmented objects.
        If `float`, should be between `0.0` and `1.0` and represent the proportion of the original augmented objects.
        If `int`, represents the absolute number of sub samples.
        Note that if selection_size > of the original augmented objects, objects will be returned as-is.
        """
        if selection_size <= 0:
            raise ValueError(f"Selection size should be a positive value")
        self.selection_size = selection_size

    @abstractmethod
    def _raw_transform(self, object: Any) -> List[Any]:
        raise NotImplementedError

    def _select_augmentations(self, transformed: List[Any]) -> List[Any]:
        if self.selection_size == 1 or self.selection_size >= len(transformed):
            return transformed

        if self.selection_size < 1:
            return random.choices(
                transformed, k=int(len(transformed) * self.selection_size)
            )

        # If user set exact number of augmentations
        return random.choices(transformed, k=int(self.selection_size))

    def transform(self, object: Any) -> List[Any]:
        return self._select_augmentations(self._raw_transform(object))
