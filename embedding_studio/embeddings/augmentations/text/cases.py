from typing import List

from embedding_studio.embeddings.augmentations.augmentation_with_random_selection import (
    AugmentationWithRandomSelection,
)


class ChangeCases(AugmentationWithRandomSelection):
    _CASES = [
        lambda v: v,
        lambda v: v.lower(),
        lambda v: v.upper(),
    ]

    def __init__(self, selection_size: float = 1.0):
        super(ChangeCases, self).__init__(selection_size)

    def _raw_transform(self, object: str) -> List[str]:
        objects = []
        for case_func in ChangeCases._CASES:
            objects.append(case_func(object))

        return objects
