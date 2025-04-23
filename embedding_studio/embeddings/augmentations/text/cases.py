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
        """Initialize the ChangeCases augmentation.

        This augmentation transforms a string by applying different case modifications
        (original, lowercase, uppercase).

        :param selection_size: Number of case variations to be selected.
            If `float`, should be between `0.0` and `1.0` and represent the proportion
            of the total case variations.
            If `int`, represents the absolute number of case variations to select.
        """
        super(ChangeCases, self).__init__(selection_size)

    def _raw_transform(self, object: str) -> List[str]:
        """Apply case transformations to the input string.

        Transforms the input string into various case formats:
        - Original case (unchanged)
        - Lowercase
        - Uppercase

        :param object: The input string to be transformed.
        :return: A list of strings with different case variations.
        """
        objects = []
        for case_func in ChangeCases._CASES:
            objects.append(case_func(object))

        return objects
