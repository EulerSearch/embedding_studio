import logging
from typing import Any, List

from embedding_studio.embeddings.augmentations.augmentation_with_random_selection import (
    AugmentationWithRandomSelection,
)

logger = logging.getLogger(__name__)


class AugmentationsComposition(AugmentationWithRandomSelection):
    """Class to compose multiple augmentations and apply them sequentially to an object.

    The AugmentationsComposition class allows combining multiple augmentation strategies. Each augmentation
    is applied in sequence to the input object, generating a list of transformed objects.

    :param augmentations: A list of AugmentationWithRandomSelection instances to be applied sequentially.
    :param selection_size: A float indicating the proportion of augmentations to select and apply.
    """

    def __init__(
        self,
        augmentations: List[AugmentationWithRandomSelection],
        selection_size: float = 1.0,
    ):
        super(AugmentationsComposition, self).__init__(selection_size)
        self.augmentations = augmentations

    def _raw_transform(self, object: Any) -> List[Any]:
        """Apply the composed augmentations sequentially to the input object.

        :param object: The input object to be augmented.
        :return: A list of augmented objects after applying all the augmentations.
        """
        if isinstance(object, list):
            logger.warning(
                f"The object you've passed is a list, please double check the input. If it is expected input type - everything is ok."
            )
        objects = [
            object,
        ]
        for augmentation in self.augmentations:
            next_iteration = []
            for object_ in objects:
                next_iteration += augmentation.transform(object_)
            objects = next_iteration

        return objects
