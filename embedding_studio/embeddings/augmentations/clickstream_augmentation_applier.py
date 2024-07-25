from copy import deepcopy
from typing import List

from embedding_studio.embeddings.augmentations.augmentation_with_random_selection import (
    AugmentationWithRandomSelection,
)
from embedding_studio.embeddings.features.fine_tuning_input import (
    FineTuningInput,
)

# TODO:
# 1. Refactor the Clickstream Model:
#    - Create a model where queries are stored in an array.
#    - Ensure each query in the array has a corresponding array of events and other parameters.
#
# 2. Update the ClickstreamAugmenter Class:
#    - Modify the augmentation process to only duplicate queries.
#    - Ensure the new model structure is used, associating each query with its corresponding events and parameters.
#
# 3. Optimize Memory Usage:
#    - Ensure that the new model and augmentation process are optimized for memory efficiency.
#
# 4. Update Documentation:
#    - Document the new model structure and the updated ClickstreamAugmenter class.
#    - Provide examples demonstrating the new approach.


class ClickstreamQueryAugmentationApplier:
    """Class to apply an augmentation to FineTuningInput objects and generate more augmented inputs.

    The ClickstreamQueryAugmentationApplier class is designed to augment clickstream inputs by applying a specified
    augmentation strategy to the queries within the inputs. It generates additional augmented inputs
    and returns a combined list of original and augmented inputs.

    :param augmentation: An instance of an AugmentationWithRandomSelection class, which defines the transformation to be
                         applied to the queries within the clickstream inputs.
    """

    def __init__(self, augmentation: AugmentationWithRandomSelection):
        self.augmentation = augmentation

    def apply_augmentation(
        self, inputs: List[FineTuningInput]
    ) -> List[FineTuningInput]:
        """Apply the augmentation and return a list of both original and augmented clickstream inputs.

        :param inputs: A list of FineTuningInput objects containing the original inputs.
        :return: A list of FineTuningInput objects containing both the original and the augmented inputs.
        """
        augmented_inputs = []
        for input in inputs:
            for new_query in self.augmentation.transform(input.query):
                augmented_input = deepcopy(input)
                augmented_input.query = new_query
                augmented_inputs.append(augmented_input)

        return inputs + augmented_inputs
