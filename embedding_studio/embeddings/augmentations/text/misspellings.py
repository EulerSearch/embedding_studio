from typing import Any, List, Optional

from embedding_studio.embeddings.augmentations.augmentation_with_random_selection import (
    AugmentationWithRandomSelection,
)
from embedding_studio.utils.misspelling.misspellers import (
    introduce_misspellings_with_keyboard_map,
)


class Misspellings(AugmentationWithRandomSelection):
    """Class to introduce misspellings into a string based on specified error rates.

    The Misspellings class applies misspellings to input strings using defined error rates. This can
    be useful for data augmentation in tasks such as text classification or spell-checking systems.

    :param selection_size: A float indicating the proportion of misspellings to apply.
    :param error_rates: An optional list of float values representing different error rates to introduce
                        misspellings. If not provided, default error rates of 0.1 and 0.2 will be used.
    """

    def __init__(
        self,
        selection_size: float = 1.0,
        error_rates: Optional[List[float]] = None,
    ):
        super(Misspellings, self).__init__(selection_size)
        self.error_rates = error_rates if error_rates else [0.1, 0.2]

    def _raw_transform(self, object: Any) -> List[Any]:
        """Apply misspellings to the input string based on the defined error rates.

        :param object: The input string to be transformed with misspellings.
        :return: A list of strings with introduced misspellings at different error rates.
        """
        objects = [object]
        for error_rate in self.error_rates:
            objects.append(
                introduce_misspellings_with_keyboard_map(object, error_rate)
            )

        return objects
