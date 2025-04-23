import torch

from embedding_studio.embeddings.selectors.dist_based_selector import (
    DistBasedSelector,
)
from embedding_studio.models.embeddings.models import SearchIndexInfo


class ProbsDistBasedSelector(DistBasedSelector):
    """
    A probability-based selector that uses sigmoid function to make selection decisions.

    This class extends DistBasedSelector by implementing a probabilistic approach
    to selection, where distances are converted to probabilities using a sigmoid function.

    :param search_index_info: Information about the search index and its configuration
    :param is_similarity: Whether the distance values represent similarity (higher is better)
                         rather than distance (lower is better)
    :param margin: Threshold margin for positive selection
    :param softmin_temperature: Temperature parameter for softmin operations
    :param scale: Scaling factor for the sigmoid function, controls decision boundary steepness
    :param prob_threshold: Probability threshold for positive selection (0.0-1.0)
    :param scale_to_one: Whether to normalize values to a [0, 1] range
    """

    def __init__(
        self,
        search_index_info: SearchIndexInfo,
        is_similarity: bool = False,
        margin: float = 0.2,
        softmin_temperature: float = 1.0,
        scale: float = 10.0,
        prob_threshold: float = 0.5,
        scale_to_one: bool = False,
    ):
        super().__init__(
            search_index_info=search_index_info,
            is_similarity=is_similarity,
            margin=margin,
            softmin_temperature=softmin_temperature,
            scale_to_one=scale_to_one,
        )

        self._scale = scale
        self._prob_threshold = prob_threshold

    def _calculate_binary_labels(
        self, corrected_values: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates binary selection labels using a sigmoid probability function.

        This method implements the abstract method from DistBasedSelector by:
        1. Converting corrected distance values to probabilities using sigmoid
        2. Applying the probability threshold to determine selection

        :param corrected_values: Tensor of distances that have been adjusted by the margin
        :return: Binary tensor indicating which items should be selected (1) or not (0)
        """
        return (
            torch.sigmoid(corrected_values * self._scale)
            > self._prob_threshold
        )
