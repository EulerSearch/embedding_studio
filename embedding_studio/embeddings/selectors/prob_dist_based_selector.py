import torch

from embedding_studio.embeddings.selectors.dist_based_selector import (
    DistBasedSelector,
)
from embedding_studio.models.embeddings.models import SearchIndexInfo


class ProbsDistBasedSelector(DistBasedSelector):
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
        return (
            torch.sigmoid(corrected_values * self._scale)
            > self._prob_threshold
        )
