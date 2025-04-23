from abc import abstractmethod
from typing import List, Optional

import torch

from embedding_studio.embeddings.selectors.selector import AbstractSelector
from embedding_studio.models.embeddings.models import (
    MetricType,
    SearchIndexInfo,
)
from embedding_studio.models.embeddings.objects import ObjectWithDistance


class DistBasedSelector(AbstractSelector):
    """
    A selector that makes selection decisions based on distance values.

    This abstract class provides a base implementation for selectors that operate
    on pre-calculated distance metrics without requiring access to the actual vectors.

    :param search_index_info: Information about the search index and its configuration
    :param is_similarity: Whether the distance values represent similarity (higher is better)
                         rather than distance (lower is better)
    :param margin: Threshold margin for positive selection
    :param softmin_temperature: Temperature parameter for softmin operations
    :param scale_to_one: Whether to normalize values to a [0, 1] range
    """

    def __init__(
        self,
        search_index_info: SearchIndexInfo,
        is_similarity: bool = False,
        margin: float = 0.2,
        softmin_temperature: float = 1.0,
        scale_to_one: bool = False,
    ):
        self._search_index_info = search_index_info
        self._is_similarity = is_similarity
        self._margin = margin
        self._softmin_temperature = softmin_temperature
        self._scale_to_one = scale_to_one

    @property
    def vectors_are_needed(self) -> bool:
        """
        Indicates whether this selector requires access to the actual embedding vectors.

        :return: False, as distance-based selectors only use pre-calculated distances
        """
        return False

    @abstractmethod
    def _calculate_binary_labels(
        self, corrected_values: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates binary selection labels (0 or 1) from corrected distance values.

        This abstract method must be implemented by subclasses to define the specific
        decision boundary for selection based on the corrected distance values.

        :param corrected_values: Tensor of distances that have been adjusted by the margin
        :return: Binary tensor indicating which items should be selected (1) or not (0)

        Example implementation:
        ```python
        def _calculate_binary_labels(self, corrected_values: torch.Tensor) -> torch.Tensor:
            # Simple threshold-based selection
            return corrected_values > 0
        ```
        """
        raise NotImplementedError

    def _convert_values(
        self, categories: List[ObjectWithDistance]
    ) -> torch.Tensor:
        """
        Converts raw distance values from objects to a normalized tensor.

        This method extracts distance values from objects and normalizes them
        based on the metric type and similarity/distance mode.

        :param categories: List of objects with distance metrics
        :return: Tensor of normalized distance values
        """
        values = []
        for category in categories:
            value = category.distance
            if self._is_similarity:
                if self._search_index_info.metric_type == MetricType.DOT:
                    value = value * -1.0

                elif self._search_index_info.metric_type == MetricType.COSINE:
                    value = 1.0 - value

                elif self._search_index_info.metric_type == MetricType.EUCLID:
                    value = 1.0 / max(value, 1e-8)

            values.append(value)

        return torch.tensor(values)

    def select(
        self,
        categories: List[ObjectWithDistance],
        query_vector: Optional[torch.Tensor] = None,
    ) -> List[int]:
        """
        Selects indices of objects based on their distance values.

        This method implements the selection logic by:
        1. Converting raw distances to normalized values
        2. Applying the margin threshold
        3. Calculating binary selection labels
        4. Returning indices of selected objects

        :param categories: List of objects with distance metrics
        :param query_vector: Optional query vector (not used in distance-based selectors)
        :return: List of indices of selected objects
        """
        values = self._convert_values(categories)

        positive_threshold_min = (
            1 - self._margin if self._is_similarity else self._margin
        )
        corrected_values = values - positive_threshold_min
        bin_labels = self._calculate_binary_labels(corrected_values)
        return torch.nonzero(bin_labels).T[0].tolist()
