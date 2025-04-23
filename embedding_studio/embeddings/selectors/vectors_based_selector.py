from typing import List, Optional

import torch
import torch.nn.functional as F

from embedding_studio.embeddings.selectors.selector import AbstractSelector
from embedding_studio.models.embeddings.models import (
    MetricAggregationType,
    MetricType,
    SearchIndexInfo,
)
from embedding_studio.models.embeddings.objects import ObjectWithDistance


class VectorsBasedSelector(AbstractSelector):
    """
    A selector that makes selection decisions based on vector comparisons.

    This class extends AbstractSelector to implement selection logic that requires
    access to the actual embedding vectors, not just pre-calculated distances.

    :param search_index_info: Information about the search index and its configuration
    :param is_similarity: Whether higher values represent similarity rather than distance
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

        :return: True, as this selector operates directly on embedding vectors
        """
        return True

    def _calculate_distance(
        self,
        query_vector: torch.Tensor,  # Shape: [N1, D]
        item_vectors: torch.Tensor,  # Shape: [N2, M, D]
        softmin_temperature: float = 1.0,  # Temperature for soft minimum
        is_similarity: bool = False,
    ) -> torch.Tensor:
        """
        Compute similarity or distance between queries and items.

        This method calculates distances or similarities between query vectors
        and item vectors using the configured metric type and aggregation method.

        :param query_vector: Tensor of shape [N1, D] representing query embeddings
        :param item_vectors: Tensor of shape [N2, M, D] representing item embeddings
        :param softmin_temperature: Temperature for differentiable softmin approximation
        :param is_similarity: Whether to treat values as similarity or distance
        :return: Tensor of shape [N1, N2] containing distances/similarities between queries and items
        """
        # Calculate initial similarities/distances [N1, N2, M]
        if self._search_index_info.metric_type == MetricType.COSINE:
            queries_norm = (
                F.normalize(query_vector, p=2, dim=-1)
                .unsqueeze(1)
                .unsqueeze(2)
            )  # [N1, 1, 1, D]
            items_norm = F.normalize(item_vectors, p=2, dim=-1)  # [N2, M, D]
            items_norm = items_norm.unsqueeze(0)  # [1, N2, M, D]
            values = torch.sum(
                queries_norm * items_norm, dim=-1
            )  # [N1, N2, M]
            if self._scale_to_one:
                values = (values + 1) / 2

            if not is_similarity:
                values = 1 - values

        elif self._search_index_info.metric_type == MetricType.DOT:
            queries_exp = query_vector.unsqueeze(1).unsqueeze(
                2
            )  # [N1, 1, 1, D]
            items_exp = item_vectors.unsqueeze(0)  # [1, N2, M, D]
            values = torch.sum(queries_exp * items_exp, dim=-1)  # [N1, N2, M]
            print(values)
            if self._scale_to_one:
                mean = torch.mean(values)
                std_dev = torch.std(values, unbiased=True)
                values = (values - mean) / std_dev

            if not is_similarity:
                values = -values

        elif self._search_index_info.metric_type == MetricType.EUCLID:
            queries_exp = query_vector.unsqueeze(1).unsqueeze(
                2
            )  # [N1, 1, 1, D]
            items_exp = item_vectors.unsqueeze(0)  # [1, N2, M, D]
            differences = queries_exp - items_exp  # [N1, N2, M, D]
            values = torch.norm(differences, dim=-1)  # [N1, N2, M]
            if is_similarity:
                values = -values

        else:
            raise ValueError(
                f"Unsupported MetricType: {self._search_index_info.metric_type}"
            )

        # Aggregate across the `M` dimension -> [N1, N2]
        if (
            self._search_index_info.metric_aggregation_type
            == MetricAggregationType.MIN
        ):
            # Differentiable soft minimum using log-sum-exp
            softmin_weights = torch.exp(
                -values / softmin_temperature
            )  # [N1, N2, M]
            softmin_weights /= softmin_weights.sum(
                dim=-1, keepdim=True
            )  # Normalize weights along M
            values = torch.sum(
                softmin_weights * values, dim=-1
            )  # Weighted sum -> [N1, N2]

        elif (
            self._search_index_info.metric_aggregation_type
            == MetricAggregationType.AVG
        ):
            values = values.mean(dim=-1)  # [N1, N2]

        else:
            raise ValueError(
                f"Unsupported MetricAggregationType: {self._search_index_info.metric_aggregation_type}"
            )

        return values

    def _calculate_binary_labels(
        self, corrected_values: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates binary selection labels from corrected distance values.

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

    def select(
        self,
        categories: List[ObjectWithDistance],
        query_vector: Optional[torch.Tensor] = None,
    ) -> List[int]:
        """
        Selects indices of objects based on vector comparisons.

        This method implements the selection logic by:
        1. Converting objects to tensors
        2. Calculating distances/similarities between query and category vectors
        3. Applying the margin threshold
        4. Calculating binary selection labels
        5. Returning indices of selected objects

        :param categories: List of objects with distance metrics and vectors
        :param query_vector: Query vector to compare against object vectors
        :return: List of indices of selected objects
        """
        query_vector = torch.Tensor(query_vector).unsqueeze(0)
        category_vectors = self._get_categories_tensor(categories)
        values = self._calculate_distance(
            query_vector,
            category_vectors,
            self._softmin_temperature,
            self._is_similarity,
        )
        positive_threshold_min = (
            1 - self._margin if self._is_similarity else self._margin
        )
        corrected_values = values - positive_threshold_min
        bin_labels = self._calculate_binary_labels(corrected_values)
        return torch.nonzero(bin_labels).T[1].tolist()
