from typing import List

import torch
import torch.nn.functional as F

from embedding_studio.embeddings.selectors.selector import AbstractSelector
from embedding_studio.models.embeddings.models import (
    MetricAggregationType,
    MetricType,
    SearchIndexInfo,
)


class DistBasedSelector(AbstractSelector):
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

    def _calculate_distance(
        self,
        query_vector: torch.Tensor,  # Shape: [N1, D]
        item_vectors: torch.Tensor,  # Shape: [N2, M, D]
        softmin_temperature: float = 1.0,  # Temperature for soft minimum
        is_similarity: bool = False,
    ) -> torch.Tensor:
        """
        Compute similarity or distance between queries and items.

        Args:
            query_vector: Tensor of shape [N1, D]
            item_vectors: Tensor of shape [N2, M, D]
            softmin_temperature: Temperature for differentiable softmin approximation
            is_similarity: Whether to treat values as similarity or distance.

        Returns:
            Tensor of shape [N1, N2]
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
            if self._scale_to_one:
                values = (values + 1) / 2

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
        raise NotImplementedError

    def select(
        self, query_vector: torch.Tensor, item_vectors: torch.Tensor
    ) -> List[int]:
        values = self._calculate_distance(
            query_vector,
            item_vectors,
            self._softmin_temperature,
            self._is_similarity,
        )
        positive_threshold_min = (
            1 - self._margin if self._is_similarity else self._margin
        )
        corrected_values = values - positive_threshold_min
        bin_labels = self._calculate_binary_labels(corrected_values)
        return torch.nonzero(bin_labels).T[1].tolist()
