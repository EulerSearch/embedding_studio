from typing import List

import torch
import torch.nn.functional as F

from embedding_studio.embeddings.data.clickstream.improvement_input import (
    ImprovementInput,
)
from embedding_studio.embeddings.improvement.vectors_adjuster import (
    VectorsAdjuster,
)
from embedding_studio.models.embeddings.models import (
    MetricAggregationType,
    MetricType,
    SearchIndexInfo,
)


# TorchBasedAdjuster implementation
class TorchBasedAdjuster(VectorsAdjuster):
    def __init__(
        self,
        search_index_info: SearchIndexInfo,
        adjustment_rate: float = 0.1,
        num_iterations: int = 10,
    ):
        self.search_index_info = search_index_info
        self.adjustment_rate = adjustment_rate
        self.num_iterations = num_iterations

    def compute_similarity(
        self,
        queries: torch.Tensor,  # Shape: [B, N1, D]
        items: torch.Tensor,  # Shape: [B, N2, M, D]
        softmin_temperature: float = 1.0,  # Temperature for soft minimum
    ) -> torch.Tensor:
        """
        Compute similarity between queries and items with aggregation.

        Args:
            queries: Tensor of shape [B, N1, D]
            items: Tensor of shape [B, N2, M, D]
            metric_type: MetricType to use for similarity computation
            aggregation: MetricAggregationType for aggregating similarities
            softmin_temperature: Temperature for differentiable softmin approximation

        Returns:
            Tensor of shape [B, N2, M]
        """
        if self.search_index_info.metric_type == MetricType.COSINE:
            queries_norm = (
                F.normalize(queries, p=2, dim=-1).unsqueeze(2).unsqueeze(3)
            )  # [B, N1, 1, 1, D]
            items_norm = F.normalize(items, p=2, dim=-1).unsqueeze(
                1
            )  # [B, 1, N2, M, D]
            similarities = torch.sum(
                queries_norm * items_norm, dim=-1
            )  # [B, N2, N1, M]
        elif self.search_index_info.metric_type == MetricType.DOT:
            queries_exp = queries.unsqueeze(2).unsqueeze(3)  # [B, N1, 1, 1, D]
            items_exp = items.unsqueeze(1)  # [B, 1, N2, M, D]
            similarities = torch.sum(
                queries_exp * items_exp, dim=-1
            )  # [B, N2, N1, M]
        elif self.search_index_info.metric_type == MetricType.EUCLID:
            queries_exp = queries.unsqueeze(2).unsqueeze(3)  # [B, N1, 1, 1, D]
            items_exp = items.unsqueeze(1)  # [B, 1, N2, M, D]
            differences = queries_exp - items_exp  # [B, N1, N2, M, D]
            distances = torch.norm(differences, dim=-1)  # [B, N2, N1, M]
            similarities = -distances  # Negative for similarity
        else:
            raise ValueError(
                f"Unsupported MetricType: {self.search_index_info.metric_type}"
            )

        # Shape after similarities: [B, N2, N1, M]

        if (
            self.search_index_info.metric_aggregation_type
            == MetricAggregationType.MIN
        ):
            # Differentiable soft minimum using log-sum-exp
            softmin_weights = torch.exp(
                -similarities / softmin_temperature
            )  # [B, N2, N1, M]
            softmin_weights /= softmin_weights.sum(
                dim=2, keepdim=True
            )  # Normalize weights along N1
            similarities = torch.sum(
                softmin_weights * similarities, dim=2
            )  # Weighted sum -> [B, N2, M]
        elif (
            self.search_index_info.metric_aggregation_type
            == MetricAggregationType.AVG
        ):
            # Avg aggregation: compute mean across the query dimension (N1)
            similarities = similarities.mean(dim=2)  # [B, N2, M]
        else:
            raise ValueError(
                f"Unsupported MetricAggregationType: {self.search_index_info.metric_aggregation_type}"
            )

        return similarities  # [B, N2, M]

    def adjust_vectors(
        self, data_for_improvement: List[ImprovementInput]
    ) -> List[ImprovementInput]:
        queries = torch.stack(
            [inp.query.vector for inp in data_for_improvement]
        )  # [B, N1, D]
        clicked_vectors = torch.stack(
            [
                torch.stack([ce.vector for ce in inp.clicked_elements], dim=1)
                for inp in data_for_improvement
            ]
        ).transpose(
            1, 2
        )  # [B, N2, M, D]
        non_clicked_vectors = torch.stack(
            [
                torch.stack(
                    [nce.vector for nce in inp.non_clicked_elements], dim=1
                )
                for inp in data_for_improvement
            ]
        ).transpose(
            1, 2
        )  # [B, N2, M, D]

        clicked_vectors.requires_grad_(True)
        non_clicked_vectors.requires_grad_(True)

        optimizer = torch.optim.AdamW(
            [clicked_vectors, non_clicked_vectors], lr=self.adjustment_rate
        )

        for _ in range(self.num_iterations):
            optimizer.zero_grad()

            clicked_similarity = self.compute_similarity(
                queries, clicked_vectors, self.search_index_info.metric_type
            )
            non_clicked_similarity = self.compute_similarity(
                queries,
                non_clicked_vectors,
                self.search_index_info.metric_type,
            )

            loss = -torch.mean(clicked_similarity**3) + torch.mean(
                non_clicked_similarity**3
            )
            loss.backward()
            optimizer.step()

        # Update the original data structure
        for batch_idx, inp in enumerate(data_for_improvement):
            for n2_idx, ce in enumerate(inp.clicked_elements):
                ce.vector = clicked_vectors[batch_idx, n2_idx].detach()
            for n2_idx, nce in enumerate(inp.non_clicked_elements):
                nce.vector = non_clicked_vectors[batch_idx, n2_idx].detach()

        return data_for_improvement
