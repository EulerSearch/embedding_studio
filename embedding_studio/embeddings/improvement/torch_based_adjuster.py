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
        softmin_temperature: float = 1.0,
    ):
        """
        Initialize the TorchBasedAdjuster.

        :param search_index_info: Information about the search index, containing
                                 metric type and aggregation method
        :param adjustment_rate: Learning rate for the optimizer (default: 0.1)
        :param num_iterations: Number of optimization iterations (default: 10)
        :param softmin_temperature: Temperature parameter for the softmin
                                   function (default: 1.0)
        """
        self.search_index_info = search_index_info
        self.adjustment_rate = adjustment_rate
        self.num_iterations = num_iterations
        self.softmin_temperature = softmin_temperature

    def compute_similarity(
        self,
        queries: torch.Tensor,  # Shape: [B, N1, D]
        items: torch.Tensor,  # Shape: [B, N2, M, D]
        softmin_temperature: float = 1.0,  # Temperature for soft minimum
    ) -> torch.Tensor:
        """
        Compute similarity between queries and items with aggregation.

        This method calculates the similarity between query vectors and item vectors
        using the metric type specified in the search_index_info. It then aggregates
        the similarities according to the specified aggregation type.

        :param queries: Tensor of shape [B, N1, D] where:
                        B = batch size
                        N1 = number of queries
                        D = embedding dimension
        :param items: Tensor of shape [B, N2, M, D] where:
                     B = batch size
                     N2 = number of items
                     M = number of elements per item
                     D = embedding dimension
        :param softmin_temperature: Temperature for differentiable softmin approximation
                                   (default: 1.0)
        :return: Tensor of shape [B, N2, M] containing the computed similarities
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
            softmin_weights = softmin_weights / softmin_weights.sum(
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
        """
        Adjust vectors using PyTorch optimization to improve search relevance.

        This method implements the abstract method from VectorsAdjuster. It performs
        gradient-based optimization to adjust clicked and non-clicked vectors to
        increase similarity between queries and clicked items while decreasing
        similarity between queries and non-clicked items.

        The method works by:
        1. Converting input data into PyTorch tensors
        2. Setting up gradients and optimization for clicked and non-clicked vectors
        3. Running multiple iterations of optimization to maximize similarity between
           queries and clicked items while minimizing similarity with non-clicked items
        4. Using a cubic (x^3) loss function to emphasize strong similarities/differences
        5. Updating the original data structure with the optimized vectors

        :param data_for_improvement: A list of ImprovementInput objects containing
                                     query vectors and corresponding clicked and
                                     non-clicked element vectors
        :return: The updated list of ImprovementInput objects with adjusted vectors
        """
        # Stack query vectors into a tensor of shape [B, N1, D] where:
        # B = batch size (number of ImprovementInput objects)
        # N1 = number of queries per input (typically 1)
        # D = embedding dimension
        queries = torch.stack(
            [inp.query.vector for inp in data_for_improvement]
        )  # [B, N1, D]

        # Stack clicked element vectors into a tensor of shape [B, N2, M, D] where:
        # B = batch size
        # N2 = number of clicked elements per input
        # M = number of vectors per clicked element
        # D = embedding dimension
        # The transpose operation rearranges dimensions to get the desired shape
        clicked_vectors = torch.stack(
            [
                torch.stack([ce.vector for ce in inp.clicked_elements], dim=1)
                for inp in data_for_improvement
            ]
        ).transpose(
            1, 2
        )  # [B, N2, M, D]

        # Similarly, stack non-clicked element vectors into a tensor of shape [B, N2, M, D]
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

        # Enable gradient tracking for the vectors that will be optimized
        clicked_vectors.requires_grad_(True)
        non_clicked_vectors.requires_grad_(True)

        # Set up AdamW optimizer with the specified learning rate (adjustment_rate)
        # Only clicked and non-clicked vectors will be optimized; query vectors remain fixed
        optimizer = torch.optim.AdamW(
            [clicked_vectors, non_clicked_vectors], lr=self.adjustment_rate
        )

        # Run optimization for the specified number of iterations
        for _ in range(self.num_iterations):
            # Reset gradients at the start of each iteration
            optimizer.zero_grad()

            # Compute similarity between queries and clicked items
            # This returns a tensor of shape [B, N2, M] containing similarity scores
            clicked_similarity = self.compute_similarity(
                queries, clicked_vectors, self.softmin_temperature
            )

            # Compute similarity between queries and non-clicked items
            # This also returns a tensor of shape [B, N2, M]
            non_clicked_similarity = self.compute_similarity(
                queries, non_clicked_vectors, self.softmin_temperature
            )

            # Define the loss function:
            # - Negative mean of clicked similarities (cubed) to maximize these similarities
            # - Plus mean of non-clicked similarities (cubed) to minimize these similarities
            # The cubic function (x^3) emphasizes larger values, effectively prioritizing
            # significant improvements in similarity scores
            loss = -torch.mean(clicked_similarity**3) + torch.mean(
                non_clicked_similarity**3
            )

            # Compute gradients with respect to clicked_vectors and non_clicked_vectors
            loss.backward()

            # Update vectors using the computed gradients
            optimizer.step()

        # Update the original data structure with the optimized vectors
        for batch_idx, inp in enumerate(data_for_improvement):
            # Update clicked element vectors in the original data structure
            # detach() removes the tensor from the computation graph to prevent
            # further gradient tracking and convert to a regular tensor
            for n2_idx, ce in enumerate(inp.clicked_elements):
                ce.vector = clicked_vectors[batch_idx, n2_idx].detach()

            # Update non-clicked element vectors in the original data structure
            for n2_idx, nce in enumerate(inp.non_clicked_elements):
                nce.vector = non_clicked_vectors[batch_idx, n2_idx].detach()

        # Return the updated data structure with optimized vectors
        return data_for_improvement
