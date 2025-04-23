from abc import ABC, abstractmethod
from typing import List, Optional

import torch

from embedding_studio.models.embeddings.objects import ObjectWithDistance


class AbstractSelector(ABC):
    """
    Abstract base class for selector algorithms that filter embedding search results.

    This class provides the framework for implementing different selection strategies
    for filtering objects based on their distance metrics and embedding vectors.
    """

    def _get_categories_tensor(
        self, items: List[ObjectWithDistance]
    ) -> torch.Tensor:
        """
        Converts a list of ObjectWithDistance instances into a padded tensor of vectors.

        This method creates a tensor containing all part vectors from the provided objects,
        padding as necessary to ensure consistent dimensions across all objects.

        :param items: List of objects with distance metrics and embedding vectors
        :return: A tensor of shape [N, D, M] where:
                 - N is the number of objects
                 - D is the embedding dimension
                 - M is the maximum number of parts across all objects
        """
        max_parts = max(len(obj.parts) for obj in items)
        category_vectors = torch.stack(
            [
                torch.stack(
                    [torch.Tensor(part.vector) for part in obj.parts]
                    + [torch.zeros(len(obj.parts[0].vector))]
                    * (max_parts - len(obj.parts)),
                    dim=1,
                )  # Pad
                for obj in items
            ]
        ).transpose(1, 2)

        return category_vectors

    @abstractmethod
    def select(
        self,
        categories: List[ObjectWithDistance],
        query_vector: Optional[torch.Tensor] = None,
    ) -> List[int]:
        """
        Selects indices of objects that meet the selection criteria.

        This method determines which objects from the provided list should be selected
        based on their distances, vectors, and the optional query vector.

        :param categories: List of objects with distance metrics and embedding vectors
        :param query_vector: Optional tensor representing the query embedding
        :return: List of indices of selected objects

        Example implementation:
        ```python
        def select(self, categories: List[ObjectWithDistance],
                  query_vector: Optional[torch.Tensor] = None) -> List[int]:
            # Convert raw distances to selection scores
            scores = torch.tensor([obj.distance for obj in categories])

            # Apply threshold to determine which objects to select
            threshold = 0.5
            selected_indices = torch.where(scores < threshold)[0].tolist()

            return selected_indices
        ```
        """

    @property
    @abstractmethod
    def vectors_are_needed(self) -> bool:
        """
        Indicates whether this selector requires access to the actual embedding vectors.

        Some selectors can operate solely on pre-computed distances, while others
        need direct access to the embedding vectors for their selection logic.

        :return: True if the selector needs embedding vectors, False if it can work with distances only

        Example implementation:
        ```python
        @property
        def vectors_are_needed(self) -> bool:
            # This selector works with pre-computed distances only
            return False
        ```
        """
        raise NotImplementedError
