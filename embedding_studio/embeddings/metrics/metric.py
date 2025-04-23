from abc import ABC, abstractmethod
from typing import List, Tuple

import torch

from embedding_studio.clickstream_storage.query_retriever import QueryRetriever
from embedding_studio.embeddings.data.items.items_set import ItemsSet
from embedding_studio.embeddings.features.extractor import FeaturesExtractor
from embedding_studio.embeddings.features.fine_tuning_input import (
    FineTuningInput,
)
from embedding_studio.experiments.metrics_accumulator import MetricValue


class MetricCalculator(ABC):
    """
    Interface for metrics calculators used to evaluate embedding model performance.

    This abstract base class defines the contract for all metric calculators,
    ensuring they implement a standard way to calculate performance metrics
    over batches of fine-tuning inputs.
    """

    @abstractmethod
    @torch.no_grad()
    def __call__(
        self,
        batch: List[Tuple[FineTuningInput, FineTuningInput]],
        extractor: FeaturesExtractor,
        items_set: ItemsSet,
        query_retriever: QueryRetriever,
    ) -> List[MetricValue]:
        """
        Calculate metric values over a provided batch of items.

        This method evaluates model performance by calculating specific metrics
        over pairs of inputs (typically one relevant and one irrelevant).
        The @torch.no_grad decorator ensures that no gradients are calculated
        during metric evaluation, which improves performance during validation.

        :param batch: Batch of pairs of clickstream inputs where each pair consists of
                     (not_irrelevant input, irrelevant input). Either element can be None.
        :param extractor: Object to extract FineTuningFeatures from provided inputs
                         and to access the underlying model for inference
        :param items_set: Dataset of items used to retrieve embeddings by item IDs
        :param query_retriever: Function to retrieve embedding values for input queries
        :return: List of calculated metrics as MetricValue objects

        Example implementation:
        def __call__(
            self,
            batch: List[Tuple[FineTuningInput, FineTuningInput]],
            extractor: FeaturesExtractor,
            items_set: ItemsSet,
            query_retriever: QueryRetriever,
        ) -> List[MetricValue]:
            results = []
            for inputs_pair in batch:
                value = self._calculate_metric(inputs_pair, extractor)
                results.append(value)

            return [MetricValue("my_metric", np.mean(results))]
        """
        raise NotImplemented()
