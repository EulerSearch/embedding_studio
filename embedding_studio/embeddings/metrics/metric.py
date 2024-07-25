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
    """Interface of metrics calculator"""

    @abstractmethod
    @torch.no_grad()
    def __call__(
        self,
        batch: List[Tuple[FineTuningInput, FineTuningInput]],
        extractor: FeaturesExtractor,
        items_set: ItemsSet,
        query_retriever: QueryRetriever,
    ) -> List[MetricValue]:
        """Calculate abstract metric value over provided batch of items.

        :param batch: batch of pairs clickstream inputs (not_irrelevant, irrelevant)
        :param extractor: object to extract FineTuningFeatures out of provided inputs
        :param items_set: items dataset
        :param query_retriever: how to retrieve a value related to session query
        :return: list of calculated metrics
        """
