from abc import ABC, abstractmethod
from typing import List, Tuple

import torch

from embedding_studio.clickstream_storage.query_retriever import QueryRetriever
from embedding_studio.embeddings.data.storages.storage import ItemsStorage
from embedding_studio.embeddings.features.extractor import FeaturesExtractor
from embedding_studio.embeddings.features.feature_extractor_input import (
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
        items_storage: ItemsStorage,
        query_retriever: QueryRetriever,
    ) -> List[MetricValue]:
        """Calculate abstract metric value over provided batch of items.

        :param batch: batch of pairs clickstream inputs (not_irrelevant, irrelevant)
        :param extractor: object to extract SessionFeatures out of provided inputs
        :param items_storage: items dataset
        :param query_retriever: how to retrieve a value related to session query
        :return: list of calculated metrics
        """
