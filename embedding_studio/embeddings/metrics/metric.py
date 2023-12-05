from abc import ABC, abstractmethod
from typing import List, Tuple

import torch

from embedding_studio.embeddings.data.clickstream.query_retriever import (
    QueryRetriever,
)
from embedding_studio.embeddings.data.clickstream.raw_session import (
    ClickstreamSession,
)
from embedding_studio.embeddings.data.storages.storage import ItemsStorage
from embedding_studio.embeddings.features.extractor import FeaturesExtractor
from embedding_studio.worker.experiments.metrics_accumulator import MetricValue


class MetricCalculator(ABC):
    """Interface of metrics calculator"""

    @abstractmethod
    @torch.no_grad()
    def __call__(
        self,
        batch: List[Tuple[ClickstreamSession, ClickstreamSession]],
        extractor: FeaturesExtractor,
        items_storage: ItemsStorage,
        query_retriever: QueryRetriever,
    ) -> List[MetricValue]:
        """Calculate abstract metric value over provided batch of items.

        :param batch: batch of pairs clickstream sessions (not_irrelevant, irrelevant)
        :type batch: List[Tuple[ClickstreamSession, ClickstreamSession]]
        :param extractor: object to extract SessionFeatures out of provided sessions
        :type extractor: FeaturesExtractor
        :param items_storage: items dataset
        :type items_storage: ItemsStorage
        :param query_retriever: how to retrieve a value related to session query
        :type query_retriever: QueryRetriever
        :return: list of calculated metrics
        :type: List[MetricValue]
        """
