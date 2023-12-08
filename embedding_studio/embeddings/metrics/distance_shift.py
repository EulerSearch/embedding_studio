from typing import List, Tuple

import numpy as np
import torch
from torch import FloatTensor

from embedding_studio.embeddings.data.clickstream.query_retriever import (
    QueryRetriever,
)
from embedding_studio.embeddings.data.clickstream.raw_session import (
    ClickstreamSession,
)
from embedding_studio.embeddings.data.storages.storage import ItemsStorage
from embedding_studio.embeddings.features.extractor import FeaturesExtractor
from embedding_studio.embeddings.metrics.metric import MetricCalculator
from embedding_studio.worker.experiments.metrics_accumulator import MetricValue


class DistanceShift(MetricCalculator):
    def _calc_dist_shift(
        self,
        session: ClickstreamSession,
        extractor: FeaturesExtractor,
        items_storage: ItemsStorage,
        query_retriever: QueryRetriever,
    ) -> float:
        # TODO: encapsulate inference in one class / object
        query_vector: FloatTensor = extractor.model.forward_query(
            query_retriever(session.query)
        )
        items_vectors: FloatTensor = extractor.model.forward_items(
            items_storage.items_by_ids(session.events + session.not_events)
        )

        ranks: FloatTensor = (
            extractor.ranker(query_vector, items_vectors).cpu().tolist()
        )

        # for similarity ranks should be higher for results of not irrelevant sessions,
        # for distances should be vice versa
        target: int = 1 if extractor.is_similarty else -1
        compare = lambda prev, new: target * float(new - prev)
        results: List[str] = session.events
        if session.is_irrelevant:
            results = session.results
            compare = lambda prev, new: target * float(prev - new)

        return float(
            np.mean(
                [
                    compare(session.ranks[id_], new_rank)
                    for id_, new_rank in zip(results, ranks)
                ]
            )
        )

    @torch.no_grad()
    def __call__(
        self,
        batch: List[Tuple[ClickstreamSession, ClickstreamSession]],
        extractor: FeaturesExtractor,
        items_storage: ItemsStorage,
        query_retriever: QueryRetriever,
    ) -> List[MetricValue]:
        """Calculate metric, that means how ranks of provided sessions were changed .

        :param batch: batch of pairs clickstream sessions (not_irrelevant, irrelevant)
        :type batch: List[Tuple[ClickstreamSession, ClickstreamSession]]
        :param extractor: object to extract SessionFeatures out of provided sessions
        :type extractor: FeaturesExtractor
        :param items_storage: items dataset
        :type items_storage: ItemsStorage
        :param query_retriever: how to retrieve a value related to session query
        :type query_retriever: QueryRetriever
        :return: list of calculated not_irrelevant_dist_shift and irrelevant_dist_shift metrics
        :type: List[MetricValue]
        """
        not_irrelevenat_shifts: List[float] = []
        irrelevenat_shifts: List[float] = []
        for index, (not_irrelevenat_session, irrelevant_session) in enumerate(
            batch
        ):
            not_irrelevenat_shifts.append(
                self._calc_dist_shift(
                    not_irrelevenat_session,
                    extractor,
                    items_storage,
                    query_retriever,
                )
            )
            irrelevenat_shifts.append(
                self._calc_dist_shift(
                    irrelevant_session,
                    extractor,
                    items_storage,
                    query_retriever,
                )
            )

        return [
            MetricValue(
                "not_irrelevant_dist_shift",
                float(np.mean(not_irrelevenat_shifts)),
            ),
            MetricValue(
                "irrelevant_dist_shift", float(np.mean(irrelevenat_shifts))
            ),
        ]
