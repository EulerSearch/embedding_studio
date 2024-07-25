from typing import List, Tuple

import numpy as np
import torch
from torch import FloatTensor

from embedding_studio.clickstream_storage.query_retriever import QueryRetriever
from embedding_studio.embeddings.data.items.items_set import ItemsSet
from embedding_studio.embeddings.features.extractor import FeaturesExtractor
from embedding_studio.embeddings.features.fine_tuning_input import (
    FineTuningInput,
)
from embedding_studio.embeddings.metrics.metric import MetricCalculator
from embedding_studio.experiments.metrics_accumulator import MetricValue


class DistanceShift(MetricCalculator):
    def _calc_dist_shift(
        self,
        fine_tuning_input: FineTuningInput,
        extractor: FeaturesExtractor,
        items_set: ItemsSet,
        query_retriever: QueryRetriever,
    ) -> float:
        # TODO: encapsulate inference in one class / object
        query_vector: FloatTensor = extractor.model.forward_query(
            query_retriever(fine_tuning_input.query)
        )
        items_vectors: FloatTensor = extractor.model.forward_items(
            items_set.items_by_ids(
                fine_tuning_input.events + fine_tuning_input.not_events
            )
        )

        ranks: FloatTensor = (
            extractor.ranker(query_vector, items_vectors).cpu().tolist()
        )

        # for similarity ranks should be higher for results of not irrelevant inputs,
        # for distances should be vice versa
        target: int = 1 if extractor.is_similarity else -1
        compare = lambda prev, new: target * float(new - prev)
        results: List[str] = fine_tuning_input.events
        if fine_tuning_input.is_irrelevant:
            results = fine_tuning_input.results
            compare = lambda prev, new: target * float(prev - new)

        return float(
            np.mean(
                [
                    compare(fine_tuning_input.ranks[id_], new_rank)
                    for id_, new_rank in zip(results, ranks)
                ]
            )
        )

    @torch.no_grad()
    def __call__(
        self,
        batch: List[Tuple[FineTuningInput, FineTuningInput]],
        extractor: FeaturesExtractor,
        items_set: ItemsSet,
        query_retriever: QueryRetriever,
    ) -> List[MetricValue]:
        """Calculate metric, that means how ranks of provided inputs were changed .

        :param batch: batch of pairs clickstream inputs (not_irrelevant, irrelevant)
        :param extractor: object to extract FineTuningFeatures out of provided inputs
        :param items_set: items dataset
        :param query_retriever: how to retrieve a value related to fine-tuning input's query
        :return: list of calculated not_irrelevant_dist_shift and irrelevant_dist_shift metrics
        """
        not_irrelevenat_shifts: List[float] = []
        irrelevenat_shifts: List[float] = []
        for index, (not_irrelevenat_input, irrelevant_input) in enumerate(
            batch
        ):
            if not_irrelevenat_input is not None:
                not_irrelevenat_shifts.append(
                    self._calc_dist_shift(
                        not_irrelevenat_input,
                        extractor,
                        items_set,
                        query_retriever,
                    )
                )

            if irrelevant_input is not None:
                irrelevenat_shifts.append(
                    self._calc_dist_shift(
                        irrelevant_input,
                        extractor,
                        items_set,
                        query_retriever,
                    )
                )

        return [
            MetricValue(
                "not_irrelevant_dist_shift",
                (
                    float(np.mean(not_irrelevenat_shifts))
                    if len(not_irrelevenat_shifts) > 0
                    else 0.0
                ),
            ),
            MetricValue(
                "irrelevant_dist_shift",
                (
                    float(np.mean(irrelevenat_shifts))
                    if len(irrelevenat_shifts) > 0
                    else 0.0
                ),
            ),
        ]
