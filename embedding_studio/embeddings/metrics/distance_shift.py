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
    """
    Metric calculator that measures how ranks of provided inputs changed after model processing.

    This class calculates two metrics:
    1. not_irrelevant_dist_shift: Average rank change for relevant/positive items
    2. irrelevant_dist_shift: Average rank change for irrelevant/negative items

    For similarity metrics, higher ranks for relevant inputs are better.
    For distance metrics, lower ranks for relevant inputs are better.
    """

    def _calc_dist_shift(
        self,
        fine_tuning_input: FineTuningInput,
        extractor: FeaturesExtractor,
        items_set: ItemsSet,
        query_retriever: QueryRetriever,
    ) -> float:
        """
        Calculate the distance/similarity shift for a single fine-tuning input.

        :param fine_tuning_input: Input containing query and events information
        :param extractor: Object to extract features and perform model inference
        :param items_set: Dataset of items to retrieve embeddings from
        :param query_retriever: Object to retrieve query embedding
        :return: Average shift in rank for the given input
        """
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
        # Initialize lists to store shift values for both types of inputs
        not_irrelevenat_shifts: List[
            float
        ] = []  # Stores shifts for relevant/positive inputs
        irrelevenat_shifts: List[
            float
        ] = []  # Stores shifts for irrelevant/negative inputs

        # Iterate through each pair of inputs in the batch
        for index, (not_irrelevenat_input, irrelevant_input) in enumerate(
            batch
        ):
            # Process relevant/positive inputs if present
            if not_irrelevenat_input is not None:
                # Calculate the distance shift for this relevant input and add to collection
                # This measures how the rank has improved (or worsened) after model processing
                not_irrelevenat_shifts.append(
                    self._calc_dist_shift(
                        not_irrelevenat_input,
                        extractor,
                        items_set,
                        query_retriever,
                    )
                )

            # Process irrelevant/negative inputs if present
            if irrelevant_input is not None:
                # Calculate the distance shift for this irrelevant input and add to collection
                # For irrelevant inputs, we want to see ranks decreasing (for similarity metrics)
                # or increasing (for distance metrics)
                irrelevenat_shifts.append(
                    self._calc_dist_shift(
                        irrelevant_input,
                        extractor,
                        items_set,
                        query_retriever,
                    )
                )

        # Return metrics as a list of MetricValue objects
        return [
            # First metric: average shift for relevant/positive inputs
            # Positive values indicate improvement in ranking
            MetricValue(
                "not_irrelevant_dist_shift",
                (
                    float(
                        np.mean(not_irrelevenat_shifts)
                    )  # Calculate average if we have values
                    if len(not_irrelevenat_shifts) > 0
                    else 0.0  # Default to 0.0 if no relevant inputs were processed
                ),
            ),
            # Second metric: average shift for irrelevant/negative inputs
            # Positive values indicate that irrelevant items are being properly pushed down/away
            MetricValue(
                "irrelevant_dist_shift",
                (
                    float(
                        np.mean(irrelevenat_shifts)
                    )  # Calculate average if we have values
                    if len(irrelevenat_shifts) > 0
                    else 0.0  # Default to 0.0 if no irrelevant inputs were processed
                ),
            ),
        ]
