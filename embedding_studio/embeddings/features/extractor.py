import logging
import random
from typing import Callable, List, Optional, Tuple

import pytorch_lightning as pl
import torch
from torch import FloatTensor, Tensor

from embedding_studio.embeddings.data.items.items_set import ItemsSet
from embedding_studio.embeddings.features.clicks_aggregator.clicks_aggregator import (
    ClicksAggregator,
)
from embedding_studio.embeddings.features.clicks_aggregator.max_aggregator import (
    MaxClicksAggregator,
)
from embedding_studio.embeddings.features.event_confidences import (
    dummy_confidences,
)
from embedding_studio.embeddings.features.fine_tuning_features import (
    FineTuningFeatures,
)
from embedding_studio.embeddings.features.fine_tuning_input import (
    FineTuningInput,
)
from embedding_studio.embeddings.features.ranks_aggregators.mean_aggregator import (
    MeanAggregator,
)
from embedding_studio.embeddings.features.ranks_aggregators.ranks_aggregator import (
    RanksAggregator,
)
from embedding_studio.embeddings.models.interface import (
    EmbeddingsModelInterface,
)
from embedding_studio.experiments.finetuning_params import ExamplesType

COSINE_SIMILARITY = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

logger = logging.getLogger(__name__)


class FeaturesExtractor(pl.LightningModule):
    def __init__(
        self,
        model: EmbeddingsModelInterface,
        ranker: Optional[
            Callable[[FloatTensor, FloatTensor], FloatTensor]
        ] = COSINE_SIMILARITY,
        is_similarity: Optional[bool] = True,
        not_irrelevant_only: Optional[bool] = True,
        negative_downsampling_factor: Optional[float] = 1.0,
        min_abs_difference_threshold: Optional[float] = 0.0,
        max_abs_difference_threshold: Optional[float] = 1.0,
        confidence_calculator: Optional[Callable] = dummy_confidences,
        examples_order: Optional[List[ExamplesType]] = None,
        ranks_aggregator: RanksAggregator = MeanAggregator(),
        clicks_aggregator: ClicksAggregator = MaxClicksAggregator(),
    ):
        """Logic of extracting features:
        1. Positive and negative examples ranks
        2. Event confidences
        3. Target: 1 if is_similarity else -1

        and pack it in FineTuningFeatures object.

        :param model: embedding model itself
        :param ranker: ranking function (query, items) -> ranks (defult: cosine similarity)
        :param is_similarity: is ranking function similarity like or distance (default: True)
        :param not_irrelevant_only: use only not irrelevant inputs (default: True)
                                    True - Triplet loss
                                    False - Contrastive-like loss
        :param negative_downsampling_factor: in real tasks amount of results is much larger than  not-results,
                                             use this parameters to fix a balance (default: 0.5)
        :param min_abs_difference_threshold: filter out soft pairs abs(neg_dist - pos_dist) < small value
        :param max_abs_difference_threshold: filter out hard pairs abs(neg_dist - pos_dist) > huge value
        :param confidence_calculator: function to calculate results confidences (default: dummy_confidences)
        :param examples_order: order of passing examples to a trainer (default: None)
        :param ranks_aggregator: if an item is split into subitems, ranks should be aggregated
        :param clicks_aggregator: if an item is split into subtimes, clicks should be aggregated too
        """
        super(FeaturesExtractor, self).__init__()
        # Check model type
        if not isinstance(model, EmbeddingsModelInterface):
            raise ValueError(
                "Model must be an instance of EmbeddingsModelInterface."
            )
        self.model = model

        # Check ranker type and value
        if not callable(ranker):
            raise ValueError("Ranker must be a callable function.")
        self.ranker = ranker

        # Check is_similarity type
        if not isinstance(is_similarity, bool):
            raise ValueError("is_similarity must be a boolean.")
        self.is_similarity = is_similarity

        # Check not_irrelevant_only type
        if not isinstance(not_irrelevant_only, bool):
            raise ValueError("not_irrelevant_only must be a boolean.")
        self.not_irrelevant_only = not_irrelevant_only

        # TODO: use pydantic models here
        if (
            not isinstance(negative_downsampling_factor, float)
            or negative_downsampling_factor < 0.0
            or negative_downsampling_factor >= 1
        ):
            raise ValueError(
                "negative downsampling factor should be un range (0.0, 1.0)"
            )
        self.negative_downsampling_factor = negative_downsampling_factor

        if (
            not isinstance(min_abs_difference_threshold, float)
            or min_abs_difference_threshold < 0.0
        ):
            raise ValueError(
                "min_abs_difference_threshold should be positive numeric"
            )
        self.min_abs_difference_threshold = min_abs_difference_threshold
        if (
            not isinstance(max_abs_difference_threshold, float)
            or max_abs_difference_threshold <= 0.0
        ):
            raise ValueError(
                "max_abs_difference_threshold should be positive numeric"
            )
        self.max_abs_difference_threshold = max_abs_difference_threshold
        self.confidence_calculator = confidence_calculator

        if not examples_order:
            examples_order = [ExamplesType.all_examples]
            logger.debug("All types of examples will be used in training")

        if len({isinstance(e, ExamplesType) for e in examples_order}) > 1:
            raise ValueError(
                "Some of exmaple types are not instances of ExampleType"
            )
        self.exmaples_order = (
            examples_order  # TODO: use different examples order
        )

        self.ranks_aggregator = ranks_aggregator
        self.clicks_aggregator = clicks_aggregator

    def _confidences(
        self, fine_tuning_input: FineTuningInput, not_events: List[str]
    ) -> Tuple[Tensor, Tensor]:
        """Calculate confidences for a given fine-tuning input's items.

        :param fine_tuning_input: provided fine-tuning input
        :param not_events: not-results (negatives) used for ranks prediction
        :return: positive (results) confidences, negative (not-results) confidences
        """
        only_used_ids = [
            id_
            for id_ in fine_tuning_input.results
            if id_ in fine_tuning_input.events or id_ in not_events
        ]

        # Initialize dictionaries to store grouped ranks and clicks
        group_ranks = {}
        group_clicks = {}
        group_map = {
            id_: fine_tuning_input.get_object_id(id_) for id_ in only_used_ids
        }
        for id_ in only_used_ids:
            group_id = group_map[id_]
            if group_id not in group_ranks:
                group_ranks[group_id] = []
                group_clicks[group_id] = []
            group_ranks[group_id].append(fine_tuning_input.ranks[id_])
            group_clicks[group_id].append(
                1 if id_ in fine_tuning_input.events else 0
            )

        # Aggregate ranks and binary clicks across part_to_object_dict
        aggregated_ranks = dict()
        for group_id in group_ranks:
            if len(group_ranks[group_id]) == 1:
                aggregated_ranks[group_id] = group_ranks[group_id][0]
            else:
                aggregated_ranks[group_id] = self.ranks_aggregator(
                    group_ranks[group_id], differentiable=False
                )

        aggregated_clicks = self.clicks_aggregator(group_clicks)

        # Calculate confidences for aggregated ranks and clicks
        group_confidences = {}
        for group_id in aggregated_ranks:
            rank = torch.FloatTensor([aggregated_ranks[group_id]])
            clicks = torch.FloatTensor([aggregated_clicks[group_id]])
            group_confidences[group_id] = self.confidence_calculator(
                rank, clicks
            )

        # Sort confidences among positive and negative types
        positive_confidences = torch.zeros(len(fine_tuning_input.events))
        negative_confidences = torch.zeros(len(not_events))

        # Assign confidences to appropriate positive or negative tensors based on group participation
        for id_ in only_used_ids:
            group_id = group_map[id_]
            if id_ in fine_tuning_input.events:
                index = fine_tuning_input.events.index(id_)
                positive_confidences[index] = group_confidences[group_id]
            elif id_ in not_events:
                index = not_events.index(id_)
                negative_confidences[index] = group_confidences[group_id]

        return positive_confidences.to(self.device), negative_confidences.to(
            self.device
        )

    def _downsample_not_events(
        self, fine_tuning_input: FineTuningInput
    ) -> List[str]:
        # Group not-events by their respective group IDs
        group_to_not_events = {}
        for id_ in fine_tuning_input.not_events:
            group_id = fine_tuning_input.get_object_id(id_)
            if group_id not in group_to_not_events:
                group_to_not_events[group_id] = []
            group_to_not_events[group_id].append(id_)

        # Calculate the number of part_to_object_dict to include based on the downsampling factor
        total_groups = len(group_to_not_events)
        groups_to_sample = int(
            self.negative_downsampling_factor * total_groups
        )

        # Randomly select part_to_object_dict to be included in the downsampled set
        selected_groups = random.sample(
            list(group_to_not_events.keys()), groups_to_sample
        )

        # Collect all not-events from the selected part_to_object_dict
        downsampled_not_events = []
        for group_id in selected_groups:
            downsampled_not_events.extend(group_to_not_events[group_id])

        return downsampled_not_events

    def _get_fine_tuning_features(
        self, fine_tuning_input: FineTuningInput, dataset: ItemsSet
    ) -> FineTuningFeatures:
        """Calculate features for a single fine-tuning input, ensuring tensors are prepared for gradient calculations in fine-tuning.

        :param fine_tuning_input: given fine-tuning input
        :param dataset: items items_set related to a given fine-tuning input
        :return: provided fine-tuning iput's features
        """
        features = FineTuningFeatures()

        # Downsample not-events while respecting group boundaries
        not_events = self._downsample_not_events(fine_tuning_input)

        # Calculate confidences for both positive and negative items, considering part_to_object_dict
        positive_confidences, negative_confidences = self._confidences(
            fine_tuning_input, not_events
        )

        # Calculate query and item vectors
        query_vector = self.model.forward_query(fine_tuning_input.query)
        items, ids = dataset.items_by_ids(
            fine_tuning_input.events + not_events
        )
        items_vectors = self.model.forward_items(items)

        # Map items to their part_to_object_dict
        group_map = {id_: fine_tuning_input.get_object_id(id_) for id_ in ids}

        # Group items vectors by their group IDs for aggregation
        grouped_vectors = {}
        for i, id_ in enumerate(ids):
            group_id = group_map[id_]
            if group_id not in grouped_vectors:
                grouped_vectors[group_id] = []
            grouped_vectors[group_id].append(items_vectors[i])

        # Aggregate vectors within each group using the rank aggregator's method
        for group_id in grouped_vectors:
            vectors = torch.stack(grouped_vectors[group_id])
            grouped_vectors[group_id] = vectors

        # Calculate ranks for aggregated vectors
        aggregated_ranks = {}
        for group_id, vector in grouped_vectors.items():
            ranks = self.ranker(query_vector, vector.unsqueeze(0))
            aggregated_ranks[group_id] = self.ranks_aggregator(
                ranks.squeeze(), differentiable=True
            )

        # Prepare lists to collect ranks and confidences for tensor conversion
        positive_ranks = []
        negative_ranks = []
        positive_confidences_values = []
        negative_confidences_values = []

        # Assign aggregated ranks, confidences, and targets to each ID based on their group
        for id_ in ids:
            group_id = group_map[id_]
            if id_ in fine_tuning_input.events:
                positive_ranks.append(aggregated_ranks[group_id])
                positive_confidences_values.append(
                    positive_confidences[group_id]
                )
            else:
                negative_ranks.append(aggregated_ranks[group_id])
                negative_confidences_values.append(
                    negative_confidences[group_id]
                )

        # Convert lists to tensors for backprop compatibility
        features.positive_ranks = torch.stack(positive_ranks)
        features.negative_ranks = torch.stack(negative_ranks)
        features.positive_confidences = torch.tensor(
            positive_confidences_values
        )
        features.negative_confidences = torch.tensor(
            negative_confidences_values
        )

        # Prepare the target value based on similarity settings
        features.target = torch.tensor(
            [1 if self.is_similarity else -1] * len(not_events)
        ).to(self.device)

        # Filter out noises
        features.clamp_diff_in(
            self.min_abs_difference_threshold,
            self.max_abs_difference_threshold,
        )

        return features

    def _get_paired_inputs_features(
        self,
        not_irrelevant_input: FineTuningInput,
        irrelevant_input: FineTuningInput,
        dataset: ItemsSet,
    ) -> FineTuningFeatures:
        """Calculate features for a given pair: irrelevant and not irrelevant inputs

        :param not_irrelevant_input: not-irrelevant fine-tuning input
        :param irrelevant_input: irrelevant fine-tuning input
        :param dataset: items_set of items related to fine-tuning inputs
        :return: features related for both irrelevant and not irrelevant inputs
        """
        not_irrelevant_features: FineTuningFeatures = (
            self._get_fine_tuning_features(not_irrelevant_input, dataset)
        )
        irrelevant_features: FineTuningFeatures = (
            self._get_fine_tuning_features(irrelevant_input, dataset)
        )

        irrelevant_features.use_positive_from(not_irrelevant_features)

        not_irrelevant_features += irrelevant_features

        return not_irrelevant_features

    def forward(
        self,
        batch: List[Tuple[FineTuningInput, FineTuningInput]],
        dataset: ItemsSet,
    ) -> FineTuningFeatures:
        """Calculate features for a given batch of pairs: irrelevant and not irrelevant inputs

        :param batch: list of pairs: irrelevant and not irrelevant inputs
        :param dataset:  items_set of items related to clickstream inputs
        :return: fine-tuning features related to a given batch
        """
        features = FineTuningFeatures()

        for not_irrelevant_input, irrelevant_input in batch:
            if len(not_irrelevant_input.events) == 0:
                logger.warning("Not irrelevant input has no results")
                continue

            if (
                irrelevant_input is not None and len(irrelevant_input) > 0
            ) and not self.not_irrelevant_only:
                features += self._get_paired_inputs_features(
                    not_irrelevant_input,
                    irrelevant_input,
                    dataset,
                )

            else:
                features += self._get_fine_tuning_features(
                    not_irrelevant_input, dataset
                )

        return features
