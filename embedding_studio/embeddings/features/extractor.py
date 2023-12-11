import logging
import random
from typing import Callable, List, Optional, Tuple

import pytorch_lightning as pl
import torch
from torch import FloatTensor, Tensor

from embedding_studio.embeddings.data.clickstream.query_retriever import (
    QueryRetriever,
)
from embedding_studio.embeddings.data.clickstream.raw_session import (
    ClickstreamSession,
)
from embedding_studio.embeddings.data.storages.storage import ItemsStorage
from embedding_studio.embeddings.features.event_confidences import (
    dummy_confidences,
)
from embedding_studio.embeddings.features.session_features import (
    SessionFeatures,
)
from embedding_studio.embeddings.models.interface import (
    EmbeddingsModelInterface,
)
from embedding_studio.worker.experiments.finetuning_params import ExamplesType

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
        exmaples_order: Optional[List[ExamplesType]] = None,
    ):
        """Logic of extracting features:
        1. Positive and negative examples ranks
        2. Event confidences
        3. Target: 1 if is_similarity else -1

        and pack it in SessionFeatures object.

        :param model: embedding model itself
        :type model: EmbeddingsModelInterface
        :param ranker: ranking function (query, items) -> ranks (defult: cosine similarity)
        :type ranker: Optional[Callable[[FloatTensor, FloatTensor], FloatTensor]]
        :param is_similarity: is ranking function similarity like or distance (default: True)
        :type is_similarity: Optional[bool]
        :param not_irrelevant_only: use only not irrelevant sessions (default: True)
                                    True - Triplet loss
                                    False - Contrastive-like loss
        :type not_irrelevant_only: Optional[bool]
        :param negative_downsampling_factor: in real tasks amount of results is much larger than  not-results,
                                             use this parameters to fix a balance (default: 0.5)
        :type negative_downsampling_factor: Optional[float]
        :param min_abs_difference_threshold: filter out soft pairs abs(neg_dist - pos_dist) < small value
        :type min_abs_difference_threshold: float
        :param max_abs_difference_threshold: filter out hard pairs abs(neg_dist - pos_dist) > huge value
        :type max_abs_difference_threshold: float
        :param confidence_calculator: function to calculate results confidences (default: dummy_confidences)
        :type confidence_calculator: Optional[Callable]
        :param examples_order: order of passing examples to a trainer (default: None)
        :type examples_order: Optional[List[ExamplesType]]
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
        self.negative_donwsampling_factor = negative_downsampling_factor

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

        if not exmaples_order:
            exmaples_order = [ExamplesType.all_examples]
            logger.debug("All types of examples will be used in training")

        if len({isinstance(e, ExamplesType) for e in exmaples_order}) > 1:
            raise ValueError(
                "Some of exmaple types are not instances of ExampleType"
            )
        self.exmaples_order = (
            exmaples_order  # TODO: use different examples order
        )

    def _confidences(
        self, session: ClickstreamSession, not_events: List[str]
    ) -> Tuple[Tensor, Tensor]:
        """Calculate confidences for a given clickstream session items.

        :param session: provided clickstream session
        :type session: ClickstreamSession
        :param not_events: not-results (negatives) used for ranks prediction
        :type not_events: List[str]
        :return: positive (results) confidences, negative (not-results) confidences
        :rtype: Tuple[Tensor, Tensor]
        """
        only_used: List[bool] = [
            (id_ in session.events or id_ in not_events)
            for id_ in session.results
        ]
        only_used_ids: List[str] = [
            id_
            for id_ in session.results
            if (id_ in session.events or id_ in not_events)
        ]
        ranks: FloatTensor = FloatTensor(
            [session.ranks[i] for i in session.results]
        )
        bin_clicks: FloatTensor = FloatTensor(
            [(1 if i in session.events else 0) for i in session.results]
        )
        confidences: FloatTensor = self.confidence_calculator(
            ranks, bin_clicks
        )[only_used]

        # Sort confidences among positive and negative types
        positive_confidences: Tensor = torch.zeros(len(session.events))
        negative_confidences: Tensor = torch.zeros(len(not_events))

        for id_index, id_ in enumerate(only_used_ids):
            if id_ in session.events:
                positive_confidences[session.events.index(id_)] = confidences[
                    id_index
                ]

            elif id_ in not_events:
                negative_confidences[not_events.index(id_)] = confidences[
                    id_index
                ]

        return positive_confidences.to(self.device), negative_confidences.to(
            self.device
        )

    def _get_session_features(
        self,
        session: ClickstreamSession,
        dataset: ItemsStorage,
        query_retriever: QueryRetriever,
    ) -> SessionFeatures:
        """Calculate features for a single session

        :param session: given session
        :type session: ClickstreamSession
        :param dataset: items storage related to a given session
        :type dataset: ItemsStorage
        :param query_retriever: object to get item related to query, that can be used in "forward"
        :type query_retriever: QueryRetriever
        :return: provided session's features
        :rtype: SessionFeatures
        """
        features = SessionFeatures()

        # For keep balance between results and not-results, we decrease a number of not-results
        not_events_count: int = int(
            self.negative_donwsampling_factor * len(session.not_events)
        )
        not_events_indexes: List[int] = random.choices(
            list(range(len(session.not_events))), k=not_events_count
        )  # we use indexes instead of ids to keep order
        not_events: List[str] = [
            session.not_events[i] for i in sorted(not_events_indexes)
        ]

        # And calculate confidences for two groups of items
        (
            features.positive_confidences,
            features.negative_confidences,
        ) = self._confidences(session, not_events)

        # Then we calculate query and items vectors
        query_vector: FloatTensor = self.model.forward_query(
            query_retriever(session.query)
        )
        items_vectors: FloatTensor = self.model.forward_items(
            dataset.items_by_ids(session.events + not_events)
        )

        positive_indexes: List[int] = [i for i in range(len(session.events))]
        negative_indexes: List[int] = [
            i + len(session.events) for i in range(len(not_events))
        ]

        # For each group we calculate ranks
        positive_ranks_: FloatTensor = self.ranker(
            query_vector, items_vectors[positive_indexes]
        )
        negative_ranks_: FloatTensor = self.ranker(
            query_vector, items_vectors[negative_indexes]
        )

        if len(positive_indexes) > 0:
            positive_idx = []
            negatie_idx = []
            for pos_i_ in positive_indexes:
                for neg_i_ in negative_indexes:
                    pos_i = pos_i_
                    neg_i = neg_i_ - len(session.events)
                    positive_idx.append(pos_i)
                    negatie_idx.append(neg_i)

            features.positive_ranks = positive_ranks_[positive_idx]
            features.negative_ranks = negative_ranks_[negatie_idx]

            features.positive_confidences = features.positive_confidences[
                positive_idx
            ]
            features.negative_confidences = features.negative_confidences[
                negatie_idx
            ]

        else:

            features.negative_distances = negative_ranks_

        target_value: int = 1 if self.is_similarity else -1
        features.target = target_value * torch.ones(
            features.negative_confidences.shape[0]
        ).to(self.device)

        # Filter out noises
        features.clamp_diff_in(
            self.min_abs_difference_threshold,
            self.max_abs_difference_threshold,
        )

        return features

    def _get_paired_sessions_features(
        self,
        not_irrelevant_session: ClickstreamSession,
        irrelevant_session: ClickstreamSession,
        dataset: ItemsStorage,
        query_retriever: QueryRetriever,
    ) -> SessionFeatures:
        """Calculate features for a given pair: irrelevant and not irrelevant sessions

        :param not_irrelevant_session: not-irrelevant session
        :type not_irrelevant_session: ClickstreamSession
        :param irrelevant_session: irrelevant session
        :type irrelevant_session: ClickstreamSession
        :param dataset: storage of items related to clickstream sessions
        :type dataset: ItemsStorage
        :param query_retriever: object to get item related to query, that can be used in "forward"
        :type query_retriever: QueryRetriever
        :return: features related for both irrelevant and not irrelevant sessions
        :rtype: SessionFeatures
        """
        not_irrelevant_features: SessionFeatures = self._get_session_features(
            not_irrelevant_session, dataset, query_retriever
        )
        irrelevant_features: SessionFeatures = self._get_session_features(
            irrelevant_session, dataset, query_retriever
        )

        irrelevant_features.use_positive_from(not_irrelevant_features)

        not_irrelevant_features += irrelevant_features

        return not_irrelevant_features

    def forward(
        self,
        batch: List[Tuple[ClickstreamSession, ClickstreamSession]],
        dataset: ItemsStorage,
        query_retriever: QueryRetriever,
    ) -> SessionFeatures:
        """Calculate features for a given batch of pairs: irrelevant and not irrelevant sessions

        :param batch: list of pairs: irrelevant and not irrelevant sessions
        :type batch: List[Tuple[ClickstreamSession, ClickstreamSession]]
        :param dataset:  storage of items related to clickstream sessions
        :type dataset: ItemsStorage
        :param query_retriever: object to get item related to query, that can be used in "forward"
        :type query_retriever: QueryRetriever
        :return: session features related to a given batch
        :rtype: SessionFeatures
        """
        features = SessionFeatures()

        for not_irrelevant_session, irrelevant_session in batch:
            if len(not_irrelevant_session.events) == 0:
                logger.warning("Not irrelevant session has no results")
                continue

            if (
                irrelevant_session is not None and len(irrelevant_session) > 0
            ) and not self.not_irrelevant_only:
                features += self._get_paired_sessions_features(
                    not_irrelevant_session,
                    irrelevant_session,
                    dataset,
                    query_retriever,
                )

            else:
                features += self._get_session_features(
                    not_irrelevant_session, dataset, query_retriever
                )

        return features
