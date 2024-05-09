from typing import Callable, List, Optional, Union

from pydantic import BaseModel
from torch import FloatTensor

from embedding_studio.embeddings.features.event_confidences import (
    dummy_confidences,
)
from embedding_studio.embeddings.features.extractor import COSINE_SIMILARITY
from embedding_studio.embeddings.losses.ranking_loss_interface import (
    RankingLossInterface,
)
from embedding_studio.embeddings.metrics.metric import MetricCalculator


class FineTuningSettings(BaseModel):
    """

    :param loss_func: loss object for a ranking task
    :param metric_calculators: list of trackable metrics calculators (default: None)
                           by default only DistanceShift metric
    :param ranker: ranking function (query, items) -> ranks (defult: cosine similarity)
    :param is_similarity: is ranking function similarity like or distance (default: True)
    :param confidence_calculator: function to calculate results confidences (default: dummy_confidences)
    :param step_size: optimizer steps (default: 500)
    :param gamma: optimizers gamma (default: 0.9)
    :param num_epochs: num of training epochs (default: 10)
    :param batch_size: count of sessions in a batch (default: 1)
    :param test_each_n_sessions: frequency of validation, if value in range [0, 1] - used as ratio (default: -1)
    """

    loss_func: RankingLossInterface
    metric_calculators: Optional[List[MetricCalculator]] = None
    ranker: Optional[
        Callable[[FloatTensor, FloatTensor], FloatTensor]
    ] = COSINE_SIMILARITY
    is_similarity: Optional[bool] = True
    confidence_calculator: Optional[Callable] = dummy_confidences
    step_size: Optional[int] = 500
    gamma: Optional[float] = 0.9
    num_epochs: Optional[int] = 10
    batch_size: Optional[int] = 1
    test_each_n_sessions: Optional[Union[float, int]] = -1

    class Config:
        arbitrary_types_allowed = True
