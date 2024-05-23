import logging
from collections import defaultdict
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
from datasets import DatasetDict
from torch import FloatTensor, Tensor
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import LRScheduler, StepLR

from embedding_studio.clickstream_storage.query_retriever import QueryRetriever
from embedding_studio.embeddings.features.event_confidences import (
    dummy_confidences,
)
from embedding_studio.embeddings.features.extractor import (
    COSINE_SIMILARITY,
    FeaturesExtractor,
)
from embedding_studio.embeddings.features.feature_extractor_input import (
    FineTuningInput,
)
from embedding_studio.embeddings.features.session_features import (
    SessionFeatures,
)
from embedding_studio.embeddings.losses.ranking_loss_interface import (
    RankingLossInterface,
)
from embedding_studio.embeddings.metrics.distance_shift import DistanceShift
from embedding_studio.embeddings.metrics.metric import MetricCalculator
from embedding_studio.embeddings.models.interface import (
    EmbeddingsModelInterface,
)
from embedding_studio.experiments.experiments_tracker import ExperimentsManager
from embedding_studio.experiments.finetuning_params import FineTuningParams
from embedding_studio.experiments.finetuning_settings import FineTuningSettings
from embedding_studio.experiments.metrics_accumulator import MetricValue

logger = logging.getLogger(__name__)


class EmbeddingsFineTuner(pl.LightningModule):
    def __init__(
        self,
        model: EmbeddingsModelInterface,
        items_storages: DatasetDict,
        query_retriever: QueryRetriever,
        loss_func: RankingLossInterface,
        fine_tuning_params: FineTuningParams,
        tracker: ExperimentsManager,
        metric_calculators: Optional[List[MetricCalculator]] = None,
        ranker: Callable[
            [FloatTensor, FloatTensor], FloatTensor
        ] = COSINE_SIMILARITY,
        is_similarity: bool = True,
        confidence_calculator: Callable = dummy_confidences,
        step_size: int = 500,
        gamma: float = 0.9,
    ):
        """This is a class, that represents embeddings fine-tuning logic,
        designed in the way to be use PytorchLightning Trainer.

        :param model: embedding model itself
        :param items_storages:  items storage related to a given iteration, as a datasetdict with train and test keys
        :param query_retriever: object to get item related to query, that can be used in "forward"
        :param loss_func: loss object for a ranking task
        :param fine_tuning_params: hyper params of fine-tuning task
        :param tracker: experiment management object
        :param metric_calculators: list of trackable metrics calculators (default: None)
                                   by default_params only DistanceShift metric
        :param ranker: ranking function (query, items) -> ranks (defult: cosine similarity)
        :param is_similarity: is ranking function similarity like or distance (default: True)
        :param confidence_calculator: function to calculate results confidences (default: dummy_confidences)
        :param step_size: optimizer steps (default: 500)
        :param gamma: optimizers gamma (default: 0.9)
        """
        if not isinstance(model, EmbeddingsModelInterface):
            raise TypeError(
                "model must be an instance of EmbeddingsModelInterface"
            )

        if not isinstance(items_storages, DatasetDict):
            raise TypeError("items_storages must be a DatasetDict")

        if not isinstance(query_retriever, QueryRetriever):
            raise TypeError(
                "query_retriever must be an instance of QueryRetriever"
            )

        if not isinstance(loss_func, RankingLossInterface):
            raise TypeError(
                "loss_func must be an instance of RankingLossInterface"
            )

        if not isinstance(fine_tuning_params, FineTuningParams):
            raise TypeError(
                "fine_tuning_params must be an instance of FineTuningParams"
            )

        if not isinstance(tracker, ExperimentsManager):
            raise TypeError(
                "tracker must be an instance of ExperimentsManager"
            )

        if not isinstance(fine_tuning_params, FineTuningParams):
            raise TypeError(
                "fine_tuning_params must be an instance of FineTuningParams"
            )

        super(EmbeddingsFineTuner, self).__init__()
        self.features_extractor = FeaturesExtractor(
            model,
            ranker,
            is_similarity,
            fine_tuning_params.not_irrelevant_only,
            fine_tuning_params.negative_downsampling,
            fine_tuning_params.min_abs_difference_threshold,
            fine_tuning_params.max_abs_difference_threshold,
            confidence_calculator,
        )
        self.items_storages = items_storages
        self.query_retriever = query_retriever

        if not metric_calculators:
            logger.debug(
                "metric_calculators list is empty - DistanceShift metric will be used by default."
            )
        self.calculators = (
            metric_calculators
            if metric_calculators is not None
            else [DistanceShift()]
        )

        self.loss_func = loss_func
        self.loss_func.set_margin(fine_tuning_params.margin)
        self.fine_tuning_params = fine_tuning_params
        self.tracker = tracker
        self.step_size = step_size
        self.gamma = gamma
        self._validation_metrics = defaultdict(list)

        # Fix layers
        self.features_extractor.model.fix_item_model(
            fine_tuning_params.num_fixed_layers
        )
        self.features_extractor.model.fix_query_model(
            fine_tuning_params.num_fixed_layers
        )

        self.automatic_optimization = False

    def preprocess_sessions(self, clickstream_dataset: DatasetDict):
        for key in clickstream_dataset.keys():
            item_storage = self.items_storages[key]
            logger.info(
                f"Calculate ranks for {key} not irrelevant clickstream sessions"
            )
            for session in clickstream_dataset[key].not_irrelevant:
                unique_values = set(session.ranks.values())
                if len(unique_values) == 0 or None in unique_values:
                    session.ranks = self.features_extractor.calculate_ranks(
                        session, item_storage, self.query_retriever
                    )

            logger.info(
                f"Calculate ranks for {key} irrelevant clickstream sessions"
            )
            for session in clickstream_dataset[key].irrelevant:
                unique_values = set(session.ranks.values())
                if len(unique_values) == 0 or None in unique_values:
                    session.ranks = self.features_extractor.calculate_ranks(
                        session, item_storage, self.query_retriever
                    )

    # Standart LightningModule methods to be overrided to be used in PytorchLightning Trainer
    # 1. Configure optimizers and schedulers
    def configure_optimizers(
        self,
    ) -> Tuple[List[Optimizer], List[LRScheduler]]:
        if not (isinstance(self.step_size, int) and self.step_size > 0):
            raise ValueError("step_size must be a positive integer")

        if not (isinstance(self.gamma, float) and 0 < self.gamma < 1):
            raise ValueError("gamma must be a float in the range (0, 1)")

        items_optimizer: SGD = SGD(
            self.features_extractor.model.get_items_model_params(),
            lr=self.fine_tuning_params.items_lr,
            weight_decay=self.fine_tuning_params.items_weight_decay,
        )
        items_scheduler: StepLR = StepLR(
            items_optimizer, step_size=self.step_size, gamma=self.gamma
        )

        if self.features_extractor.model.same_query_and_items:
            return [items_optimizer], [items_scheduler]

        query_optimizer: SGD = SGD(
            self.features_extractor.model.get_query_model_params(),
            lr=self.fine_tuning_params.query_lr,
            weight_decay=self.fine_tuning_params.query_weight_decay,
        )
        query_scheduler: StepLR = torch.optim.lr_scheduler.StepLR(
            query_optimizer, step_size=self.step_size, gamma=self.gamma
        )

        return [items_optimizer, query_optimizer], [
            items_scheduler,
            query_scheduler,
        ]

    # 2. Training step code with one batch
    def training_step(
        self,
        batch: List[Tuple[FineTuningInput, FineTuningInput]],
        batch_idx: int,
    ) -> Union[FloatTensor, Tensor]:
        if not (
            isinstance(batch, (list, tuple))
            and all(
                isinstance(session, tuple) and len(session) == 2
                for session in batch
            )
        ):
            raise ValueError(
                "batch must be a list or tuple, and each element must be a tuple of two ClickstreamSessions"
            )

        if isinstance(batch, tuple):
            batch = [
                batch,
            ]

        # Get current optimizers
        query_optimizer: Optional[Optimizer] = None
        if self.features_extractor.model.same_query_and_items:
            items_optimizer: Optimizer = self.optimizers()
        else:
            items_optimizer, query_optimizer = self.optimizers()

        # Reset the gradients of all optimized
        items_optimizer.zero_grad()
        if query_optimizer:
            query_optimizer.zero_grad()

        # Calculate features and loss
        # TODO: encapsulate all inference
        features: SessionFeatures = self.features_extractor.forward(
            batch, self.items_storages["train"]
        )
        loss: FloatTensor = self.loss_func(features)
        # Gradient backward step
        loss.backward()

        # Log train loss
        self.tracker.save_metric(MetricValue("train_loss", loss.item()))

        # Do a gradient step
        items_optimizer.step()
        if query_optimizer:
            query_optimizer.step()

        with torch.no_grad():
            # And calculate metrics
            for calculator in self.calculators:
                for metric in calculator(
                    batch,
                    self.features_extractor,
                    self.items_storages["train"],
                    self.query_retriever,
                ):
                    self.tracker.save_metric(metric.add_prefix("train"))

        return loss

    # 3. Validation step code with one batch
    @torch.no_grad()
    def validation_step(
        self,
        batch: List[Tuple[FineTuningInput, FineTuningInput]],
        batch_idx: int,
    ) -> Union[FloatTensor, Tensor]:
        if not (
            isinstance(batch, (list, tuple))
            and all(
                isinstance(session, tuple) and len(session) == 2
                for session in batch
            )
        ):
            raise ValueError(
                "batch must be a list or tuple, and each element must be a tuple of two ClickstreamSessions"
            )

        if isinstance(batch, tuple):
            batch = [
                batch,
            ]

        # TODO: encapsulate all inference
        features: SessionFeatures = self.features_extractor.forward(
            batch, self.items_storages["test"]
        )
        loss: FloatTensor = self.loss_func(features)

        # Instead of log test / validation metrics immediately
        # We will accumulate them
        self._validation_metrics["loss"].append(loss.item())

        for calculator in self.calculators:
            for metric in calculator(
                batch,
                self.features_extractor,
                self.items_storages["test"],
                self.query_retriever,
            ):
                self._validation_metrics[metric.name].append(metric.value)

        return loss

    # 4. Aggregation of validation results
    def on_validation_epoch_end(self) -> float:
        loss: Optional[float] = None
        # And log only averages at the end of validation epoch
        for name, values in self._validation_metrics.items():
            mean_value = float(np.mean(values))
            if name == "loss":
                loss = mean_value
            self.tracker.save_metric(
                MetricValue(name, mean_value).add_prefix("test")
            )

        self._validation_metrics = defaultdict(list)

        return loss

    @staticmethod
    def create(
        model: EmbeddingsModelInterface,
        settings: FineTuningSettings,
        items_storages: DatasetDict,
        query_retriever: QueryRetriever,
        fine_tuning_params: FineTuningParams,
        tracker: ExperimentsManager,
    ):
        """Create embedding fine tuner from settings.

        :param model: embedding model itself
        :param settings: fine-tuning settings
        :param items_storages:  items storage related to a given iteration, as a datasetdict with train and test keys
        :param query_retriever: object to get item related to query, that can be used in "forward"
        :param fine_tuning_params: hyper params of fine-tuning task
        :param tracker: experiment management object
        :return:
        """
        return EmbeddingsFineTuner(
            model=model,
            items_storages=items_storages,
            query_retriever=query_retriever,
            loss_func=settings.loss_func,
            fine_tuning_params=fine_tuning_params,
            tracker=tracker,
            metric_calculators=settings.metric_calculators,
            ranker=settings.ranker,
            is_similarity=settings.is_similarity,
            confidence_calculator=settings.confidence_calculator,
            step_size=settings.step_size,
            gamma=settings.gamma,
        )
