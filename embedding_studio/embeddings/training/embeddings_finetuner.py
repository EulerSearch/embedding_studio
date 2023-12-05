from collections import defaultdict
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
from datasets import DatasetDict
from torch import FloatTensor, Tensor
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import LRScheduler, StepLR

from embedding_studio.embeddings.data.clickstream.query_retriever import (
    QueryRetriever,
)
from embedding_studio.embeddings.data.clickstream.raw_session import (
    ClickstreamSession,
)
from embedding_studio.embeddings.features.event_confidences import (
    dummy_confidences,
)
from embedding_studio.embeddings.features.extractor import (
    COSINE_SIMILARITY,
    FeaturesExtractor,
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
from embedding_studio.worker.experiments.experiments_tracker import (
    ExperimentsManager,
)
from embedding_studio.worker.experiments.finetuning_params import (
    FineTuningParams,
)
from embedding_studio.worker.experiments.finetuning_settings import (
    FineTuningSettings,
)
from embedding_studio.worker.experiments.metrics_accumulator import MetricValue


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
        :type model: EmbeddingsModelInterface
        :param items_storages:  items storage related to a given session, as a datasetdict with train and test keys
        :type items_storages: DatasetDict
        :param query_retriever: object to get item related to query, that can be used in "forward"
        :type query_retriever: QueryRetriever
        :param loss_func: loss object for a ranking task
        :type loss_func: RankingLossInterface
        :param fine_tuning_params: hyper params of fine-tuning task
        :type fine_tuning_params: FineTuningParams
        :param tracker: experiment management object
        :type tracker: ExperimentsManager
        :param metric_calculators: list of trackable metrics calculators (default: None)
                                   by default only DistanceShift metric
        :type metric_calculators: Optional[List[MetricCalculator]]
        :param ranker: ranking function (query, items) -> ranks (defult: cosine similarity)
        :type ranker: Callable[[FloatTensor, FloatTensor], FloatTensor]
        :param is_similarity: is ranking function similarity like or distance (default: True)
        :type is_similarity: bool
        :param confidence_calculator: function to calculate events confidences (default: dummy_confidences)
        :type confidence_calculator: Callable
        :param step_size: optimizer steps (default: 500)
        :type step_size: int
        :param gamma: optimizers gamma (default: 0.9)
        :type gamma: float
        """
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

    # Standart LightningModule methods to be overrided to be used in PytorchLightning Trainer
    # 1. Configure optimizers and schedulers
    def configure_optimizers(
        self,
    ) -> Tuple[List[Optimizer], List[LRScheduler]]:
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
        batch: List[Tuple[ClickstreamSession, ClickstreamSession]],
        batch_idx: int,
    ) -> Union[FloatTensor, Tensor]:
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
            batch, self.items_storages["train"], self.query_retriever
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
        batch: List[Tuple[ClickstreamSession, ClickstreamSession]],
        batch_idx: int,
    ) -> Union[FloatTensor, Tensor]:
        if isinstance(batch, tuple):
            batch = [
                batch,
            ]

        # TODO: encapsulate all inference
        features: SessionFeatures = self.features_extractor.forward(
            batch, self.items_storages["test"], self.query_retriever
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
        :type model: EmbeddingsModelInterface
        :param settings: fine-tuning settings
        :type settings: FineTuningSettings
        :param items_storages:  items storage related to a given session, as a datasetdict with train and test keys
        :type items_storages: DatasetDict
        :param query_retriever: object to get item related to query, that can be used in "forward"
        :type query_retriever: QueryRetriever
        :param fine_tuning_params: hyper params of fine-tuning task
        :type fine_tuning_params: FineTuningParams
        :param tracker: experiment management object
        :type tracker: ExperimentsManager
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
