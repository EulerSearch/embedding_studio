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
from embedding_studio.embeddings.features.fine_tuning_features import (
    FineTuningFeatures,
)
from embedding_studio.embeddings.features.fine_tuning_input import (
    FineTuningInput,
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
    """This is a class that represents embeddings fine-tuning logic, designed to be used with
    PytorchLightning Trainer.

    :param model: Embedding model itself, must implement EmbeddingsModelInterface
    :param items_sets: Items dataset related to a given iteration, as a DatasetDict with train and test keys
    :param query_retriever: Object to retrieve items related to queries
    :param loss_func: Loss object for the ranking task
    :param fine_tuning_params: Hyperparameters of the fine-tuning task
    :param tracker: Experiment management object for tracking metrics
    :param metric_calculators: List of trackable metrics calculators. If None, only DistanceShift metric will be used
    :param ranker: Ranking function that takes (query, items) and returns ranks
    :param is_similarity: Whether the ranking function is similarity-based (True) or distance-based (False)
    :param confidence_calculator: Function to calculate result confidences
    :param step_size: Optimizer step size for learning rate scheduler
    :param gamma: Optimizer's gamma parameter for learning rate scheduler
    """

    def __init__(
        self,
        model: EmbeddingsModelInterface,
        items_sets: DatasetDict,
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
        if not isinstance(model, EmbeddingsModelInterface):
            raise TypeError(
                "model must be an instance of EmbeddingsModelInterface"
            )

        if not isinstance(items_sets, DatasetDict):
            raise TypeError("items_sets must be a DatasetDict")

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
        self.items_sets = items_sets
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

    def preprocess_inputs(self, clickstream_dataset: DatasetDict):
        """Preprocesses inputs by calculating ranks for all fine-tuning inputs in the dataset.

        This method ensures that all fine-tuning inputs in the dataset have valid rank values.
        For each input that has either empty ranks or contains None values, it calculates
        proper ranks using the features_extractor. This ensures all inputs are properly
        prepared before the fine-tuning process.

        :param clickstream_dataset: Dataset containing fine-tuning inputs to be preprocessed
        """
        # Process each split in the dataset (train, test, etc.)
        for key in clickstream_dataset.keys():
            # Get the corresponding items set for this split
            items_set = self.items_sets[key]

            # Process not irrelevant examples first (positive examples)
            logger.debug(
                f"Calculate ranks for {key} not irrelevant fine-tuning inputs"
            )
            for fine_tuning_input in clickstream_dataset[key].not_irrelevant:
                # Check if ranks need to be calculated (empty or containing None)
                unique_values = set(fine_tuning_input.ranks.values())
                if len(unique_values) == 0 or None in unique_values:
                    # Calculate ranks using the features extractor
                    fine_tuning_input.ranks = (
                        self.features_extractor.calculate_ranks(
                            fine_tuning_input, items_set, self.query_retriever
                        )
                    )

            # Process irrelevant examples (negative examples)
            logger.debug(
                f"Calculate ranks for {key} irrelevant fine-tuning inputs"
            )
            for fine_tuning_input in clickstream_dataset[key].irrelevant:
                # Check if ranks need to be calculated (empty or containing None)
                unique_values = set(fine_tuning_input.ranks.values())
                if len(unique_values) == 0 or None in unique_values:
                    # Calculate ranks using the features extractor
                    fine_tuning_input.ranks = (
                        self.features_extractor.calculate_ranks(
                            fine_tuning_input, items_set, self.query_retriever
                        )
                    )

    # Standart LightningModule methods to be overrided to be used in PytorchLightning Trainer
    # 1. Configure optimizers and schedulers
    def configure_optimizers(
        self,
    ) -> Tuple[List[Optimizer], List[LRScheduler]]:
        """Configures optimizers and schedulers for the fine-tuning process.

        This is a standard PyTorch Lightning method for setting up optimizers and learning rate schedulers.

        :return: A tuple containing a list of optimizers and a list of schedulers
        """
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
        """Performs a single training step with one batch.

        This is a standard PyTorch Lightning method for defining the training logic.

        :param batch: A list of tuples, where each tuple contains two FineTuningInput objects
        :param batch_idx: The index of the current batch
        :return: The computed loss
        """
        if not (
            isinstance(batch, (list, tuple))
            and all(
                isinstance(fine_tuning_input, tuple)
                and len(fine_tuning_input) == 2
                for fine_tuning_input in batch
            )
        ):
            raise ValueError(
                "batch must be a list or tuple, and each element must be a tuple of two FineTuningInputs."
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
        features: FineTuningFeatures = self.features_extractor.forward(
            batch, self.items_sets["train"]
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
                    self.items_sets["train"],
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
        """Performs a single validation step with one batch.

        This is a standard PyTorch Lightning method for defining the validation logic.

        :param batch: A list of tuples, where each tuple contains two FineTuningInput objects
        :param batch_idx: The index of the current batch
        :return: The computed loss
        """
        if not (
            isinstance(batch, (list, tuple))
            and all(
                isinstance(fine_tuning_input, tuple)
                and len(fine_tuning_input) == 2
                for fine_tuning_input in batch
            )
        ):
            raise ValueError(
                "batch must be a list or tuple, and each element must be a tuple of two FineTuningInputs"
            )

        if isinstance(batch, tuple):
            batch = [
                batch,
            ]

        # TODO: encapsulate all inference
        features: FineTuningFeatures = self.features_extractor.forward(
            batch, self.items_sets["test"]
        )
        loss: FloatTensor = self.loss_func(features)

        # Instead of log test / validation metrics immediately
        # We will accumulate them
        self._validation_metrics["loss"].append(loss.item())

        for calculator in self.calculators:
            for metric in calculator(
                batch,
                self.features_extractor,
                self.items_sets["test"],
                self.query_retriever,
            ):
                self._validation_metrics[metric.name].append(metric.value)

        return loss

    # 4. Aggregation of validation results
    def on_validation_epoch_end(self) -> float:
        """Aggregates validation results at the end of a validation epoch.

        This is a standard PyTorch Lightning method for post-validation processing.

        :return: The mean validation loss
        """
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
        items_sets: DatasetDict,
        query_retriever: QueryRetriever,
        fine_tuning_params: FineTuningParams,
        tracker: ExperimentsManager,
    ):
        """Create embedding fine tuner from settings.

        :param model: Embedding model itself
        :param settings: Fine-tuning settings
        :param items_sets: Items dataset related to a given iteration, as a DatasetDict with train and test keys
        :param query_retriever: Object to retrieve items related to queries
        :param fine_tuning_params: Hyperparameters of the fine-tuning task
        :param tracker: Experiment management object for tracking metrics
        :return: A configured EmbeddingsFineTuner instance
        """
        return EmbeddingsFineTuner(
            model=model,
            items_sets=items_sets,
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
