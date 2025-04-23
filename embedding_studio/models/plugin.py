import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, FieldValidationInfo, field_validator

from embedding_studio.clickstream_storage.converters.converter import (
    ClickstreamSessionConverter,
)
from embedding_studio.clickstream_storage.query_retriever import QueryRetriever
from embedding_studio.data_storage.loaders.data_loader import DataLoader
from embedding_studio.embeddings.data.clickstream.train_test_splitter import (
    TrainTestSplitter,
)
from embedding_studio.embeddings.data.items.manager import ItemSetManager
from embedding_studio.embeddings.data.ranking_data import RankingData
from embedding_studio.embeddings.data.utils.fields_normalizer import (
    DatasetFieldsNormalizer,
)
from embedding_studio.experiments.experiments_tracker import ExperimentsManager
from embedding_studio.experiments.finetuning_settings import FineTuningSettings
from embedding_studio.experiments.metrics_accumulator import MetricsAccumulator


class PluginMeta(BaseModel):
    """
    Information about a plugin's identity and version.
    This model provides key metadata like the plugin's name, version number,
    and description, helping the system identify and manage different plugins.
    """

    name: str
    version: str = "1.0.0"
    description: Optional[str] = None

    @field_validator("name")
    def validate_name(cls, value: str, info: FieldValidationInfo) -> str:
        # Python identifier regex
        identifier_regex = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
        if not identifier_regex.match(value):
            raise ValueError(
                f"Invalid name '{value}'. Names must start with a letter or underscore, "
                "and can only contain letters, digits, and underscores."
            )
        return value


@dataclass
class FineTuningBuilder:
    """
    A container for all components needed to run a fine-tuning operation.
    This brings together all the specialized tools required for fine-tuning:
    data handling, model training, evaluation metrics, and configuration settings.
    It's like a toolkit that has everything needed for the fine-tuning process.

    :param data_loader: Component responsible for loading data from various sources
        like S3, PostgreSQL, or other storage systems. Provides the content that
        will be embedded during training.

    :param query_retriever: Extracts and formats search queries from user sessions.
        Helps connect user searches with the items they interacted with.

    :param clickstream_sessions_converter: Transforms raw clickstream data into a
        structured format suitable for training. Converts user interactions into
        training examples.

    :param clickstream_sessions_splitter: Divides clickstream data into training and
        testing sets. Ensures model evaluation is done on separate data from what it
        was trained on.

    :param dataset_fields_normalizer: Standardizes field names and formats across
        different data sources. Creates consistency in how data is processed.

    :param items_set_manager: Manages the collection of items used for training.
        Handles splitting, augmentation, and preprocessing of item data.

    :param accumulators: Collectors that track various metrics during the training
        process. Monitor progress and help determine when training is complete.

    :param experiments_manager: Manages experiment tracking, including saving models,
        recording metrics, and comparing different runs. Integrates with MLflow.

    :param fine_tuning_settings: Configuration parameters for the fine-tuning process
        such as loss function, step size, and test frequency.

    :param initial_params: Starting hyperparameters for model training. Used for the
        hyperparameter optimization process when finding optimal settings.

    :param ranking_data: The prepared dataset used for training the model, containing
        query-item pairs with relevance information.

    :param initial_max_evals: Maximum number of evaluations to perform during initial
        hyperparameter optimization. Controls how extensively the system searches for
        optimal parameters.
    """

    data_loader: DataLoader
    query_retriever: QueryRetriever
    clickstream_sessions_converter: ClickstreamSessionConverter
    clickstream_sessions_splitter: TrainTestSplitter
    dataset_fields_normalizer: DatasetFieldsNormalizer
    items_set_manager: ItemSetManager
    accumulators: List[MetricsAccumulator]
    experiments_manager: ExperimentsManager
    fine_tuning_settings: FineTuningSettings
    initial_params: Dict[str, List[Any]]
    ranking_data: RankingData
    initial_max_evals: int = 100
