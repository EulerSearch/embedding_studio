import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, FieldValidationInfo, field_validator

from embedding_studio.clickstream_storage.parsers.parser import (
    ClickstreamParser,
)
from embedding_studio.clickstream_storage.query_retriever import QueryRetriever
from embedding_studio.data_storage.loaders.data_loader import DataLoader
from embedding_studio.embeddings.data.clickstream.train_test_splitter import (
    TrainTestSplitter,
)
from embedding_studio.embeddings.data.ranking_data import RankingData
from embedding_studio.embeddings.data.storages.producer import (
    ItemStorageProducer,
)
from embedding_studio.embeddings.data.utils.fields_normalizer import (
    DatasetFieldsNormalizer,
)
from embedding_studio.experiments.experiments_tracker import ExperimentsManager
from embedding_studio.experiments.finetuning_settings import FineTuningSettings
from embedding_studio.experiments.metrics_accumulator import MetricsAccumulator


class PluginMeta(BaseModel):
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
    data_loader: DataLoader
    query_retriever: QueryRetriever
    clickstream_parser: ClickstreamParser
    clickstream_sessions_splitter: TrainTestSplitter
    dataset_fields_normalizer: DatasetFieldsNormalizer
    item_storage_producer: ItemStorageProducer
    accumulators: List[MetricsAccumulator]
    experiments_manager: ExperimentsManager
    fine_tuning_settings: FineTuningSettings
    initial_params: Dict[str, List[Any]]
    ranking_data: RankingData
    initial_max_evals: int = 100
