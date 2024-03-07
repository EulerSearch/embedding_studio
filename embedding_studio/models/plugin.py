from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from embedding_studio.clickstream_storage.parsers import ClickstreamParser
from embedding_studio.clickstream_storage.query_retriever import QueryRetriever
from embedding_studio.data_storage.loaders.data_loader import DataLoader
from embedding_studio.embeddings.data.clickstream.splitter import (
    ClickstreamSessionsSplitter,
)
from embedding_studio.embeddings.data.ranking_data import RankingData
from embedding_studio.embeddings.data.storages.producer import (
    ItemStorageProducer,
)
from embedding_studio.embeddings.data.utils.fields_normalizer import (
    DatasetFieldsNormalizer,
)
from embedding_studio.workers.fine_tuning.experiments.experiments_tracker import (
    ExperimentsManager,
)
from embedding_studio.workers.fine_tuning.experiments.finetuning_settings import (
    FineTuningSettings,
)
from embedding_studio.workers.fine_tuning.experiments.metrics_accumulator import (
    MetricsAccumulator,
)


class PluginMeta(BaseModel):
    name: str
    version: str = "1.0.0"
    description: Optional[str] = None


@dataclass
class FineTuningBuilder:
    data_loader: DataLoader
    query_retriever: QueryRetriever
    clickstream_parser: ClickstreamParser
    clickstream_sessions_splitter: ClickstreamSessionsSplitter
    dataset_fields_normalizer: DatasetFieldsNormalizer
    item_storage_producer: ItemStorageProducer
    accumulators: List[MetricsAccumulator]
    experiments_manager: ExperimentsManager
    fine_tuning_settings: FineTuningSettings
    initial_params: Dict[str, List[Any]]
    ranking_data: RankingData
    initial_max_evals: int = 100
