import os
from typing import List

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from plugins.custom.data_storage.loaders.hf_datasets_query_generator import (
    HugsearchDatasetsQueryGenerator,
)
from plugins.custom.data_storage.loaders.hf_models_query_generator import (
    HugsearchModelsQueryGenerator,
)
from plugins.custom.data_storage.loaders.item_meta import (
    PgsqlItemMetaWithSourceInfo,
)
from plugins.custom.optimizations.indexes import (
    CreateOrderingIndexesOptimization,
)

from embedding_studio.clickstream_storage.converters.converter import (
    ClickstreamSessionConverter,
)
from embedding_studio.clickstream_storage.query_retriever import QueryRetriever
from embedding_studio.clickstream_storage.text_query_retriever import (
    TextQueryRetriever,
)
from embedding_studio.context.app_context import context
from embedding_studio.core.config import settings
from embedding_studio.core.plugin import FineTuningMethod
from embedding_studio.data_storage.loaders.aggregated_data_loader import (
    AggregatedDataLoader,
)
from embedding_studio.data_storage.loaders.data_loader import DataLoader
from embedding_studio.data_storage.loaders.sql.pgsql.pgsql_multi_text_column_loader import (
    PgsqlMultiTextColumnLoader,
)
from embedding_studio.embeddings.augmentations.compose import (
    AugmentationsComposition,
)
from embedding_studio.embeddings.augmentations.items_set_augmentation_applier import (
    ItemsSetAugmentationApplier,
)
from embedding_studio.embeddings.augmentations.text.cases import ChangeCases
from embedding_studio.embeddings.augmentations.text.misspellings import (
    Misspellings,
)
from embedding_studio.embeddings.data.clickstream.train_test_splitter import (
    TrainTestSplitter,
)
from embedding_studio.embeddings.data.items.managers.dict import (
    DictItemSetManager,
)
from embedding_studio.embeddings.data.preprocessors.preprocessor import (
    ItemsDatasetDictPreprocessor,
)
from embedding_studio.embeddings.data.transforms.dict.line_from_dict import (
    get_json_line_from_dict,
)
from embedding_studio.embeddings.data.utils.fields_normalizer import (
    DatasetFieldsNormalizer,
)
from embedding_studio.embeddings.improvement.torch_based_adjuster import (
    TorchBasedAdjuster,
)
from embedding_studio.embeddings.improvement.vectors_adjuster import (
    VectorsAdjuster,
)
from embedding_studio.embeddings.inference.triton.client import (
    TritonClientFactory,
)
from embedding_studio.embeddings.inference.triton.text_to_text.e5 import (
    TextToTextE5TritonClientFactory,
)
from embedding_studio.embeddings.losses.prob_cosine_margin_ranking_loss import (
    CosineProbMarginRankingLoss,
)
from embedding_studio.embeddings.models.text_to_text.e5 import (
    TextToTextE5Model,
)
from embedding_studio.embeddings.splitters.dataset_splitter import (
    ItemsSetSplitter,
)
from embedding_studio.embeddings.splitters.dict.json_splitter import (
    JSONSplitter,
)
from embedding_studio.embeddings.splitters.text.tokenized_grouped_splitter import (
    TokenGroupTextSplitter,
)
from embedding_studio.experiments.experiments_tracker import ExperimentsManager
from embedding_studio.experiments.finetuning_settings import FineTuningSettings
from embedding_studio.experiments.initial_params.clip import INITIAL_PARAMS
from embedding_studio.experiments.metrics_accumulator import MetricsAccumulator
from embedding_studio.models.clickstream.sessions import SessionWithEvents
from embedding_studio.models.embeddings.models import (
    MetricAggregationType,
    MetricType,
    SearchIndexInfo,
)
from embedding_studio.models.plugin import FineTuningBuilder, PluginMeta
from embedding_studio.vectordb.optimization import Optimization
from embedding_studio.workers.fine_tuning.prepare_data import prepare_data


class HFDictTextFineTuningMethod(FineTuningMethod):
    """
    Fine-tuning plugin for Hugging Face models/datasets using dict-style text.

    This is a production-grade plugin used in the demo. It loads data from a
    PostgreSQL database, processes dictionary-structured text fields (like
    'readme', 'tags'), and fine-tunes an E5-based embedding model.

    Key features:
    - Loads from multiple sources (hf_models, hf_datasets).
    - Aggregates multi-column text into a single embedding input.
    - Uses JSONSplitter to respect structure when tokenizing.
    - Supports PostgreSQL env config (HF_POSTGRES_* variables).
    - Fully compatible with Hugging Face model cards and datasets.
    """

    meta = PluginMeta(
        name="HFDictTextFineTuningMethod",  # Should be a python-like naming
        version="0.0.1",
        description="A hugging-face fine-tuning text plugin for dict data",
    )

    def __init__(self):
        """
        Initializes the fine-tuning pipeline with PostgreSQL-backed loaders,
        dict-aware splitting, data augmentation, and experiment tracking.

        Steps:
        1. Load PostgreSQL connection details from environment variables.
        2. Create a multi-source data loader combining HF models and datasets.
        3. Set up item transformation pipeline: normalize, split, augment.
        4. Configure session-to-training-conversion logic.
        5. Initialize experiment tracking and training metrics.
        6. Define search hyperparameters and training configuration.
        """
        # 1. Define model and tokenizer
        self.model_name = "intfloat/multilingual-e5-large"
        self.inference_client_factory = None

        # 2. Load PostgreSQL connection details from env
        HF_POSTGRES_HOST = os.getenv("HF_POSTGRES_HOST")
        HF_POSTGRES_PORT = os.getenv("HF_POSTGRES_PORT")
        HF_POSTGRES_USER = os.getenv("HF_POSTGRES_USER")
        HF_POSTGRES_PASSWORD = os.getenv("HF_POSTGRES_PASSWORD")
        HF_POSTGRES_DB = os.getenv("HF_POSTGRES_DB")

        connection_string = (
            f"postgresql://{HF_POSTGRES_USER}:{HF_POSTGRES_PASSWORD}"
            f"@{HF_POSTGRES_HOST}:{HF_POSTGRES_PORT}/{HF_POSTGRES_DB}"
        )

        # 3. Load data from multiple tables using field-level text merging
        self.data_loader = AggregatedDataLoader(
            {
                "hf_models": PgsqlMultiTextColumnLoader(
                    connection_string=connection_string,
                    query_generator=HugsearchModelsQueryGenerator,
                    text_columns=["id", "readme", "tags"],
                ),
                "hf_datasets": PgsqlMultiTextColumnLoader(
                    connection_string=connection_string,
                    query_generator=HugsearchDatasetsQueryGenerator,
                    text_columns=["id", "readme", "tags"],
                ),
            },
            item_meta_cls=PgsqlItemMetaWithSourceInfo,
        )

        # 4. Fields from each table to be encoded into a single string
        self.field_names = ["id", "readme", "tags"]

        # 5. Retrieve user text queries (from clickstream)
        self.retriever = TextQueryRetriever()

        # 6. Convert clickstream events into training data
        self.sessions_converter = ClickstreamSessionConverter(
            item_type=PgsqlItemMetaWithSourceInfo
        )

        # 7. Split clickstream into train/test sets
        self.splitter = TrainTestSplitter()

        # 8. Normalize raw data structure for consistency
        self.normalizer = DatasetFieldsNormalizer("item", "item_id")

        # 9. Configure the item splitter to tokenize structured dicts
        self.items_splitter = TokenGroupTextSplitter(
            tokenizer=AutoTokenizer.from_pretrained(self.model_name),
            blocks_splitter=JSONSplitter(field_names=self.field_names),
            max_tokens=512,
        )

        # 10. Preprocess items: split, augment, format
        self.items_set_manager = DictItemSetManager(
            self.normalizer,
            items_set_splitter=ItemsSetSplitter(self.items_splitter),
            augmenter=ItemsSetAugmentationApplier(
                AugmentationsComposition(
                    [
                        ChangeCases(5),
                        Misspellings(5),
                    ]
                )
            ),
            do_augment_test=False,
            transform=get_json_line_from_dict,
        )

        # 11. Define metrics for tracking training/testing behavior
        self.accumulators = [
            MetricsAccumulator(
                "train_loss",
                calc_mean=True,
                calc_sliding=True,
                calc_min=True,
                calc_max=True,
            ),
            MetricsAccumulator(
                "train_not_irrelevant_dist_shift",
                calc_mean=True,
                calc_sliding=True,
                calc_min=True,
                calc_max=True,
            ),
            MetricsAccumulator(
                "train_irrelevant_dist_shift",
                calc_mean=True,
                calc_sliding=True,
                calc_min=True,
                calc_max=True,
            ),
            MetricsAccumulator("test_loss"),
            MetricsAccumulator("test_not_irrelevant_dist_shift"),
            MetricsAccumulator("test_irrelevant_dist_shift"),
        ]

        # 12. Connect to experiment tracker (e.g. MLflow)
        self.manager = ExperimentsManager.from_wrapper(
            wrapper=context.mlflow_client,
            main_metric="test_not_irrelevant_dist_shift",
            plugin_name=self.meta.name,
            accumulators=self.accumulators,
        )

        # 13. Define search grid for hyperparameter exploration
        self.initial_params = INITIAL_PARAMS
        self.initial_params.update(
            {
                "not_irrelevant_only": [True],
                "negative_downsampling": [0.5],
                "examples_order": [[11]],
            }
        )

        # 14. Define actual training loop config
        self.settings = FineTuningSettings(
            loss_func=CosineProbMarginRankingLoss(),
            step_size=35,
            test_each_n_sessions=0.5,
            num_epochs=3,
        )

    def upload_initial_model(self) -> None:
        """
        Downloads the E5 model and registers it as the base model
        for future training sessions.
        """
        model = context.model_downloader.download_model(
            model_name=self.model_name,
            download_fn=lambda mn: SentenceTransformer(mn),
        )
        model = TextToTextE5Model(model)
        self.manager.upload_initial_model(model)

    def get_items_preprocessor(self) -> ItemsDatasetDictPreprocessor:
        """
        Returns the dict-based preprocessor that combines fields into a
        structured JSONL line, with splitting and augmentation applied.
        """
        return self.items_set_manager.preprocessor

    def get_query_retriever(self) -> QueryRetriever:
        """
        Returns the retriever that pulls text queries from sessions.
        """
        return self.retriever

    def get_data_loader(self) -> DataLoader:
        """
        Returns the AggregatedDataLoader that merges HF models and datasets.
        """
        return self.data_loader

    def get_manager(self) -> ExperimentsManager:
        """
        Returns the MLflow-compatible manager that logs runs, metrics, and config.
        """
        return self.manager

    def get_items_splitter(self) -> TokenGroupTextSplitter:
        """
        Returns the JSONSplitter-powered splitter that respects dict structure
        and ensures max token length is not exceeded.
        """
        return self.items_splitter

    def get_inference_client_factory(self) -> TritonClientFactory:
        """
        Returns the client factory to perform model inference on Triton,
        using E5 model and preprocessor defined above.
        """
        if self.inference_client_factory is None:
            self.inference_client_factory = TextToTextE5TritonClientFactory(
                f"{settings.INFERENCE_HOST}:{settings.INFERENCE_GRPC_PORT}",
                plugin_name=self.meta.name,
                preprocessor=self.items_set_manager.preprocessor,
                model_name=self.model_name,
            )

        return self.inference_client_factory

    def get_fine_tuning_builder(
        self, clickstream: List[SessionWithEvents]
    ) -> FineTuningBuilder:
        """
        Prepares the ranking dataset from clickstream sessions and returns a
        builder that encapsulates the full fine-tuning config.

        :param clickstream: List of user sessions with clicks and queries.
        :return: Configured FineTuningBuilder instance.
        """
        ranking_dataset = prepare_data(
            clickstream,
            self.sessions_converter,
            self.splitter,
            self.retriever,
            self.data_loader,
            self.items_set_manager,
        )
        fine_tuning_builder = FineTuningBuilder(
            data_loader=self.data_loader,
            query_retriever=self.retriever,
            clickstream_sessions_converter=self.sessions_converter,
            clickstream_sessions_splitter=self.splitter,
            dataset_fields_normalizer=self.normalizer,
            items_set_manager=self.items_set_manager,
            accumulators=self.accumulators,
            experiments_manager=self.manager,
            fine_tuning_settings=self.settings,
            initial_params=self.initial_params,
            ranking_data=ranking_dataset,
            initial_max_evals=2,
        )
        return fine_tuning_builder

    def get_search_index_info(self) -> SearchIndexInfo:
        """
        Returns the search index config: 1024D cosine-min aggregation.
        """
        return SearchIndexInfo(
            dimensions=1024,
            # dimensions=384,
            metric_type=MetricType.COSINE,
            metric_aggregation_type=MetricAggregationType.MIN,
        )

    def get_vectors_adjuster(self) -> VectorsAdjuster:
        """
        Returns a vector adjuster that applies minor updates to
        embeddings post-training based on feedback signals.
        """
        return TorchBasedAdjuster(
            adjustment_rate=1e-3,
            search_index_info=self.get_search_index_info(),
        )

    def get_vectordb_optimizations(self) -> List[Optimization]:
        """
        Returns a list of vector DB optimizations to apply.
        In this case: index ordering by similarity or freshness.
        """
        return [CreateOrderingIndexesOptimization()]
