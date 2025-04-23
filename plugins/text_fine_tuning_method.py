from typing import List

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

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
from embedding_studio.data_storage.loaders.cloud_storage.gcp.gcp_text_loader import (
    GCPTextLoader,
)
from embedding_studio.data_storage.loaders.cloud_storage.gcp.item_meta import (
    GCPFileMeta,
)
from embedding_studio.data_storage.loaders.data_loader import DataLoader
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
from embedding_studio.embeddings.data.items.managers.text import (
    TextItemSetManager,
)
from embedding_studio.embeddings.data.preprocessors.preprocessor import (
    ItemsDatasetDictPreprocessor,
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
from embedding_studio.embeddings.splitters.item_splitter import ItemSplitter
from embedding_studio.embeddings.splitters.text.dummy_sentence_splitter import (
    DummySentenceSplitter,
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


class DefaultTextFineTuningMethod(FineTuningMethod):
    """
    Plugin for fine-tuning models on plain text using a GCP-based loader.

    This plugin processes individual text fields (e.g. articles, reviews) using
    sentence-level splitting and E5-based embedding models. It provides all
    required components to integrate with the fine-tuning framework.

    How to build your own:
    1. Subclass FineTuningMethod.
    2. Define the `meta` attribute with name, version, description.
    3. Implement all required methods: loaders, preprocessors, trainer, etc.
    4. Register via PluginManager and reference it in config.
    """

    meta = PluginMeta(
        name="TextDefaultFineTuningMethodForText",  # Should be a python-like naming
        version="0.0.1",
        description="A default fine-tuning text plugin",
    )

    def __init__(self):
        """
        Step-by-step setup of a plugin for plain text fine-tuning.

        Steps:
        1. Define the embedding model and tokenizer names.
        2. Initialize GCP text data loader (anonymous or with credentials).
        3. Set up query retriever to extract search terms from user sessions.
        4. Convert clickstream sessions into training labels.
        5. Split sessions into train/test using a basic splitter.
        6. Normalize item fields to unify across loaders.
        7. Configure sentence-based splitting and augmentation strategies.
        8. Track training/eval metrics using accumulators.
        9. Setup the experiment tracker (MLflow, etc.).
        10. Define hyperparameter grid for tuning experiments.
        11. Set training config: loss function, epoch count, etc.
        """
        # uncomment and pass your credentials to use your own gcp bucket
        # creds = {
        #     "credentials_path": "./etc/your-gcp-credentials.json",
        #     "use_system_info": False
        # }

        # 1. Set base model for encoding and tokenizer
        self.model_name = "intfloat/multilingual-e5-base"
        self.inference_client_factory = None

        # 2. GCP loader setup (anonymous or pass credentials path)
        creds = {"use_system_info": True}
        self.data_loader = GCPTextLoader(**creds)

        # 3. Retrieve search queries from clickstream
        self.retriever = TextQueryRetriever()

        # 4. Convert user feedback into training data
        self.sessions_converter = ClickstreamSessionConverter(
            item_type=GCPFileMeta
        )

        # 5. Split sessions into train/test groups
        self.splitter = TrainTestSplitter()

        # 6. Normalize data fields (ensure "item" and "item_id" fields exist)
        self.normalizer = DatasetFieldsNormalizer("item", "item_id")

        # 7. Manage sentence tokenization and augmentation (misspellings, case)
        self.items_set_manager = TextItemSetManager(
            self.normalizer,
            items_set_splitter=ItemsSetSplitter(
                TokenGroupTextSplitter(
                    tokenizer=AutoTokenizer.from_pretrained(self.model_name),
                    blocks_splitter=DummySentenceSplitter(),
                )
            ),
            augmenter=ItemsSetAugmentationApplier(
                AugmentationsComposition(
                    [
                        ChangeCases(5),
                        Misspellings(5),
                    ]
                )
            ),
            do_augment_test=False,
        )

        # 8. Define metrics for tracking during training/eval
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

        # 9. Setup experiment tracking
        self.manager = ExperimentsManager.from_wrapper(
            wrapper=context.mlflow_client,
            main_metric="test_not_irrelevant_dist_shift",
            plugin_name=self.meta.name,
            accumulators=self.accumulators,
        )

        # 10. Define search grid for initial hyperparameters
        self.initial_params = INITIAL_PARAMS
        self.initial_params.update(
            {
                "not_irrelevant_only": [True],
                "negative_downsampling": [0.5],
                "examples_order": [[11]],
            }
        )

        # 11. Define fine-tuning settings: loss, steps, epochs
        self.settings = FineTuningSettings(
            loss_func=CosineProbMarginRankingLoss(),
            step_size=35,
            test_each_n_sessions=0.5,
            num_epochs=3,
        )

    def upload_initial_model(self) -> None:
        """
        Download and wrap the base model in an E5 architecture.
        Upload it to the experiment manager for versioning.
        """
        model = context.model_downloader.download_model(
            model_name=self.model_name,
            download_fn=lambda mn: SentenceTransformer(mn),
        )
        model = TextToTextE5Model(model)
        self.manager.upload_initial_model(model)

    def get_query_retriever(self) -> QueryRetriever:
        """
        Return the component that retrieves user queries from sessions.
        """
        return self.retriever

    def get_items_splitter(self) -> ItemSplitter:
        """
        Return the sentence splitter for breaking long text into chunks.
        """
        return DummySentenceSplitter()

    def get_items_preprocessor(self) -> ItemsDatasetDictPreprocessor:
        """
        Return the preprocessor that tokenizes and prepares text fields.
        """
        return self.items_set_manager.preprocessor

    def get_data_loader(self) -> DataLoader:
        """
        Return the data loader that fetches text objects from GCP buckets.
        """
        return self.data_loader

    def get_manager(self) -> ExperimentsManager:
        """
        Return the experiment tracker responsible for logging and metrics.
        """
        return self.manager

    def get_inference_client_factory(self) -> TritonClientFactory:
        """
        Return or create a Triton client for inference via the E5 model.
        Automatically wraps preprocessing and endpoint config.
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
        Prepare ranking dataset and return the configured builder to run training.

        :param clickstream: List of user interaction sessions with events.
        :return: A FineTuningBuilder object to orchestrate training.
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
        Return the vector search configuration (dimension, metric type).
        """
        return SearchIndexInfo(
            dimensions=768,
            metric_type=MetricType.COSINE,
            metric_aggregation_type=MetricAggregationType.AVG,
        )

    def get_vectors_adjuster(self) -> VectorsAdjuster:
        """
        Return a vector adjuster to apply out-of-training item vector
        improvement after fine-tuning.
        """
        return TorchBasedAdjuster(
            adjustment_rate=0.1, search_index_info=self.get_search_index_info()
        )

    def get_vectordb_optimizations(self) -> List[Optimization]:
        """
        Return a list of vector DB optimization steps (none by default).
        """
        return []
