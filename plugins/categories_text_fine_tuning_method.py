import gc
from typing import List

import torch.cuda
from transformers import AutoModel, AutoTokenizer

from embedding_studio.clickstream_storage.converters.converter import (
    ClickstreamSessionConverter,
)
from embedding_studio.clickstream_storage.query_retriever import QueryRetriever
from embedding_studio.clickstream_storage.text_query_retriever import (
    TextQueryRetriever,
)
from embedding_studio.context.app_context import context
from embedding_studio.core.config import settings
from embedding_studio.core.plugin import CategoriesFineTuningMethod
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
from embedding_studio.embeddings.improvement.vectors_adjuster import (
    VectorsAdjuster,
)
from embedding_studio.embeddings.inference.triton.client import (
    TritonClientFactory,
)
from embedding_studio.embeddings.inference.triton.text_to_text.bert import (
    TextToTextBERTTritonClientFactory,
)
from embedding_studio.embeddings.losses.prob_cosine_margin_ranking_loss import (
    CosineProbMarginRankingLoss,
)
from embedding_studio.embeddings.models.text_to_text.bert import (
    TextToTextBertModel,
)
from embedding_studio.embeddings.selectors.prob_dist_based_selector import (
    ProbsDistBasedSelector,
)
from embedding_studio.embeddings.selectors.selector import AbstractSelector
from embedding_studio.embeddings.splitters.dataset_splitter import (
    ItemsSetSplitter,
)
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


class CategoriesTextFineTuningMethod(CategoriesFineTuningMethod):
    """
    Plugin for category prediction fine-tuning using text embeddings.

    This plugin is designed for classification-based tasks where embeddings are
    optimized to predict category labels. Uses BERT-style models and GCP loader.

    Steps to implement your own:
    1. Subclass CategoriesFineTuningMethod.
    2. Define `meta` with plugin name, version, and description.
    3. Implement all abstract methods including category selection logic.
    4. Configure loaders, preprocessors, and metrics trackers.
    5. Register in PluginManager and list it in inference config.
    """

    meta = PluginMeta(
        name="CategoriesTextFineTuningMethod",  # Should be a python-like naming
        version="0.0.1",
        description="A default text fine-tuning plugin for categories predictor",
    )

    def __init__(self):
        """
        Step-by-step setup of a text fine-tuning plugin for categories.

        Steps:
        1. Set up model and tokenizer names.
        2. Configure the data loader to read from GCP.
        3. Initialize retriever and clickstream converter.
        4. Split sessions into train/test.
        5. Normalize fields to match item schema.
        6. Configure sentence splitting and text augmentations.
        7. Define training and test metrics.
        8. Set up experiment manager (e.g. MLflow).
        9. Define hyperparameters for fine-tuning trials.
        10. Set training config: loss, epochs, validation.
        """
        # uncomment and pass your credentials to use your own gcp bucket
        # creds = {
        #     "credentials_path": "./etc/your-gcp-credentials.json",
        #     "use_system_info": False
        # }

        # 1. Define model + tokenizer
        self.model_name = (
            "EmbeddingStudio/all-MiniLM-L6-v2-huggingface-categories"
        )
        self.tokenizer_name = self.model_name
        self.inference_client_factory = None

        # 2. GCP loader (credentials optional)
        creds = {"use_system_info": True}
        self.data_loader = GCPTextLoader(**creds)

        # 3. Optional fields (if using dict-style structure)
        self.field_names = []

        # 4. Retrieve user queries from sessions
        self.retriever = TextQueryRetriever()

        # 5. Convert user clicks into labeled data
        self.sessions_converter = ClickstreamSessionConverter(
            item_type=GCPFileMeta
        )

        # 6. Session split logic
        self.splitter = TrainTestSplitter()

        # 7. Field normalization
        self.normalizer = DatasetFieldsNormalizer("item", "item_id")

        # 8. Sentence tokenization + augmentations (case/misspell)
        self.items_set_manager = TextItemSetManager(
            self.normalizer,
            items_set_splitter=ItemsSetSplitter(
                TokenGroupTextSplitter(
                    tokenizer=AutoTokenizer.from_pretrained(self.model_name),
                    blocks_splitter=DummySentenceSplitter(),
                )
            ),
            augmenter=ItemsSetAugmentationApplier(
                AugmentationsComposition([ChangeCases(5), Misspellings(5)])
            ),
            do_augment_test=False,
        )

        # 9. Training and test metrics
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

        # 10. MLflow or other experiment backend
        self.manager = ExperimentsManager.from_wrapper(
            wrapper=context.mlflow_client,
            main_metric="test_not_irrelevant_dist_shift",
            plugin_name=self.meta.name,
            accumulators=self.accumulators,
        )

        # 11. Define hyperparameter grid
        self.initial_params = INITIAL_PARAMS
        self.initial_params.update(
            {
                "not_irrelevant_only": [True],
                "negative_downsampling": [0.5],
                "examples_order": [[11]],
            }
        )

        # 12. Training config
        self.settings = FineTuningSettings(
            loss_func=CosineProbMarginRankingLoss(),
            step_size=35,
            test_each_n_inputs=0.5,
            num_epochs=3,
        )

    def upload_initial_model(self) -> None:
        """
        Downloads and wraps the base BERT model. Uploads it to the
        experiment manager as the initial reference for training.
        """
        model = context.model_downloader.download_model(
            model_name=self.model_name,
            download_fn=lambda mn: AutoModel.from_pretrained(mn),
        )
        model = TextToTextBertModel(model, max_length=256)
        self.manager.upload_initial_model(model)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    def get_query_retriever(self) -> QueryRetriever:
        """
        Returns the retriever that fetches search queries from session data.
        """
        return self.retriever

    def get_items_preprocessor(self) -> ItemsDatasetDictPreprocessor:
        """
        Returns the preprocessor for sentence-split and normalized text.
        """
        return self.items_set_manager.preprocessor

    def get_data_loader(self) -> DataLoader:
        """
        Returns the GCP-based text loader used for category data ingestion.
        """
        return self.data_loader

    def get_manager(self) -> ExperimentsManager:
        """
        Returns the MLflow-compatible experiment manager.
        """
        return self.manager

    def get_inference_client_factory(self) -> TritonClientFactory:
        """
        Returns the Triton client factory for serving BERT embeddings.
        Initializes it on first access with model and plugin name.
        """
        if self.inference_client_factory is None:
            self.inference_client_factory = TextToTextBERTTritonClientFactory(
                f"{settings.INFERENCE_HOST}:{settings.INFERENCE_GRPC_PORT}",
                plugin_name=self.meta.name,
                model_name=self.model_name,
            )
        return self.inference_client_factory

    def get_fine_tuning_builder(
        self, clickstream: List[SessionWithEvents]
    ) -> FineTuningBuilder:
        """
        Prepares ranking dataset and returns a builder to start training.

        :param clickstream: A list of session events with user feedback.
        :return: A configured FineTuningBuilder.
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
        Defines vector DB indexing schema (384D, cosine, AVG).
        """
        return SearchIndexInfo(
            dimensions=384,
            metric_type=MetricType.COSINE,
            metric_aggregation_type=MetricAggregationType.AVG,
        )

    def get_vectors_adjuster(self) -> VectorsAdjuster:
        """
        Optional vector adjustment logic. Not implemented in this plugin.
        """
        raise NotImplementedError()

    def get_category_selector(self) -> AbstractSelector:
        """
        Returns the selector used to determine category similarity and
        margins based on softmaxed embedding logits.
        """
        # TODO: do a category selector versioning, just as embedding model
        return ProbsDistBasedSelector(
            search_index_info=SearchIndexInfo(
                dimensions=384,
                metric_type=MetricType.DOT,
                metric_aggregation_type=MetricAggregationType.AVG,
            ),
            is_similarity=True,
            margin=0.2,
            scale=10.0,
            scale_to_one=True,
            prob_threshold=0.7,
        )

    def get_max_similar_categories(self) -> int:
        """
        Max number of top-k similar categories to retrieve.
        """
        return 20

    def get_max_margin(self) -> float:
        """
        Max allowed distance/similarity between categories.
        """
        return 0.7

    def get_vectordb_optimizations(self) -> List[Optimization]:
        """
        Optional vector DB post-processing (empty by default).
        """
        return []
