import gc
import os
from typing import List

import torch.cuda
from transformers import AutoModel, AutoTokenizer

from plugins.custom.data_storage.loaders.hf_categories_query_generator import (
    HugsearchCategoriesQueryGenerator,
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
from embedding_studio.core.plugin import CategoriesFineTuningMethod
from embedding_studio.data_storage.loaders.data_loader import DataLoader
from embedding_studio.data_storage.loaders.sql.pgsql.item_meta import (
    PgsqlFileMeta,
)
from embedding_studio.data_storage.loaders.sql.pgsql.pgsql_text_loader import (
    PgsqlTextLoader,
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
    HnswParameters,
    MetricAggregationType,
    MetricType,
    SearchIndexInfo,
)
from embedding_studio.models.plugin import FineTuningBuilder, PluginMeta
from embedding_studio.vectordb.optimization import Optimization
from embedding_studio.workers.fine_tuning.prepare_data import prepare_data


class HFCategoriesTextFineTuningMethod(CategoriesFineTuningMethod):
    """
    Fine-tuning plugin for category prediction using Hugging Face data.

    This plugin is designed to support semantic category suggestion or
    classification using HF card-level text. It loads PostgreSQL-backed
    data from a `synthetic_text` column and fine-tunes a BERT-like model.

    Used in demo and production for Hugging Face category prediction.

    Key features:
    - PostgreSQL loading with `synthetic_text` built via query generator.
    - Tokenizes long inputs using sentence-level splitting.
    - Supports metric tracking and MLflow integration.
    - Exposes category selector for inference ranking via logits.
    """

    meta = PluginMeta(
        name="HFCategoriesTextFineTuningMethod",
        version="0.0.1",
        description="A default text fine-tuning plugin for huggingface categories predictor",
    )

    def __init__(self):
        """
        Step-by-step initialization of the Hugging Face categories plugin.

        Steps:
        1. Set model and tokenizer names (must match Triton version).
        2. Load PostgreSQL credentials from environment variables.
        3. Configure PgsqlTextLoader with HugsearchCategoriesQueryGenerator.
        4. Initialize components: retriever, session converter, splitter.
        5. Apply sentence-based splitting and text augmentation.
        6. Define metric accumulators for training and validation.
        7. Set up experiment tracking via MLflow wrapper.
        8. Add initial fine-tuning parameters for experimentation.
        9. Configure training loop settings (loss, steps, etc).
        """
        # 1. Model and tokenizer names
        self.model_name = (
            "EmbeddingStudio/all-MiniLM-L6-v2-huggingface-categories"
        )
        self.tokenizer_name = self.model_name
        self.inference_client_factory = None

        # 2. PostgreSQL connection from env
        HF_POSTGRES_HOST = os.getenv("HF_POSTGRES_HOST")
        HF_POSTGRES_PORT = os.getenv("HF_POSTGRES_PORT")
        HF_POSTGRES_USER = os.getenv("HF_POSTGRES_USER")
        HF_POSTGRES_PASSWORD = os.getenv("HF_POSTGRES_PASSWORD")
        HF_POSTGRES_DB = os.getenv("HF_POSTGRES_DB")

        connection_string = (
            f"postgresql://{HF_POSTGRES_USER}:{HF_POSTGRES_PASSWORD}"
            f"@{HF_POSTGRES_HOST}:{HF_POSTGRES_PORT}/{HF_POSTGRES_DB}"
        )

        # 3. PostgreSQL loader with synthetic text field
        self.data_loader = PgsqlTextLoader(
            connection_string=connection_string,
            query_generator=HugsearchCategoriesQueryGenerator,
            text_column="synthetic_text",
        )

        self.field_names = []  # Unused, placeholder for compatibility

        # 4. Session → query retrieval & session → labeled dataset conversion
        self.retriever = TextQueryRetriever()
        self.sessions_converter = ClickstreamSessionConverter(
            item_type=PgsqlFileMeta
        )
        self.splitter = TrainTestSplitter()

        # 5. Field normalization and item preparation with tokenization
        self.normalizer = DatasetFieldsNormalizer("item", "item_id")
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

        # 6. Metric accumulators (loss, dist shifts)
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

        # 7. Track experiments
        self.manager = ExperimentsManager.from_wrapper(
            wrapper=context.mlflow_client,
            main_metric="test_not_irrelevant_dist_shift",
            plugin_name=self.meta.name,
            accumulators=self.accumulators,
        )

        # 8. Define parameter search space
        self.initial_params = INITIAL_PARAMS
        self.initial_params.update(
            {
                "not_irrelevant_only": [True],
                "negative_downsampling": [0.5],
                "examples_order": [[11]],
            }
        )

        # 9. Training configuration
        self.settings = FineTuningSettings(
            loss_func=CosineProbMarginRankingLoss(),
            step_size=35,
            test_each_n_inputs=0.5,
            num_epochs=3,
        )

    def upload_initial_model(self) -> None:
        """
        Downloads and wraps the pretrained BERT model for HF categories.
        Uploads the wrapped model to the experiment manager.
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
        Returns the retriever for session-based query extraction.
        """
        return self.retriever

    def get_data_loader(self) -> DataLoader:
        """
        Returns the PostgreSQL-backed loader for category rows.
        """
        return self.data_loader

    def get_items_preprocessor(self) -> ItemsDatasetDictPreprocessor:
        """
        Returns the sentence-splitting preprocessor with augmentation.
        """
        return self.items_set_manager.preprocessor

    def get_manager(self) -> ExperimentsManager:
        """
        Returns the experiment tracker for logging metrics and parameters.
        """
        return self.manager

    def get_inference_client_factory(self) -> TritonClientFactory:
        """
        Returns a Triton client factory for inference with the BERT model.
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
        Builds the fine-tuning pipeline from session click data.

        :param clickstream: List of session interactions and queries.
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
        Returns vector DB configuration for indexing embeddings.

        Uses 384-dim dot-product vectors with HNSW config tuned for recall.
        """
        return SearchIndexInfo(
            dimensions=384,
            metric_type=MetricType.DOT,
            metric_aggregation_type=MetricAggregationType.MIN,
            hnsw=HnswParameters(m=16, ef_construction=96),
        )

    def get_vectors_adjuster(self) -> VectorsAdjuster:
        """
        (Not implemented) Post-training vector adjustment is disabled here.
        """
        raise NotImplementedError()

    def get_category_selector(self) -> AbstractSelector:
        """
        Returns a selector that ranks categories based on probability score.

        Uses temperature scaling and logit margin constraints.
        """
        # TODO: do a category selector versioning, just as embedding model
        return ProbsDistBasedSelector(
            search_index_info=SearchIndexInfo(
                dimensions=384,
                metric_type=MetricType.DOT,
                metric_aggregation_type=MetricAggregationType.MIN,
            ),
            is_similarity=True,
            margin=0.2,
            scale=10.0,
            scale_to_one=False,
            prob_threshold=0.985,
        )

    def get_max_similar_categories(self) -> int:
        """
        Returns the maximum number of top categories retrieved at inference.
        """
        return 36

    def get_max_margin(self) -> float:
        """
        Returns the similarity margin for filtering predictions.
        """
        return -1.0

    def get_vectordb_optimizations(self) -> List[Optimization]:
        """
        Returns no additional vector DB optimization steps.
        """
        return []
