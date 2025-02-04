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
from embedding_studio.workers.fine_tuning.prepare_data import prepare_data


class CategoriesTextFineTuningMethod(CategoriesFineTuningMethod):
    meta = PluginMeta(
        name="CategoriesTextFineTuningMethod",  # Should be a python-like naming
        version="0.0.1",
        description="A default text fine-tuning plugin for categories predictor",
    )

    def __init__(self):
        # uncomment and pass your credentials to use your own gcp bucket
        # creds = {
        #     "credentials_path": "./etc/your-gcp-credentials.json",
        #     "use_system_info": False
        # }

        self.model_name = (
            "EmbeddingStudio/all-MiniLM-L6-v2-huggingface-categories"
        )
        self.tokenizer_name = (
            "EmbeddingStudio/all-MiniLM-L6-v2-huggingface-categories"
        )
        self.inference_client_factory = None
        # with empty creds, use anonymous session
        creds = {"use_system_info": True}
        self.data_loader = GCPTextLoader(**creds)

        self.field_names = []  # Provide your dict field names here

        self.retriever = TextQueryRetriever()
        self.sessions_converter = ClickstreamSessionConverter(
            item_type=GCPFileMeta
        )
        self.splitter = TrainTestSplitter()
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
                AugmentationsComposition([ChangeCases(5), Misspellings(5)])
            ),
            do_augment_test=False,
        )

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

        self.manager = ExperimentsManager(
            tracking_uri=settings.MLFLOW_TRACKING_URI,
            main_metric="test_not_irrelevant_dist_shift",
            plugin_name=self.meta.name,
            accumulators=self.accumulators,
        )

        self.initial_params = INITIAL_PARAMS
        self.initial_params.update(
            {
                "not_irrelevant_only": [True],
                "negative_downsampling": [
                    0.5,
                ],
                "examples_order": [
                    [
                        11,
                    ]
                ],
            }
        )

        self.settings = FineTuningSettings(
            loss_func=CosineProbMarginRankingLoss(),
            step_size=35,
            test_each_n_inputs=0.5,
            num_epochs=3,
        )

    def upload_initial_model(self) -> None:
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
        return self.retriever

    def get_data_loader(self) -> DataLoader:
        return self.data_loader

    def get_manager(self) -> ExperimentsManager:
        return self.manager

    def get_inference_client_factory(self) -> TritonClientFactory:
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
        """Return a SearchIndexInfo instance. Define a parameters of vectordb index."""
        return SearchIndexInfo(
            dimensions=384,
            metric_type=MetricType.COSINE,
            metric_aggregation_type=MetricAggregationType.AVG,
        )

    def get_vectors_adjuster(self) -> VectorsAdjuster:
        raise NotImplementedError()

    def get_category_selector(self) -> AbstractSelector:
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
        return 100

    def get_max_margin(self) -> float:
        return 0.7
