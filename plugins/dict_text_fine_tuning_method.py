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
from embedding_studio.data_storage.loaders.cloud_storage.s3.item_meta import (
    S3FileMeta,
)
from embedding_studio.data_storage.loaders.cloud_storage.s3.s3_json_loader import (
    AwsS3JSONLoader,
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
from embedding_studio.embeddings.data.items.managers.dict import (
    DictItemSetManager,
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
from embedding_studio.embeddings.splitters.dict.field_combined_splitter import (
    FieldCombinedSplitter,
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


class DefaultDictTextFineTuningMethod(FineTuningMethod):
    meta = PluginMeta(
        name="DictDefaultMethodForObjectsTextOnly",  # Should be a python-like naming
        version="0.0.1",
        description="A default fine-tuning text plugin",
    )

    def __init__(self):
        # uncomment and pass your credentials to use your own s3 bucket
        # creds = {
        #     "role_arn": "arn:aws:iam::123456789012:role/some_data"
        #     "aws_access_key_id": "TESTACCESSKEIDTEST11",
        #     "aws_secret_access_key": "QWERTY1232qdsadfasfg5349BBdf30ekp23odk03",
        # }
        # self.data_loader = AwsS3DataLoader(**creds)

        self.model_name = "intfloat/multilingual-e5-large"
        self.inference_client_factory = None
        # with empty creds, use anonymous session
        creds = {}
        self.data_loader = AwsS3JSONLoader(**creds)
        self.field_names = []  # Provide your dict field names here

        self.retriever = TextQueryRetriever()
        self.sessions_converter = ClickstreamSessionConverter(
            item_type=S3FileMeta
        )
        self.splitter = TrainTestSplitter()
        self.normalizer = DatasetFieldsNormalizer("item", "item_id")
        self.items_set_manager = DictItemSetManager(
            self.normalizer,
            items_set_splitter=ItemsSetSplitter(
                TokenGroupTextSplitter(
                    tokenizer=AutoTokenizer.from_pretrained(self.model_name),
                    blocks_splitter=FieldCombinedSplitter(
                        field_names=self.field_names
                    ),
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

        self.manager = ExperimentsManager.from_wrapper(
            wrapper=context.mlflow_client,
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
            test_each_n_sessions=0.5,
            num_epochs=3,
        )

    def upload_initial_model(self) -> None:
        model = context.model_downloader.download_model(
            model_name=self.model_name,
            download_fn=lambda mn: SentenceTransformer(mn),
        )
        model = TextToTextE5Model(model)
        self.manager.upload_initial_model(model)

    def get_query_retriever(self) -> QueryRetriever:
        return self.retriever

    def get_data_loader(self) -> DataLoader:
        return self.data_loader

    def get_manager(self) -> ExperimentsManager:
        return self.manager

    def get_inference_client_factory(self) -> TritonClientFactory:
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
            dimensions=1024,
            metric_type=MetricType.COSINE,
            metric_aggregation_type=MetricAggregationType.AVG,
        )

    def get_vectors_adjuster(self) -> VectorsAdjuster:
        return TorchBasedAdjuster(
            adjustment_rate=0.1, search_index_info=self.get_search_index_info()
        )
