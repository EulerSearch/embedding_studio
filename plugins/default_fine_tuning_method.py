import gc
from typing import List

import torch.cuda
from sentence_transformers import SentenceTransformer

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
from embedding_studio.data_storage.loaders.cloud_storage.s3.s3_image_loader import (
    AwsS3ImageLoader,
)
from embedding_studio.data_storage.loaders.data_loader import DataLoader
from embedding_studio.embeddings.data.clickstream.train_test_splitter import (
    TrainTestSplitter,
)
from embedding_studio.embeddings.data.items.managers.clip import (
    CLIPItemSetManager,
)
from embedding_studio.embeddings.data.utils.fields_normalizer import (
    DatasetFieldsNormalizer,
)
from embedding_studio.embeddings.inference.triton.client import (
    TritonClientFactory,
)
from embedding_studio.embeddings.inference.triton.text_to_image.clip import (
    CLIPModelTritonClientFactory,
)
from embedding_studio.embeddings.losses.prob_cosine_margin_ranking_loss import (
    CosineProbMarginRankingLoss,
)
from embedding_studio.embeddings.models.text_to_image.clip import (
    TextToImageCLIPModel,
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


class DefaultFineTuningMethod(FineTuningMethod):
    meta = PluginMeta(
        name="DefaultFineTuningMethod",  # Should be a python-like naming
        version="0.0.1",
        description="A default fine-tuning plugin",
    )

    def __init__(self):
        # uncomment and pass your credentials to use your own s3 bucket
        # creds = {
        #     "role_arn": "arn:aws:iam::123456789012:role/some_data"
        #     "aws_access_key_id": "TESTACCESSKEIDTEST11",
        #     "aws_secret_access_key": "QWERTY1232qdsadfasfg5349BBdf30ekp23odk03",
        # }
        # self.data_loader = AwsS3DataLoader(**creds)
        self.model_name = "clip-ViT-B-32"
        self.tokenizer_name = (
            "sentence-transformers/clip-ViT-B-32-multilingual-v1"
        )
        self.inference_client_factory = None
        # with empty creds, use anonymous session
        creds = {}
        self.data_loader = AwsS3ImageLoader(**creds)

        self.retriever = TextQueryRetriever()
        self.sessions_converter = ClickstreamSessionConverter(
            item_type=S3FileMeta
        )
        self.splitter = TrainTestSplitter()
        self.normalizer = DatasetFieldsNormalizer("item", "item_id")
        self.items_set_manager = CLIPItemSetManager(self.normalizer)

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
            download_fn=lambda mn: SentenceTransformer(mn),
        )
        model = TextToImageCLIPModel(model)
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
            self.inference_client_factory = CLIPModelTritonClientFactory(
                f"{settings.INFERENCE_HOST}:{settings.INFERENCE_GRPC_PORT}",
                plugin_name=self.meta.name,
                transform=self.items_set_manager.preprocessor,
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
            dimensions=512,
            metric_type=MetricType.COSINE,
            metric_aggregation_type=MetricAggregationType.AVG,
        )
