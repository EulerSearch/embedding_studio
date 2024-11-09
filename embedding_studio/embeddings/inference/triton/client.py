import logging
from abc import ABC, abstractmethod
from typing import Any, List, Optional

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

from embedding_studio.core.config import settings
from embedding_studio.inference_management.triton.model_storage_info import (
    DeployedModelInfo,
)
from embedding_studio.workers.fine_tuning.utils.config import (
    RetryConfig,
    RetryParams,
)
from embedding_studio.workers.fine_tuning.utils.retry import retry_method

logger = logging.getLogger(__name__)


class TritonClient(ABC):
    def __init__(
        self,
        url: str,
        plugin_name: str,
        embedding_model_id: str,
        same_query_and_items: bool = False,
        retry_config: Optional[RetryConfig] = None,
    ):
        """Initialize the Triton client connection with the specified model version.
        :param url: tritonserver connection URL.
        :param plugin_name: model's plugin name.
        :param embedding_model_id: deployed model ID.
        :param same_query_and_items: are query and items models acutally the same model (default: False).
        :param retry_config: retry policy (default: None).
        """
        self.url = url
        self.plugin_name = plugin_name
        self.embedding_model_id = embedding_model_id

        self.query_model_info = DeployedModelInfo(
            plugin_name=plugin_name,
            embedding_model_id=embedding_model_id,
            model_type="query",
        )
        self.items_model_info = DeployedModelInfo(
            plugin_name=plugin_name,
            embedding_model_id=embedding_model_id,
            model_type="items",
        )

        self.client = grpcclient.InferenceServerClient(url=self.url)
        self.same_query_and_items = same_query_and_items
        self.retry_config = (
            retry_config
            if retry_config
            else TritonClient._get_default_retry_config()
        )

    @staticmethod
    def _get_default_retry_config() -> RetryConfig:
        default_retry_params = RetryParams(
            max_attempts=settings.DEFAULT_MAX_ATTEMPTS,
            wait_time_seconds=settings.DEFAULT_WAIT_TIME_SECONDS,
        )

        config = RetryConfig(default_params=default_retry_params)
        config["query_inference"] = RetryParams(
            max_attempts=settings.INFERENCE_QUERY_EMBEDDING_ATTEMPTS,
            wait_time_seconds=settings.INFERENCE_QUERY_EMBEDDING_WAIT_TIME_SECONDS,
        )
        config["items_inference"] = RetryParams(
            max_attempts=settings.INFERENCE_ITEMS_EMBEDDING_ATTEMPTS,
            wait_time_seconds=settings.INFERENCE_ITEMS_EMBEDDING_WAIT_TIME_SECONDS,
        )
        return config

    @abstractmethod
    def _prepare_query(self, query: Any) -> List[grpcclient.InferInput]:
        """
        Prepare input for the Triton server.
        Must be implemented to handle different data types (e.g., images, text).
        """

    @abstractmethod
    def _prepare_items(self, data: Any) -> List[grpcclient.InferInput]:
        """
        Prepare input for the Triton server.
        Must be implemented to handle different data types (e.g., images, text).
        """

    def forward_query(self, query: Any) -> np.ndarray:
        """Send a query to the Triton server and receive the output."""
        inputs = self._prepare_query(query)
        return self._send_query_request(inputs)

    def forward_items(self, items: List[Any]) -> np.ndarray:
        """Send a list of items to the Triton server and receive the output."""
        inputs = self._prepare_items(items)
        return self._send_items_request(inputs)

    @retry_method(name="query_inference")
    def _send_query_request(
        self, inputs: List[grpcclient.InferInput]
    ) -> np.ndarray:
        """Helper function to send a request to the Triton server."""
        try:
            model_name = self.query_model_info.name
            response = self.client.infer(
                model_name,
                inputs=inputs,
                model_version="1",
                priority=0,
            )
            return response.as_numpy(
                "output"
            )  # 'output' should be the name of your model's output tensor
        except InferenceServerException as e:
            logger.exception(f"Request failed: {e}")

        return

    @retry_method(name="items_inference")
    def _send_items_request(
        self, inputs: List[grpcclient.InferInput]
    ) -> np.ndarray:
        """Helper function to send a request to the Triton server."""
        try:
            model_name = (
                self.query_model_info.name
                if self.same_query_and_items
                else self.items_model_info.name
            )
            response = self.client.infer(
                model_name,
                inputs=inputs,
                model_version="1",
                priority=1,
            )
            return response.as_numpy(
                "output"
            )  # 'output' should be the name of your model's output tensor
        except InferenceServerException as e:
            logger.exception(f"Request failed: {e}")

        return


class TritonClientFactory:
    """
    Factory for creating instances of TritonClient with common configurations but different model versions.
    """

    def __init__(
        self,
        url: str,
        plugin_name: str,
        same_query_and_items: bool = False,
        retry_config: Optional[RetryConfig] = None,
    ):
        """
        Initialize the factory with common configuration parameters.

        :param url: The URL of the Triton Inference Server.
        :param plugin_name: The name of the plugin/model used for inference tasks.
        :param same_query_and_items: Indicates whether the same model handles both queries and items.
        :param retry_config: retry policy (default: None).
        """
        self.url = url
        self.plugin_name = plugin_name
        self.same_query_and_items = same_query_and_items
        self.retry_config = retry_config

    @abstractmethod
    def get_client(self, embedding_model_id: str, **kwargs):
        """
        Create an instance of a specified TritonClient subclass with a specific model version.

        :param embedding_model_id: The deployed ID of the model.
        :param kwargs: Additional keyword arguments to pass to the client class constructor.
        :return: An instance of the specified TritonClient subclass.
        """
