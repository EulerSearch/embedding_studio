import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, List

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

from embedding_studio.core.config import settings
from embedding_studio.workers.fine_tuning.utils.config import (
    RetryConfig,
    RetryParams,
)
from embedding_studio.workers.fine_tuning.utils.retry import retry_method


logger = logging.getLogger(__name__)


class TritonClient(ABC):
    _MODEL_VERSIONS = {"blue": "1", "green": "2"}

    def __init__(
        self,
        url: str,
        plugin_name: str,
        model_version: str = "blue",
        same_query_and_items: bool = False,
        retry_config: Optional[RetryConfig] = None,
    ):
        """Initialize the Triton client connection with the specified model version.
        :param url: tritonserver connection URL.
        :param plugin_name: model's plugin name.
        :param model_version: version of a model ("green" means deployed, but not used in production,
                                                 "blue" means used in production) (default: blue).
        :param same_query_and_items: are query and items models acutally the same model (default: False).
        :param retry_config: retry policy (default: None).
        """
        self.url = url
        self.plugin_name = plugin_name
        if model_version not in TritonClient._MODEL_VERSIONS:
            logger.warning(
                f'Unknown model version: {model_version}. Excepted: {", ".join(list(TritonClient._MODEL_VERSIONS.keys()))}'
            )
        self.model_version = TritonClient._MODEL_VERSIONS.get(
            model_version, model_version
        )  # Save the model version
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
        config['query_inference'] = RetryParams(
            max_attempts=settings.INFERENCE_QUERY_EMBEDDING_ATTEMPTS,
            wait_time_seconds=settings.INFERENCE_QUERY_EMBEDDING_WAIT_TIME_SECONDS,
        )
        config['items_inference'] = RetryParams(
            max_attempts=settings.INFERENCE_ITEMS_EMBEDDING_ATTEMPTS,
            wait_time_seconds=settings.INFERENCE_ITEMS_EMBEDDING_WAIT_TIME_SECONDS,
        )
        return config

    @abstractmethod
    def _prepare_query(self, query: Any) -> grpcclient.InferInput:
        """
        Prepare input for the Triton server.
        Must be implemented to handle different data types (e.g., images, text).
        """
        pass

    @abstractmethod
    def _prepare_items(self, data: Any) -> grpcclient.InferInput:
        """
        Prepare input for the Triton server.
        Must be implemented to handle different data types (e.g., images, text).
        """
        pass

    def forward_query(self, query: Any) -> np.ndarray:
        """Send a query to the Triton server and receive the output."""
        input_tensor = self._prepare_query(query)
        return self._send_query_request([input_tensor], is_query=True)

    def forward_items(self, items: List[Any]) -> np.ndarray:
        """Send a list of items to the Triton server and receive the output."""
        inputs = [self._prepare_items(item) for item in items]
        return self._send_items_request(inputs)

    @retry_method(name="query_inference")
    def _send_query_request(
        self, inputs: List[grpcclient.InferInput]
    ) -> np.ndarray:
        """Helper function to send a request to the Triton server."""
        try:
            model_name = f"{self.plugin_name}_query"
            response = self.client.infer(
                model_name, inputs=inputs, model_version=self.model_version
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
                f"{self.plugin_name}_query"
                if self.same_query_and_items
                else f"{self.plugin_name}_items"
            )
            response = self.client.infer(
                model_name, inputs=inputs, model_version=self.model_version
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
        self, url: str,
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
    def get_client(self, model_version: str = "blue", **kwargs):
        """
        Create an instance of a specified TritonClient subclass with a specific model version.

        :param model_version: The deployment version of the model ('blue' or 'green').
        :param kwargs: Additional keyword arguments to pass to the client class constructor.
        :return: An instance of the specified TritonClient subclass.
        """
