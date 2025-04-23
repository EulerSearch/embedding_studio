import logging
from abc import ABC, abstractmethod
from typing import Any, List, Optional

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.grpc.service_pb2 import ModelReadyRequest
from tritonclient.utils import InferenceServerException

from embedding_studio.core.config import settings
from embedding_studio.inference_management.triton.model_storage_info import (
    DeployedModelInfo,
)
from embedding_studio.utils.retry import retry_method
from embedding_studio.workers.fine_tuning.utils.config import (
    RetryConfig,
    RetryParams,
)

logger = logging.getLogger(__name__)


class TritonClient(ABC):
    """
    Abstract base class for interacting with the Triton Inference Server.
    Provides functionality for model readiness checks and inference operations.
    """

    def __init__(
        self,
        url: str,
        plugin_name: str,
        embedding_model_id: str,
        same_query_and_items: bool = False,
        retry_config: Optional[RetryConfig] = None,
    ):
        """
        Initialize the Triton client connection with the specified model version.

        :param url: tritonserver connection URL
        :param plugin_name: model's plugin name
        :param embedding_model_id: deployed model ID
        :param same_query_and_items: are query and items models actually the same model
        :param retry_config: retry policy configuration for connection attempts
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

    def _is_model_ready(self, is_query: bool) -> bool:
        """
        Internal method to check if a specific model (query or items) is ready on the Triton server.

        :param is_query: True to check query model, False to check items model
        :return: True if the specified model is ready, False otherwise
        """
        try:
            model_name = (
                self.query_model_info.name
                if is_query
                else self.items_model_info.name
            )
            # Check if the model is ready
            request = ModelReadyRequest(
                name=model_name,
            )
            response = self.client._client_stub.ModelReady(request)
            return response.ready
        except Exception as e:
            logger.exception(f"Error checking model '{model_name}': {e}")
            return False

    def is_model_ready(self) -> bool:
        """
        Check if all required models are deployed and ready on the Triton server.

        :return: True if all needed models are deployed, False otherwise
        """
        if self.same_query_and_items:
            return self._is_model_ready(True)
        else:
            # If query and items model are different - check them separately
            return self._is_model_ready(
                is_query=False
            ) and self._is_model_ready(is_query=True)

    @staticmethod
    def _get_default_retry_config() -> RetryConfig:
        """
        Create default retry configuration for connection and inference attempts.

        :return: RetryConfig object with default parameters
        """
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
        Prepare input for query embedding from the Triton server.
        Must be implemented to handle different data types (e.g., images, text).

        :param query: Query data to be prepared for inference
        :return: List of InferInput objects ready for submission to Triton

        Example implementation:
        ```python
        def _prepare_query(self, query: str) -> List[grpcclient.InferInput]:
            # For text models:
            encoded_text = self.tokenizer.encode(query, max_length=128, truncation=True)
            text_tensor = np.array([encoded_text], dtype=np.int64)
            infer_input = grpcclient.InferInput("input_ids", text_tensor.shape, "INT64")
            infer_input.set_data_from_numpy(text_tensor)
            return [infer_input]
        ```
        """

    @abstractmethod
    def _prepare_items(self, data: Any) -> List[grpcclient.InferInput]:
        """
        Prepare input for items embedding from the Triton server.
        Must be implemented to handle different data types (e.g., images, text).

        :param data: Items data to be prepared for inference
        :return: List of InferInput objects ready for submission to Triton

        Example implementation:
        ```python
        def _prepare_items(self, data: List[str]) -> List[grpcclient.InferInput]:
            # For batch text processing:
            batch_tensors = []
            for item in data:
                encoded_text = self.tokenizer.encode(item, max_length=128, truncation=True)
                batch_tensors.append(encoded_text)

            padded_batch = pad_sequences(batch_tensors, maxlen=128, padding='post')
            batch_tensor = np.array(padded_batch, dtype=np.int64)

            infer_input = grpcclient.InferInput("input_ids", batch_tensor.shape, "INT64")
            infer_input.set_data_from_numpy(batch_tensor)
            return [infer_input]
        ```
        """

    def forward_query(self, query: Any) -> np.ndarray:
        """
        Send a query to the Triton server and receive embedding output.

        :param query: Query data to be embedded
        :return: Numpy array containing query embedding
        """
        inputs = self._prepare_query(query)
        return self._send_query_request(inputs)

    def forward_items(self, items: List[Any]) -> np.ndarray:
        """
        Send a list of items to the Triton server and receive embedding outputs.

        :param items: List of items data to be embedded
        :return: Numpy array containing item embeddings
        """
        inputs = self._prepare_items(items)
        return self._send_items_request(inputs)

    @retry_method(name="query_inference")
    def _send_query_request(
        self, inputs: List[grpcclient.InferInput]
    ) -> np.ndarray:
        """
        Helper function to send a query request to the Triton server with retry logic.

        :param inputs: List of prepared InferInput objects
        :return: Numpy array with model output
        """

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
        """
        Helper function to send an items request to the Triton server with retry logic.

        :param inputs: List of prepared InferInput objects
        :return: Numpy array with model output
        """
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
    Provides a standardized way to create client instances for specific embedding models.
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

        :param url: The URL of the Triton Inference Server
        :param plugin_name: The name of the plugin/model used for inference tasks
        :param same_query_and_items: Indicates whether the same model handles both queries and items
        :param retry_config: Retry policy configuration
        """
        self.url = url
        self.plugin_name = plugin_name
        self.same_query_and_items = same_query_and_items
        self.retry_config = retry_config

    @abstractmethod
    def get_client(self, embedding_model_id: str, **kwargs):
        """
        Create an instance of a specified TritonClient subclass with a specific model version.

        :param embedding_model_id: The deployed ID of the model
        :param kwargs: Additional keyword arguments to pass to the client class constructor
        :return: An instance of a TritonClient subclass

        Example implementation:
        ```python
        def get_client(self, embedding_model_id: str, **kwargs):
            return TextTritonClient(
                url=self.url,
                plugin_name=self.plugin_name,
                embedding_model_id=embedding_model_id,
                tokenizer=self.tokenizer,
                retry_config=self.retry_config,
                **kwargs
            )
        ```
        """
