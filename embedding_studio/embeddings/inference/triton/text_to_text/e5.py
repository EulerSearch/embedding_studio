from typing import Callable, List, Optional, Union

import numpy as np
from transformers import AutoTokenizer
from tritonclient.grpc import InferInput

from embedding_studio.embeddings.inference.triton.client import (
    TritonClient,
    TritonClientFactory,
)
from embedding_studio.workers.fine_tuning.utils.config import (
    RetryConfig,
)


class TextToTextE5TritonClient(TritonClient):
    """
    A specialized TritonClient to handle E5 model inputs for text-to-text applications,
    using transformers for text tokenization.
    """

    def __init__(
        self,
        url: str,
        plugin_name: str,
        model_version: str = "blue",
        preprocessor: Callable[[Union[str, dict]], str] = None,
        model_name: str = "intfloat/multilingual-e5-large",
        retry_config: Optional[RetryConfig] = None,
    ):
        """
        Initialize the Triton client with the capability to process text data.

        :param url: The URL of the Triton Inference Server.
        :param plugin_name: The name of the plugin/model used for inference tasks.
        :param model_version: The deployment version of the model ('blue' or 'green').
        :param preprocessor: The text preprocessing function.
        :param model_name: The name of the model for which the tokenizer is tailored.
        :param retry_config: retry policy (default: None).
        """
        super().__init__(
            url, plugin_name, model_version, same_query_and_items=True, retry_config=retry_config
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True
        )
        self.preprocessor = preprocessor
        self.model_name = model_name

    def _prepare_input(self, query: str) -> InferInput:
        """
        Prepare text input for the Triton server by tokenizing the query string.

        :param query: A string containing the text to be processed.
        """
        inputs = self.tokenizer(
            query,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
        )
        inputs = (
            inputs["input_ids"].numpy().astype(np.int32)
        )  # Adjust data type if necessary
        infer_input = InferInput("text_input", inputs.shape, "INT32")
        infer_input.set_data_from_numpy(inputs)
        return infer_input

    def _prepare_query(self, query: str) -> InferInput:
        """
        Prepare a single text input for the Triton server by tokenizing the query string.

        :param query: A string containing the text to be processed.
        """
        inputs = self.tokenizer(
            f"query: {query}",
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
        )
        inputs = (
            inputs["input_ids"].numpy().astype(np.int32)
        )  # Adjust data type if necessary
        infer_input = InferInput("input0", inputs.shape, "INT32")
        infer_input.set_data_from_numpy(inputs)
        return infer_input

    def _prepare_items(self, data: List[Union[str, dict]]) -> List[InferInput]:
        """
        Prepare a list of text inputs for the Triton server. This method tokenizes each text entry in the list.

        :param data: A list of text data to be tokenized.
        """
        prep = self.preprocessor if self.preprocessor else lambda v: v
        inputs = self.tokenizer(
            [prep(item) for item in data],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
        )
        inputs = inputs["input_ids"].numpy().astype(np.int32)
        infer_inputs = [InferInput("input0", inputs.shape, "INT32")]
        infer_inputs[0].set_data_from_numpy(
            inputs
        )  # All text inputs are batched into one tensor
        return infer_inputs


class TextToTextE5TritonClientFactory(TritonClientFactory):
    """
    Factory for creating instances of TritonClient with common configurations but different model versions.
    """

    def __init__(
        self,
        url: str,
        plugin_name: str,
        preprocessor: Callable[[Union[str, dict]], str] = None,
        model_name: str = "intfloat/multilingual-e5-large",
        retry_config: Optional[RetryConfig] = None,
    ):
        """
        Initialize the factory with common configuration parameters.

        :param url: The URL of the Triton Inference Server.
        :param plugin_name: The name of the plugin/model used for inference tasks.
        :param preprocessor: The text preprocessing function.
        :param model_name: The name of the model for which the tokenizer is tailored.
        :param retry_config: retry policy (default: None).
        """
        super(TextToTextE5TritonClientFactory, self).__init__(
            url=url, plugin_name=plugin_name, same_query_and_items=True, retry_config=retry_config
        )
        self.preprocessor = preprocessor
        self.model_name = model_name

    def get_client(self, model_version: str = "blue", **kwargs):
        """
        Create an instance of a specified TritonClient subclass with a specific model version.

        :param model_version: The deployment version of the model ('blue' or 'green').
        :param kwargs: Additional keyword arguments to pass to the client class constructor.
        :return: An instance of the specified TritonClient subclass.
        """
        return TextToTextE5TritonClient(
            url=self.url,
            plugin_name=self.plugin_name,
            model_version=model_version,
            preprocessor=self.preprocessor,
            model_name=self.model_name,
            retry_config=self.retry_config
        )
