from typing import Callable, List, Optional, Union

import numpy as np
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from tritonclient.grpc import InferInput

from embedding_studio.context.app_context import context
from embedding_studio.embeddings.inference.triton.client import (
    TritonClient,
    TritonClientFactory,
)
from embedding_studio.workers.fine_tuning.utils.config import RetryConfig


class TextToTextBERTTritonClient(TritonClient):
    """
    A specialized TritonClient to handle BERT model inputs for text-to-text applications,
    using transformers for text tokenization.
    """

    def __init__(
        self,
        url: str,
        plugin_name: str,
        embedding_model_id: str,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        preprocessor: Callable[[Union[str, dict]], str] = None,
        model_name: str = "EmbeddingStudio/all-MiniLM-L6-v2-huggingface-categories",
        retry_config: Optional[RetryConfig] = None,
        max_length: int = 256,
    ):
        """
        Initialize the Triton client with the capability to process text data.

        :param url: The URL of the Triton Inference Server.
        :param plugin_name: The name of the plugin/model used for inference tasks.
        :param embedding_model_id: deployed model ID.
        :param tokenizer: query text tokenizer
        :param preprocessor: The text preprocessing function.
        :param retry_config: retry policy (default: None).
        :param max_length: max tokenization length.
        """
        super().__init__(
            url,
            plugin_name,
            embedding_model_id,
            same_query_and_items=True,
            retry_config=retry_config,
        )

        self.preprocessor = preprocessor
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _prepare_query(self, query: str) -> List[InferInput]:
        """
        Prepare a single text input for the Triton server by tokenizing the query string.

        :param query: A string containing the text to be processed.
        """
        inputs = self.tokenizer(
            [
                query,
            ],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        infer_inputs = []
        for key, value in inputs.items():
            if key not in ["attention_mask", "input_ids"]:
                continue

            tensor_np = value.numpy().astype(
                np.int64
            )  # Adjust data type if necessary
            infer_input = InferInput(key, tensor_np.shape, "INT64")
            infer_input.set_data_from_numpy(tensor_np)
            infer_inputs.append(infer_input)

        return infer_inputs

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
            max_length=self.max_length,
        )
        infer_inputs = []
        for key, value in inputs.items():
            if key not in ["attention_mask", "input_ids"]:
                continue

            tensor_np = value.numpy().astype(np.int64)
            infer_input = InferInput(key, tensor_np.shape, "INT64")
            infer_input.set_data_from_numpy(tensor_np)
            infer_inputs.append(infer_input)

        return infer_inputs


class TextToTextBERTTritonClientFactory(TritonClientFactory):
    """
    Factory for creating instances of TritonClient with common configurations but different model versions.
    """

    def __init__(
        self,
        url: str,
        plugin_name: str,
        preprocessor: Callable[[Union[str, dict]], str] = None,
        model_name: str = "EmbeddingStudio/all-MiniLM-L6-v2-huggingface-categories",
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
        super(TextToTextBERTTritonClientFactory, self).__init__(
            url=url,
            plugin_name=plugin_name,
            same_query_and_items=True,
            retry_config=retry_config,
        )
        self.preprocessor = preprocessor
        self.model_name = model_name
        self.tokenizer = context.model_downloader.download_model(
            model_name=model_name,
            download_fn=lambda m: AutoTokenizer.from_pretrained(
                m, use_fast=False
            ),
        )

    def get_client(self, embedding_model_id: str, **kwargs):
        """
        Create an instance of a specified TritonClient subclass with a specific model version.

        :param embedding_model_id: The deployed ID of the model.
        :param kwargs: Additional keyword arguments to pass to the client class constructor.
        :return: An instance of the specified TritonClient subclass.
        """
        return TextToTextBERTTritonClient(
            url=self.url,
            plugin_name=self.plugin_name,
            embedding_model_id=embedding_model_id,
            preprocessor=self.preprocessor,
            tokenizer=self.tokenizer,
            retry_config=self.retry_config,
        )
