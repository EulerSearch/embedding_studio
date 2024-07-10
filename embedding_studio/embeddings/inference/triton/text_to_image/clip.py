import gc
from copy import deepcopy
from typing import Callable, List, Optional, Union

import cv2
import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from tritonclient.grpc import InferInput

from embedding_studio.embeddings.inference.triton.client import (
    TritonClient,
    TritonClientFactory,
)
from embedding_studio.workers.fine_tuning.utils.config import RetryConfig


class CLIPModelTritonClient(TritonClient):
    """
    A specialized TritonClient designed to handle different types of models (e.g., CLIP)
    using transformers for text tokenization and customizable image preprocessing for vision inputs.
    """

    def __init__(
        self,
        url: str,
        plugin_name: str,
        embedding_model_id: str,
        transform: Callable[[Image.Image], torch.Tensor] = None,
        model_name: str = "clip-ViT-B-32",
        retry_config: Optional[RetryConfig] = None,
    ):
        """
        Initialize the Triton client with the capability to process text and image data.

        :param url: The URL of the Triton Inference Server.
        :param plugin_name: The name of the plugin/model used for inference tasks.
        :param embedding_model_id: deployed model ID.
        :param transform: A function to preprocess images before sending them to the server.
        :param model_name: The name of the model for which the tokenizer is tailored (default is 'clip-ViT-B-32').
        :param retry_config: retry policy (default: None).
        """
        super().__init__(
            url,
            plugin_name,
            embedding_model_id,
            same_query_and_items=False,
            retry_config=retry_config,
        )

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, use_fast=True
            )
        except Exception:
            model = deepcopy(SentenceTransformer(model_name)[0])
            self.tokenizer = model.processor.tokenizer
            del model
            gc.collect()

        self.transform = transform

    def _prepare_query(self, query: str) -> List[InferInput]:
        """
        Prepare text input for the Triton server by tokenizing the query string.

        :param query: A string containing the text to be processed.
        """
        inputs = self.tokenizer(
            query,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        )
        inputs.pop("attention_mask")
        infer_inputs = []
        for key, value in inputs.items():
            tensor_np = value.numpy().astype(
                np.int64
            )  # Adjust data type if necessary
            infer_input = InferInput(key, tensor_np.shape, "INT64")
            infer_input.set_data_from_numpy(tensor_np)
            infer_inputs.append(infer_input)

        return infer_inputs

    def _prepare_items(
        self, data: List[Union[np.ndarray, Image.Image]]
    ) -> List[InferInput]:
        """
        Prepare a batch of image inputs for the Triton server. Handles both PIL images and numpy arrays,
        converting them as needed and applying a specified transformation.

        :param data: A list of PIL.Image instances or numpy arrays from images prepared using cv2.
        """
        processed_images = []

        for image in data:
            if isinstance(
                image, np.ndarray
            ):  # Convert from OpenCV BGR to RGB if needed
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)

            if self.transform:
                image = self.transform(image)

            if isinstance(image, Image.Image):
                image = np.array(image)  # Ensure it's a numpy array for Triton

            if isinstance(image, torch.Tensor):
                image = (
                    image.detach().cpu().numpy()
                )  # Ensure it's a numpy array for Triton

            if image.dtype != np.float32:
                image = image.astype(np.float32)

            processed_images.append(image)

        batch_images = np.stack(
            processed_images, axis=0
        )  # Stack images into a batch
        infer_input = InferInput("pixel_values", batch_images.shape, "FP32")
        infer_input.set_data_from_numpy(batch_images)
        return [
            infer_input,
        ]


class CLIPModelTritonClientFactory(TritonClientFactory):
    """
    Factory for creating instances of TritonClient with common configurations but different model versions.
    """

    def __init__(
        self,
        url: str,
        plugin_name: str,
        transform: Callable[[Image.Image], torch.Tensor] = None,
        model_name: str = "clip-ViT-B-32",
        retry_config: Optional[RetryConfig] = None,
    ):
        """
        Initialize the factory with common configuration parameters.

        :param url: The URL of the Triton Inference Server.
        :param plugin_name: The name of the plugin/model used for inference tasks.
        :param transform: A function to preprocess images before sending them to the server.
        :param model_name: The name of the model for which the tokenizer is tailored (default is 'clip-ViT-B-32').
        :param retry_config: retry policy (default: None).
        """
        super(CLIPModelTritonClientFactory, self).__init__(
            url=url,
            plugin_name=plugin_name,
            same_query_and_items=True,
            retry_config=retry_config,
        )
        self.transform = transform
        self.model_name = model_name

    def get_client(self, embedding_model_id: str, **kwargs):
        """
        Create an instance of a specified TritonClient subclass with a specific model version.

        :param embedding_model_id: The deployed ID of the model.
        :param kwargs: Additional keyword arguments to pass to the client class constructor.
        :return: An instance of the specified TritonClient subclass.
        """
        return CLIPModelTritonClient(
            url=self.url,
            plugin_name=self.plugin_name,
            embedding_model_id=embedding_model_id,
            transform=self.transform,
            model_name=self.model_name,
            retry_config=self.retry_config,
        )
