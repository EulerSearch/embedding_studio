import os
from typing import Dict, List

import torch
from torch import nn

from embedding_studio.inference_management.triton.manager import (
    TritonModelStorageManager,
)


class JitTraceTritonModelStorageManager(TritonModelStorageManager):
    """
    A Triton model storage manager that uses PyTorch JIT tracing for model serialization.

    This manager saves models by tracing them with example inputs, which optimizes them for inference
    but may not preserve dynamic control flow. Suitable for models with static computational graphs.

    :param storage_info: Information about where and how the model should be stored.
    :param do_dynamic_batching: Whether to enable dynamic batching for the model.
    :return: A manager for handling JIT-traced model storage operations.
    """

    def _get_model_artifacts(self) -> List[str]:
        """
        Returns the list of artifact filenames expected in the model directory.

        For a JIT-traced model, only the traced model file is required.

        :return: A list containing the traced model filename.
        """
        return ["model.pt"]

    def _generate_triton_config_model_info(self) -> List[str]:
        """
        Generates the model information section of the Triton configuration.

        Configures the model for the PyTorch LibTorch backend with appropriate settings.

        :return: A list of configuration lines for the model information section.
        """
        return [
            'name: "{}"'.format(self._storage_info.model_name),
            'platform: "pytorch_libtorch"',
            "max_batch_size: 16",
        ]

    def _save_model(
        self, model: nn.Module, example_inputs: Dict[str, torch.Tensor]
    ):
        """
        Saves the model using PyTorch JIT tracing.

        Traces the model execution with the provided example inputs to create an optimized
        serialized model that can be loaded by Triton's LibTorch backend.

        :param model: The PyTorch model to trace and save.
        :param example_inputs: Dictionary mapping input names to example tensors.
        :return: None
        """
        # Assumes model is prepared to accept inputs as a dictionary when traced
        traced_model = torch.jit.trace(model, tuple(example_inputs.values()))
        torch.jit.save(
            traced_model,
            os.path.join(self._storage_info.model_version_path, "model.pt"),
        )
