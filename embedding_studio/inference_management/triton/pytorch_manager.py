import os
from typing import Dict, List

import torch
from torch import nn

from embedding_studio.inference_management.triton.manager import (
    TritonModelStorageManager,
)
from embedding_studio.inference_management.triton.utils.generate_model_file import (
    generate_model_py,
    generate_sequential_model_py,
)


class PytorchTritonModelStorageManager(TritonModelStorageManager):
    """
    A Triton model storage manager that uses PyTorch state dict for model serialization,
    with a Python script for model initialization.

    This approach maintains the model's structure and dynamic behavior by saving model weights
    separately from the model definition. Suitable for models with dynamic control flow.

    :param storage_info: Information about where and how the model should be stored.
    :param do_dynamic_batching: Whether to enable dynamic batching for the model.
    :return: A manager for handling PyTorch model storage operations.
    """

    def _get_model_artifacts(self) -> List[str]:
        """
        Returns the list of artifact filenames expected in the model directory.

        For a PyTorch model, both the state dict and the model definition script are required.

        :return: A list containing the model state dict and Python script filenames.
        """
        # Returns the list of artifact filenames expected in the model directory.
        return ["model.pt", "model.py"]

    def _generate_triton_config_model_info(self) -> List[str]:
        """
        Generates the model information section of the Triton configuration.

        Configures the model for the PyTorch LibTorch backend with a reference to the Python
        script that will load and initialize the model.

        :return: A list of configuration lines for the model information section.
        """
        # Returns the base configuration for the model to be deployed on Triton.
        return [
            'name: "{}"'.format(self._storage_info.model_name),
            'platform: "pytorch_libtorch"',
            "max_batch_size: 16",
            'runtime: "model.py"',
        ]

    def _save_model(
        self, model: nn.Module, example_inputs: Dict[str, torch.Tensor]
    ):
        """
        Saves the model's state dict and generates a Python script for initialization.

        This method saves the model parameters and creates a Python script that can recreate
        the model structure when loaded by Triton. It handles both regular models and
        Sequential models differently.

        :param model: The PyTorch model to save.
        :param example_inputs: Dictionary mapping input names to example tensors.
        :return: None
        """
        # Saves the model's state dict and a Python script for model initialization.
        # Determines if the model is sequential and uses the appropriate function to generate the Python script.
        model_path = os.path.join(
            self._storage_info.model_version_path, "model.pt"
        )
        script_path = os.path.join(
            self._storage_info.model_version_path, "model.py"
        )
        torch.save(model.state_dict(), model_path)
        if not isinstance(model, nn.Sequential):
            generate_model_py(
                model, script_path, self._storage_info.embedding_studio_path
            )
        else:
            generate_sequential_model_py(
                model, script_path, self._storage_info.embedding_studio_path
            )
