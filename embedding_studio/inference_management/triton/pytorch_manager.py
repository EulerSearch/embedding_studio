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
    def _get_model_artifacts(self) -> List[str]:
        # Returns the list of artifact filenames expected in the model directory.
        return ["model.pt", "model.py"]

    def _generate_triton_config_model_info(self) -> List[str]:
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
