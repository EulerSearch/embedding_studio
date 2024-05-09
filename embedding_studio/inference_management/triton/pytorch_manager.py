import os
from typing import List

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
    def _generate_triton_config_model_info(self) -> List[str]:
        return [
            'name: "{}"'.format(self._storage_info.model_name),
            'platform: "pytorch_libtorch"',
            "max_batch_size: 16",
            'runtime: "model.py"',
        ]

    def _save_model(
        self, model: nn.Module, example_input: torch.Tensor, input_name: str
    ):

        torch.save(
            model.state_dict(),
            os.path.join(self._storage_info.model_version_path, "model.pt"),
        )
        if not isinstance(model, nn.Sequential):
            generate_model_py(
                model,
                os.path.join(
                    self._storage_info.model_version_path, "model.py"
                ),
                self._storage_info.embedding_studio_path,
            )
        else:
            generate_sequential_model_py(
                model,
                os.path.join(
                    self._storage_info.model_version_path, "model.py"
                ),
                self._storage_info.embedding_studio_path,
            )
