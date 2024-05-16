import os
from typing import List

import torch
from torch import nn

from embedding_studio.inference_management.triton.manager import (
    TritonModelStorageManager,
)


class JitTraceTritonModelStorageManager(TritonModelStorageManager):
    def _get_model_artifacts(self) -> List[str]:
        return ["model.pt"]

    def _generate_triton_config_model_info(self) -> List[str]:
        return [
            'name: "{}"'.format(self._storage_info.model_name),
            'platform: "pytorch_libtorch"',
            "max_batch_size: 16",
        ]

    def _save_model(
        self, model: nn.Module, example_input: torch.Tensor, input_name: str
    ):
        traced_model = torch.jit.trace(model, example_input)
        torch.jit.save(
            traced_model,
            os.path.join(self._storage_info.model_version_path, "model.pt"),
        )
