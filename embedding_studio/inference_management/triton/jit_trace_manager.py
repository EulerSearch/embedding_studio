import os
from typing import Dict, List

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
        self, model: nn.Module, example_inputs: Dict[str, torch.Tensor]
    ):
        # Assumes model is prepared to accept inputs as a dictionary when traced
        traced_model = torch.jit.trace(model, tuple(example_inputs.values()))
        torch.jit.save(
            traced_model,
            os.path.join(self._storage_info.model_version_path, "model.pt"),
        )
