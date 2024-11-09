import os
from abc import ABC, abstractmethod
from typing import Dict, List

import torch
from torch import nn

from embedding_studio.inference_management.triton.model_storage_info import (
    ModelStorageInfo,
)
from embedding_studio.inference_management.triton.utils.types_mapping import (
    pytorch_dtype_to_triton_dtype,
)


class TritonModelStorageManager(ABC):
    def __init__(
        self, storage_info: ModelStorageInfo, do_dynamic_batching: bool = True
    ):
        self._storage_info = storage_info
        self._kind_gpu = torch.cuda.is_available()
        self.do_dynamic_batching = do_dynamic_batching

    @abstractmethod
    def _get_model_artifacts(self) -> List[str]:
        return []

    def is_model_deployed(self) -> bool:
        return all(
            os.path.exists(
                os.path.join(self._storage_info.model_version_path, artifact)
            )
            for artifact in self._get_model_artifacts()
        )

    def _setup_folder_directory(self):
        os.makedirs(self._storage_info.model_version_path, exist_ok=True)

    @abstractmethod
    def _generate_triton_config_model_info(self) -> List[str]:
        raise NotImplemented()

    def _generate_triton_config_model_input(
        self, example_inputs: Dict[str, torch.Tensor]
    ) -> List[str]:
        config_lines = ["input ["]
        for i, (input_name, example_tensor) in enumerate(
            example_inputs.items()
        ):
            input_dtype = pytorch_dtype_to_triton_dtype(example_tensor.dtype)
            config_lines.extend(
                [
                    "  {",
                    '    name: "{}"'.format(input_name),
                    "    data_type: {}".format(input_dtype),
                    "    dims: [{}]".format(
                        ", ".join(map(str, example_tensor.shape[1:]))
                    ),
                    "  }",
                ]
            )
            if i < len(example_inputs) - 1:
                config_lines[-1] += ","
        config_lines.append("]")
        return config_lines

    def _generate_triton_config_model_output(
        self,
        model: nn.Module,
        example_inputs: Dict[str, torch.Tensor],
        named_inputs: bool = False,
    ) -> List[str]:
        with torch.no_grad():
            if named_inputs:
                output = model(**example_inputs)
            else:
                output = model(*list(example_inputs.values()))
        config_lines = ["output ["]
        if isinstance(output, tuple):
            for idx, out_tensor in enumerate(output):
                out_dtype = pytorch_dtype_to_triton_dtype(out_tensor.dtype)
                config_lines.extend(
                    [
                        "  {",
                        '    name: "output{}"'.format(idx),
                        "    data_type: {}".format(out_dtype),
                        "    dims: [{}]".format(
                            ", ".join(map(str, out_tensor.shape[1:]))
                        ),
                        "  }",
                    ]
                )
                if idx < len(output):
                    config_lines[-1] += ","
        else:
            out_dtype = pytorch_dtype_to_triton_dtype(output.dtype)
            config_lines.extend(
                [
                    "  {",
                    '    name: "output"',
                    "    data_type: {}".format(out_dtype),
                    "    dims: [{}]".format(
                        ", ".join(map(str, output.shape[1:]))
                    ),
                    "  }",
                ]
            )
        config_lines.append("]")
        return config_lines

    def _generate_triton_config_inference_mode(self) -> List[str]:
        kind = "KIND_GPU" if self._kind_gpu else "KIND_CPU"
        gpus = list(range(torch.cuda.device_count())) if self._kind_gpu else []
        gpu_line = "\n    gpus: {}".format(gpus) if gpus else ""
        return [
            "instance_group [",
            "  {",
            f"    count: 1",
            f"    kind: {kind}{gpu_line}",
            "  }",
            "]",
        ]

    def _generate_triton_config_dynamic_batching(self) -> List[str]:
        if self.do_dynamic_batching:
            return [
                "dynamic_batching {",
                "  preferred_batch_size: [1, 2, 4, 8]",
                "  max_queue_delay_microseconds: 100",
                "  priority_levels: 3",
                " default_priority_level: 1",
                "}",
            ]
        return []

    def _generate_triton_config_model_versions(self) -> List[str]:
        return [
            "version_policy {",
            "  latest {",
            "    num_versions: 2",
            "  }",
            "}",
        ]

    def _generate_extra(self) -> List[str]:
        return [
            """parameters: {
key: "ENABLE_JIT_EXECUTOR"
    value: {
    string_value: "false"
    }
}"""
        ]

    def _generate_triton_config(
        self,
        model: nn.Module,
        example_inputs: Dict[str, torch.Tensor],
        named_inputs: bool = False,
    ) -> str:
        config_lines = self._generate_triton_config_model_info()
        config_lines += self._generate_triton_config_model_input(
            example_inputs
        )
        config_lines += self._generate_triton_config_model_output(
            model, example_inputs, named_inputs
        )
        config_lines += self._generate_triton_config_inference_mode()
        config_lines += self._generate_triton_config_dynamic_batching()
        config_lines += self._generate_triton_config_model_versions()
        config_lines += self._generate_extra()
        return "\n".join(config_lines)

    def _setup_triton_config(
        self,
        model: nn.Module,
        example_inputs: Dict[str, torch.Tensor],
        named_inputs: bool = False,
    ):
        model_config = self._generate_triton_config(
            model, example_inputs, named_inputs
        )
        with open(
            os.path.join(self._storage_info.model_path, "config.pbtxt"), "w"
        ) as f:
            f.write(model_config)

    @abstractmethod
    def _save_model(
        self, model: nn.Module, example_inputs: Dict[str, torch.Tensor]
    ):
        raise NotImplemented()

    def save_model(
        self,
        model: nn.Module,
        example_inputs: Dict[str, torch.Tensor],
        named_inputs: bool = False,
    ):
        if not self.is_model_deployed():
            self._setup_folder_directory()
            self._setup_triton_config(model, example_inputs, named_inputs)
            self._save_model(model, example_inputs)
