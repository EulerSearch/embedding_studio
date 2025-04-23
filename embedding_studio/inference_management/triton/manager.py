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
    """
    Abstract base class for managing the storage of models in Triton Inference Server.

    This class provides a framework for preparing, saving, and configuring models
    for deployment with Triton Inference Server, handling details like directory
    structure, configuration files, and model serialization.

    :param storage_info: Information about where and how the model should be stored.
    :param do_dynamic_batching: Whether to enable dynamic batching for the model.
    :return: A manager for handling model storage operations.
    """

    def __init__(
        self, storage_info: ModelStorageInfo, do_dynamic_batching: bool = True
    ):
        """
        Initializes a new TritonModelStorageManager.

        :param storage_info: Information about the model storage location and naming.
        :param do_dynamic_batching: Whether to enable dynamic batching for the model.
        """
        self._storage_info = storage_info
        self._kind_gpu = torch.cuda.is_available()
        self.do_dynamic_batching = do_dynamic_batching

    @abstractmethod
    def _get_model_artifacts(self) -> List[str]:
        """
        Returns a list of expected artifact filenames for the model.

        This method must be implemented by subclasses to define which files
        should be present for a complete model deployment.

        :return: A list of required artifact filenames.

        Example implementation:
        ```python
        def _get_model_artifacts(self) -> List[str]:
            return ["model.pt", "config.json"]
        ```
        """
        return []

    def is_model_deployed(self) -> bool:
        """
        Checks if the model is already deployed by verifying the existence of all required artifacts.

        :return: True if all required artifacts exist in the model version path, False otherwise.
        """
        return all(
            os.path.exists(
                os.path.join(self._storage_info.model_version_path, artifact)
            )
            for artifact in self._get_model_artifacts()
        )

    def _setup_folder_directory(self):
        """
        Creates the necessary directory structure for the model.

        :return: None
        """
        os.makedirs(self._storage_info.model_version_path, exist_ok=True)

    @abstractmethod
    def _generate_triton_config_model_info(self) -> List[str]:
        """
        Generates the model information section of the Triton configuration.

        This method must be implemented by subclasses to define model-specific
        configuration settings like name, platform, and max batch size.

        :return: A list of configuration lines for the model information section.

        Example implementation:
        ```python
        def _generate_triton_config_model_info(self) -> List[str]:
            return [
                'name: "{}"'.format(self._storage_info.model_name),
                'platform: "pytorch_libtorch"',
                "max_batch_size: 16",
            ]
        ```
        """
        raise NotImplemented()

    def _generate_triton_config_model_input(
        self, example_inputs: Dict[str, torch.Tensor]
    ) -> List[str]:
        """
        Generates the input section of the Triton configuration based on example inputs.

        :param example_inputs: Dictionary mapping input names to example tensors.
        :return: A list of configuration lines for the model inputs section.
        """
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
        """
        Generates the output section of the Triton configuration by running the model with example inputs.

        :param model: The PyTorch model to analyze.
        :param example_inputs: Dictionary mapping input names to example tensors.
        :param named_inputs: Whether to pass inputs to the model as named arguments.
        :return: A list of configuration lines for the model outputs section.
        """
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
        """
        Generates the instance group section of the Triton configuration for specifying execution resources.

        :return: A list of configuration lines for the inference mode section.
        """
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
        """
        Generates the dynamic batching section of the Triton configuration if enabled.

        :return: A list of configuration lines for the dynamic batching section, or an empty list if disabled.
        """
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
        """
        Generates the version policy section of the Triton configuration.

        :return: A list of configuration lines for the model version policy section.
        """
        return [
            "version_policy {",
            "  latest {",
            "    num_versions: 2",
            "  }",
            "}",
        ]

    def _generate_extra(self) -> List[str]:
        """
        Generates additional configuration parameters for the Triton configuration.

        :return: A list of configuration lines for any extra parameters.
        """
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
        """
        Generates the complete Triton configuration by combining all configuration sections.

        :param model: The PyTorch model to configure.
        :param example_inputs: Dictionary mapping input names to example tensors.
        :param named_inputs: Whether to pass inputs to the model as named arguments.
        :return: The complete Triton configuration as a string.
        """
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
        """
        Creates and writes the Triton configuration file for the model.

        :param model: The PyTorch model to configure.
        :param example_inputs: Dictionary mapping input names to example tensors.
        :param named_inputs: Whether to pass inputs to the model as named arguments.
        :return: None
        """
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
        """
        Saves the model in a format compatible with Triton Inference Server.

        This method must be implemented by subclasses to handle the specific
        serialization requirements of different model types.

        :param model: The PyTorch model to save.
        :param example_inputs: Dictionary mapping input names to example tensors.
        :return: None

        Example implementation:
        ```python
        def _save_model(self, model: nn.Module, example_inputs: Dict[str, torch.Tensor]):
            model_path = os.path.join(self._storage_info.model_version_path, "model.pt")
            torch.save(model.state_dict(), model_path)
        ```
        """
        raise NotImplemented()

    def save_model(
        self,
        model: nn.Module,
        example_inputs: Dict[str, torch.Tensor],
        named_inputs: bool = False,
    ):
        """
        Sets up the model directory and saves the model and its configuration if not already deployed.

        This is the main public method for deploying a model to Triton.

        :param model: The PyTorch model to deploy.
        :param example_inputs: Dictionary mapping input names to example tensors.
        :param named_inputs: Whether to pass inputs to the model as named arguments.
        :return: None
        """
        if not self.is_model_deployed():
            self._setup_folder_directory()
            self._setup_triton_config(model, example_inputs, named_inputs)
            self._save_model(model, example_inputs)
