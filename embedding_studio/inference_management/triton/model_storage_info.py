import os
import re
from typing import Optional

from pydantic import BaseModel, FieldValidationInfo, field_validator

ARCHIVED_VERSION = "_archived"
# Python identifier regex
IDENTIFIER_REGEX = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class DeployedModelInfo(BaseModel):
    """
    Represents information about a model deployed in the Triton Inference Server.

    :param plugin_name: Name of the plugin that owns this model. Must be a valid Python identifier.
    :param model_type: Type of the model (e.g., "query", "reranker").
    :param embedding_model_id: Optional identifier for the specific embedding model.
    :param version: Version string for the model. Must be either numeric or the archived version string.
    :return: A DeployedModelInfo object containing model deployment information.
    """

    plugin_name: str = "BasePlugin"
    model_type: str = "query"
    embedding_model_id: Optional[str] = None
    version: str = "1"

    @property
    def name(self) -> str:
        """
        Generates a standardized name for the deployed model based on its attributes.

        :return: A string representing the complete model name in the format '{plugin_name}_{embedding_model_id}_{model_type}'
                 or '{plugin_name}_{model_type}' if embedding_model_id is None.
        """
        if self.embedding_model_id is None:
            return f"{self.plugin_name}_{self.model_type}"
        else:
            return f"{self.plugin_name}_{self.embedding_model_id}_{self.model_type}"

    @field_validator("plugin_name")
    def validate_plugin_name(
        cls, value: str, info: FieldValidationInfo
    ) -> str:
        """
        Validates that the plugin name follows Python identifier naming rules.

        :param value: The plugin name to validate.
        :param info: Validation context information.
        :return: The validated plugin name.
        :raises ValueError: If the plugin name doesn't match the required pattern.
        """
        if not IDENTIFIER_REGEX.match(value):
            raise ValueError(
                f"Invalid plugin_name '{value}'. Names must start with a letter or underscore, "
                "and can only contain letters, digits, and underscores."
            )
        return value

    @field_validator("version")
    def validate_version(cls, value: str, info: FieldValidationInfo) -> str:
        """
        Validates that the version is either a digit or the archived version name.

        :param value: The version string to validate.
        :param info: Validation context information.
        :return: The validated version string.
        :raises ValueError: If the version is neither a digit nor the archived version name.
        """
        if not (value.isdigit() or value == cls.archived_version_name()):
            raise ValueError(
                f"Invalid version '{value}'. Version should be either digit either {cls.archived_version_name()}."
            )
        return value


class ModelStorageInfo(BaseModel):
    """
    Contains information about where and how a model is stored in the Triton Inference Server.

    :param model_repo: The base directory path for all models in the Triton repository.
    :param embedding_studio_path: The path to the embedding_studio module for imports.
    :param deployed_model_info: Information about the deployed model.
    :return: A ModelStorageInfo object with paths and information for model storage.
    """

    model_repo: str = "/models"
    embedding_studio_path: str = "/embedding_studio"
    deployed_model_info: DeployedModelInfo = DeployedModelInfo()

    @classmethod
    def archived_version_name(self) -> str:
        """
        Returns the string identifier used for archived model versions.

        :return: The constant string used to mark archived versions.
        """
        return ARCHIVED_VERSION

    @property
    def model_name(self) -> str:
        """
        Gets the full model name derived from the deployed model info.

        :return: The complete model name for Triton Inference Server.
        """
        return self.deployed_model_info.name

    @property
    def model_path(self) -> str:
        """
        Gets the absolute path to the model directory in the Triton model repository.

        :return: The path to the model directory.
        """
        return os.path.join(self.model_repo, self.model_name)

    @property
    def model_version_path(self) -> str:
        """
        Gets the absolute path to the specific version directory of the model.

        :return: The path to the specific model version directory.
        """
        return os.path.join(self.model_path, self.deployed_model_info.version)
