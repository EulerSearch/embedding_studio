import os
import re
from typing import Optional

from pydantic import BaseModel, FieldValidationInfo, field_validator

ARCHIVED_VERSION = "_archived"
# Python identifier regex
IDENTIFIER_REGEX = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class DeployedModelInfo(BaseModel):
    plugin_name: str = "BasePlugin"
    model_type: str = "query"
    embedding_model_id: Optional[str] = None
    version: str = "1"

    @property
    def name(self) -> str:
        if self.embedding_model_id is None:
            return f"{self.plugin_name}_{self.model_type}"
        else:
            return f"{self.plugin_name}_{self.embedding_model_id}_{self.model_type}"

    @field_validator("plugin_name")
    def validate_plugin_name(
        cls, value: str, info: FieldValidationInfo
    ) -> str:
        if not IDENTIFIER_REGEX.match(value):
            raise ValueError(
                f"Invalid plugin_name '{value}'. Names must start with a letter or underscore, "
                "and can only contain letters, digits, and underscores."
            )
        return value

    @field_validator("version")
    def validate_version(cls, value: str, info: FieldValidationInfo) -> str:
        if not (value.isdigit() or value == cls.archived_version_name()):
            raise ValueError(
                f"Invalid version '{value}'. Version should be either digit either {cls.archived_version_name()}."
            )
        return value


class ModelStorageInfo(BaseModel):
    model_repo: str = "/models"
    embedding_studio_path: str = "/embedding_studio"
    deployed_model_info: DeployedModelInfo = DeployedModelInfo()

    @classmethod
    def archived_version_name(self) -> str:
        return ARCHIVED_VERSION

    @property
    def model_name(self) -> str:
        return self.deployed_model_info.name

    @property
    def model_path(self) -> str:
        return os.path.join(self.model_repo, self.model_name)

    @property
    def model_version_path(self) -> str:
        return os.path.join(self.model_path, self.deployed_model_info.version)
