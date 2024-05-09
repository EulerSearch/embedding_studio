import os
import re

from pydantic import BaseModel, FieldValidationInfo, field_validator

ARCHIVED_VERSION = "_archived"


class ModelStorageInfo(BaseModel):
    model_repo: str = "/models"
    embedding_studio_path: str = "/embedding_studio"
    plugin_name: str = "BasePlugin"
    model_type: str = "query"
    version: str = "1"

    @classmethod
    def archived_version_name(self) -> str:
        return ARCHIVED_VERSION

    @property
    def model_name(self) -> str:
        return f"{self.plugin_name}_{self.model_type}"

    @property
    def model_path(self) -> str:
        return os.path.join(self.model_repo, self.model_name)

    @property
    def model_version_path(self) -> str:
        return os.path.join(self.model_path, self.version)

    @field_validator("plugin_name")
    def validate_plugin_name(
        cls, value: str, info: FieldValidationInfo
    ) -> str:
        # Python identifier regex
        identifier_regex = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
        if not identifier_regex.match(value):
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
