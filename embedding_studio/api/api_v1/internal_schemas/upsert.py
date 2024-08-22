from typing import List

from pydantic import Field

from embedding_studio.api.api_v1.internal_schemas.task import (
    BaseInternalTaskRequest,
    BaseInternalTaskResponse,
)
from embedding_studio.api.api_v1.schemas.upsert import DataItem, FailedDataItem


class UpsertionTaskRunRequest(BaseInternalTaskRequest):
    items: List[DataItem] = Field(...)


class UpsertionTaskResponse(BaseInternalTaskResponse):
    failed_items: List[FailedDataItem] = Field(...)
