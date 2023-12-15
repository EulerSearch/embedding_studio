from typing import List, Optional

from pydantic import BaseModel

from embedding_studio.api.api_v1.schemas.clickstream_client import (
    SessionGetResponse,
)


class BatchSession(SessionGetResponse):
    session_number: int


class BatchSessionsGetResponse(BaseModel):
    batch_id: str
    last_number: Optional[int]
    sessions: List[BatchSession]


class BatchReleaseRequest(BaseModel):
    release_id: str


class BatchReleaseResponse(BaseModel):
    release_id: str
    batch_id: str
    released_at: int
