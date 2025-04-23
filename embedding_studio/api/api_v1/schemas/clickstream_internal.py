from typing import List, Optional

from pydantic import BaseModel

from embedding_studio.api.api_v1.schemas.clickstream_client import (
    SessionGetResponse,
)


class UseSessionForImprovementRequest(BaseModel):
    """
    Flags a specific session for inclusion in the search improvement process.
    Initiates the workflow to incorporate session data into quality enhancements.
    Provides the mechanism to feed real user interactions back into the system.
    Critical bridge between user behavior collection and search improvement.
    """

    session_id: str


class BatchSession(SessionGetResponse):
    """
    Extends SessionGetResponse with sequential numbering for batch processing.
    Maintains order within batches of sessions being processed for improvements.
    Enables efficient pagination and processing of large session datasets.
    Supports systematic analysis of user interactions across multiple sessions.
    """

    session_number: int


class BatchSessionsGetResponse(BaseModel):
    """
    Container for a paginated collection of sessions within a processing batch.
    Supports efficient retrieval and processing of session data in manageable chunks.
    Enables systematic analysis of search quality across multiple user sessions.
    Facilitates organized feedback loops for continuous search improvement.
    """

    batch_id: str
    last_number: Optional[int]
    sessions: List[BatchSession]


class BatchReleaseRequest(BaseModel):
    """
    Request to mark a batch of sessions as released for processing.
    Initiates the workflow to incorporate batched session data into fine-tuning and reindexing.
    Controls the flow of feedback data into the search optimization pipeline.
    Provides versioning mechanism for tracking deployment of improvements.
    """

    release_id: str


class BatchReleaseResponse(BaseModel):
    """
    Confirmation of a successfully released batch with relevant metadata.
    Provides audit trail for when session data was incorporated into improvements.
    Enables tracking the lineage of search quality enhancements over time.
    Supports monitoring and evaluation of the improvement feedback loop.
    """

    release_id: str
    batch_id: str
    released_at: int
