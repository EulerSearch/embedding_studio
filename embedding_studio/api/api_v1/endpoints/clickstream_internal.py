import logging

from fastapi import APIRouter, HTTPException, status

from embedding_studio.api.api_v1.schemas.clickstream_internal import (
    BatchReleaseRequest,
    BatchReleaseResponse,
    BatchSessionsGetResponse,
)
from embedding_studio.context.app_context import context

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/batch/sessions",
    status_code=status.HTTP_200_OK,
    response_model=BatchSessionsGetResponse,
)
def get_batch_sessions(
    batch_id: str,
    after_number: int = 0,
    limit: int = 10,
    events_limit: int = 100,
):
    logger.debug(
        f"Get batch sessions: "
        f"batch_id={batch_id}, "
        f"after_number={after_number}, "
        f"events_limit={events_limit}"
    )
    sessions = context.clickstream_dao.get_batch_sessions(
        batch_id=batch_id,
        after_number=after_number,
        limit=limit,
        events_limit=events_limit,
    )
    last_number = sessions[-1].session_number if sessions else None
    result = dict(
        batch_id=batch_id,
        last_number=last_number,
        sessions=[ses.model_dump() for ses in sessions],
    )
    logger.debug(f"Batch sessions got: {result}")
    return result


@router.post(
    "/batch/release",
    status_code=status.HTTP_200_OK,
    response_model=BatchReleaseResponse,
)
def release_batch(
    body: BatchReleaseRequest,
):
    logger.debug(f"Release batch: {body}")
    batch = context.clickstream_dao.release_batch(release_id=body.release_id)
    if not batch:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"There is no collecting batch",
        )
    logger.debug(f"Batch released: {batch}")
    return batch
