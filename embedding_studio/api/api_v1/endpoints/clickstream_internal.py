import logging

from fastapi import APIRouter, HTTPException, status

from embedding_studio.api.api_v1.schemas.clickstream_internal import (
    BatchReleaseRequest,
    BatchReleaseResponse,
    BatchSessionsGetResponse,
    UseSessionForImprovementRequest,
)
from embedding_studio.context.app_context import context
from embedding_studio.models.improvement import (
    SessionForImprovementCreateSchema,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/session/use-for-improvement",
    status_code=status.HTTP_200_OK,
)
def use_session_for_improvement(body: UseSessionForImprovementRequest) -> None:
    """
    Submits a session for use in search quality improvement processes.

    Validates the session exists and contains events, then schedules it for
    inclusion in model improvement tasks. Skips payload-search sessions as
    they're not suitable for training.
    """
    logger.debug(f"Push session to be used for improvement: {body}")
    session = context.clickstream_dao.get_session(session_id=body.session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session with id={body.session_id} not found",
        )

    if session.is_payload_search:
        logger.warning(
            f"Session with id {body.session_id} is payload search, so can't be used."
        )

    if session.is_irrelevant or len(session.events) > 0:
        task = context.sessions_for_improvement.create(
            schema=SessionForImprovementCreateSchema(
                session_id=body.session_id,
            ),
            return_obj=True,
        )
        context.sessions_for_improvement.update(obj=task)

        logger.debug(f"Session has been pushed: {session.session_id}")
    else:
        logger.debug(f"No events, session has not been pushed: {session}")


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
    """
    Retrieves a paginated batch of sessions for processing.

    Supports efficient iteration through large session collections with
    pagination and event limiting. Returns session data ready for batch
    processing in model improvement workflows.
    """
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
    """
    Marks a batch of sessions as processed and ready for deployment.

    Finalizes a batch processing operation, making processed data available
    for downstream consumers. Returns 404 if no active collection batch exists.
    """
    logger.debug(f"Release batch: {body}")
    batch = context.clickstream_dao.release_batch(release_id=body.release_id)
    if not batch:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"There is no collecting batch",
        )
    logger.debug(f"Batch released: {batch}")
    return batch
