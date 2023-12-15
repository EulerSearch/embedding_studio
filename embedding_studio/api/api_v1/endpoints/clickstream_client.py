import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, status

from embedding_studio.api.api_v1.schemas.clickstream_client import (
    SessionAddEventsRequest,
    SessionCreateRequest,
    SessionGetResponse,
    SessionMarkIrrelevantRequest,
)
from embedding_studio.context.app_context import context
from embedding_studio.core.config import settings
from embedding_studio.models.clickstream.session_events import SessionEvent
from embedding_studio.models.clickstream.sessions import (
    Session,
    SessionWithEvents,
)
from embedding_studio.utils import datetime_utils

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/session",
    status_code=status.HTTP_200_OK,
)
def create_session(
    body: SessionCreateRequest,
) -> None:
    logger.debug(f"Register session: {body}")
    body.created_at = _ensure_timestamp(body.created_at)
    session = Session.model_validate(body.model_dump())
    reg_session = context.clickstream_dao.register_session(session)
    logger.debug(f"Session registered: {reg_session}")


@router.get(
    "/session",
    status_code=status.HTTP_200_OK,
    response_model=SessionGetResponse,
)
def get_session(session_id: str) -> SessionWithEvents:
    logger.debug(f"Get session by session_id={session_id}")
    session = context.clickstream_dao.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session with id={session_id} not found",
        )
    logger.debug(f"Found session: {session}")
    return session


@router.post(
    "/session/events",
    status_code=status.HTTP_200_OK,
)
def push_events(body: SessionAddEventsRequest) -> None:
    logger.debug(f"Push session events: body")
    session_id = body.session_id
    events = [
        SessionEvent.model_validate(
            dict(
                session_id=session_id,
                created_at=_ensure_timestamp(event.created_at),
                **event.model_dump(
                    exclude={"created_at"},
                ),
            ),
        )
        for event in body.events
    ]
    context.clickstream_dao.push_events(events)


@router.post(
    "/session/irrelevant",
    status_code=status.HTTP_200_OK,
)
def mark_session_irrelevant(body: SessionMarkIrrelevantRequest) -> None:
    logger.debug(f"Mark irrelevant session: {body}")
    session = context.clickstream_dao.mark_session_irrelevant(
        session_id=body.session_id
    )
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session with id={body.session_id} not found",
        )
    logger.debug(f"Irrelevant session marked: {session}")


def _ensure_timestamp(request_timestamp: Optional[int]) -> int:
    if request_timestamp is None:
        return datetime_utils.utc_timestamp()
    timestamp_ok = datetime_utils.check_utc_timestamp(
        request_timestamp,
        delta_minus_sec=settings.CLICKSTREAM_TIME_MAX_DELTA_MINUS_SEC,
        delta_plus_sec=settings.CLICKSTREAM_TIME_MAX_DELTA_PLUS_SEC,
    )
    if not timestamp_ok:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid utc timestamp: {request_timestamp}",
        )
    return request_timestamp
