import logging
from typing import List, Optional

import pymongo

from embedding_studio.data_access.clickstream import ClickstreamDao
from embedding_studio.data_access.mongo.mongo_dao import MongoDao
from embedding_studio.models.clickstream.session_batches import (
    SessionBatch,
    SessionBatchStatus,
)
from embedding_studio.models.clickstream.session_events import (
    DbSessionEvent,
    SessionEvent,
)
from embedding_studio.models.clickstream.sessions import (
    RegisteredSession,
    Session,
    SessionWithEvents,
)
from embedding_studio.utils import datetime_utils

logger = logging.getLogger(__name__)


class MongoClickstreamDao(ClickstreamDao):
    _SESSIONS_COLLECTION: str = "sessions"
    _SESSION_EVENTS_COLLECTION: str = "session_events"
    _SESSION_BATCHES_COLLECTION: str = "session_batches"

    _SESSION_ID: str = "session_id"
    _SESSION_NUMBER: str = "session_number"
    _BATCH_ID: str = "batch_id"
    _EVENT_ID: str = "event_id"
    _CREATED_AT: str = "created_at"
    _STATUS: str = "status"
    _IS_IRRELEVANT: str = "is_irrelevant"
    _SESSION_COUNTER: str = "session_counter"
    _DB_ID: str = "db_id"
    _RELEASE_ID: str = "release_id"
    _RELEASED_AT: str = "released_at"

    _STATUS_COLLECTING: str = SessionBatchStatus.collecting.value
    _STATUS_RELEASED: str = SessionBatchStatus.released.value

    def __init__(self, mongo_database: pymongo.database.Database):
        self._session_dao = MongoDao[RegisteredSession](
            collection=mongo_database[self._SESSIONS_COLLECTION],
            model=RegisteredSession,
            model_id=self._SESSION_ID,
            additional_indexes=[
                dict(keys=self._BATCH_ID),
                dict(keys=self._SESSION_NUMBER),
                dict(keys=self._CREATED_AT),
            ],
        )
        self._event_dao = MongoDao[SessionEvent](
            collection=mongo_database[self._SESSION_EVENTS_COLLECTION],
            model=DbSessionEvent,
            model_id=self._DB_ID,
            model_mongo_id=self._DB_ID,
            additional_indexes=[
                dict(
                    keys=[self._SESSION_ID, self._EVENT_ID],
                    unique=True,
                ),
                dict(keys=self._CREATED_AT),
            ],
        )
        self._batch_dao = MongoDao[SessionBatch](
            collection=mongo_database[self._SESSION_BATCHES_COLLECTION],
            model=SessionBatch,
            model_id=self._BATCH_ID,
            model_mongo_id=self._BATCH_ID,
            additional_indexes=[
                dict(
                    keys=self._STATUS,
                    partialFilterExpression={
                        self._STATUS: self._STATUS_COLLECTING
                    },
                    unique=True,
                ),
                dict(
                    keys=self._RELEASE_ID,
                    partialFilterExpression={
                        self._RELEASE_ID: {"$exists": True},
                    },
                    unique=True,
                ),
                dict(keys=self._CREATED_AT),
            ],
        )

    def register_session(self, session: Session) -> RegisteredSession:
        batch = self._increment_session_batch()
        reg_session = RegisteredSession(
            batch_id=batch.batch_id,
            session_number=batch.session_counter,
            **session.model_dump(),
        )
        try:
            self._session_dao.insert_one(reg_session)
        except pymongo.errors.DuplicateKeyError:
            logger.warning(
                f"Session with session_id={session.session_id} already registered"
            )
            reg_session = self._session_dao.find_one(session.session_id)
        assert reg_session
        return reg_session

    def update_session(self, session: Session) -> RegisteredSession:
        batch = self._increment_session_batch()
        reg_session = RegisteredSession(
            batch_id=batch.batch_id,
            session_number=batch.session_counter,
            **session.model_dump(),
        )
        self._session_dao.update_one(reg_session)
        return reg_session

    def push_events(self, session_events: List[SessionEvent]) -> None:
        try:
            self._event_dao.insert_many(session_events, ordered=False)
        except pymongo.errors.BulkWriteError as err:
            logger.warning(
                f"Some errors occurred during events insertion: {err}"
            )

    def mark_session_irrelevant(
        self, session_id
    ) -> Optional[SessionWithEvents]:
        return self._session_dao.find_one_and_update(
            session_id,
            update={"$set": {self._IS_IRRELEVANT: True}},
            return_document=pymongo.collection.ReturnDocument.AFTER,
        )

    def get_session(self, session_id: str) -> Optional[SessionWithEvents]:
        session = self._session_dao.find_one(session_id)
        if not session:
            return None
        events = self._get_session_events(session_id=session_id)
        return SessionWithEvents(events=events, **session.model_dump())

    def get_batch_sessions(
        self,
        batch_id: str,
        after_number: Optional[int] = None,
        limit: Optional[int] = None,
        events_limit: Optional[int] = None,
    ) -> List[SessionWithEvents]:
        after_number = after_number or 0
        limit = limit or 0
        events_limit = events_limit or 0
        sessions = self._session_dao.find(
            sort_args=[self._SESSION_NUMBER],
            filter={
                self._BATCH_ID: batch_id,
                self._SESSION_NUMBER: {"$gt": after_number},
            },
            limit=limit,
        )
        return [
            SessionWithEvents(
                **session.model_dump(),
                events=self._get_session_events(
                    session_id=session.session_id, limit=events_limit
                ),
            )
            for session in sessions
        ]

    def get_batch(self, batch_id: str) -> Optional[SessionBatch]:
        return self._batch_dao.find_one(batch_id)

    def release_batch(self, release_id: str) -> Optional[SessionBatch]:
        batch: Optional[SessionBatch] = None
        try:
            batch = self._batch_dao.find_one_and_update(
                filter={self._STATUS: self._STATUS_COLLECTING},
                update={
                    "$set": {
                        self._RELEASE_ID: release_id,
                        self._STATUS: self._STATUS_RELEASED,
                        self._RELEASED_AT: datetime_utils.utc_timestamp(),
                    }
                },
                return_document=pymongo.collection.ReturnDocument.AFTER,
            )
        except pymongo.errors.DuplicateKeyError:
            logger.warning(f"Batch already released: release_id={release_id}")
        if batch is None:
            batch = self._batch_dao.find_one(
                filter={self._RELEASE_ID: release_id}
            )
        return batch

    def update_batch_status(
        self, batch_id: str, status: SessionBatchStatus
    ) -> Optional[SessionBatch]:
        return self._batch_dao.find_one_and_update(
            batch_id,
            update={"$set": {self._STATUS: status.value}},
            return_document=pymongo.collection.ReturnDocument.AFTER,
        )

    def _get_session_events(
        self, session_id, limit: int = 100
    ) -> List[SessionEvent]:
        return self._event_dao.find(
            filter={self._SESSION_ID: session_id}, limit=limit
        )

    def _increment_session_batch(self) -> SessionBatch:
        return self._batch_dao.find_one_and_update(
            filter={self._STATUS: self._STATUS_COLLECTING},
            update={
                "$inc": {self._SESSION_COUNTER: 1},
                "$setOnInsert": {
                    self._CREATED_AT: datetime_utils.utc_timestamp()
                },
            },
            return_document=pymongo.collection.ReturnDocument.AFTER,
            upsert=True,
        )
