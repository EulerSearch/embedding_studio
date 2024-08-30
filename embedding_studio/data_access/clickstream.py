from abc import ABC, abstractmethod
from typing import List, Optional

from embedding_studio.models.clickstream.session_batches import (
    SessionBatch,
    SessionBatchStatus,
)
from embedding_studio.models.clickstream.session_events import SessionEvent
from embedding_studio.models.clickstream.sessions import (
    RegisteredSession,
    Session,
    SessionWithEvents,
)


class ClickstreamDao(ABC):
    @abstractmethod
    def register_session(self, session: Session) -> RegisteredSession:
        """Register new click stream session.
        Nothing will change if session with the specified id already registered

        :param session: new session
        :return: registered session (with batch id and number)
        """
        raise NotImplementedError()

    @abstractmethod
    def update_session(self, session: Session) -> RegisteredSession:
        """Update click stream session.

        :param session: updating session
        :return: updated session (with batch id and number)
        """
        raise NotImplementedError()

    @abstractmethod
    def push_events(self, events: List[SessionEvent]) -> None:
        """Push new session events
        Nothing will change for any event that already exists

        :param events: session events
        """
        raise NotImplementedError()

    @abstractmethod
    def mark_session_irrelevant(self, session_id: str) -> RegisteredSession:
        """Mark click stream session as irrelevant

        :param session_id: session id
        :return: registered session
        """
        raise NotImplementedError()

    @abstractmethod
    def get_session(self, session_id: str) -> Optional[SessionWithEvents]:
        """Get registered click stream session with events

        :param session_id: session id
        :return: session with events or None (if not found)
        """
        raise NotImplementedError()

    @abstractmethod
    def get_batch_sessions(
        self,
        batch_id: str,
        after_number: Optional[int] = None,
        limit: Optional[int] = None,
        events_limit: Optional[int] = None,
    ) -> List[SessionWithEvents]:
        """Get registered sessions with events by batch_id

        :param batch_id: session batch id
        :param after_number: sessions with less session_number will be skipped (including this number)
        :param limit: max length of returning session list
        :param events_limit: max event list length in each returning session
        :return: list of found sessions with events
        """
        raise NotImplementedError()

    @abstractmethod
    def get_batch(self, batch_id: str) -> Optional[SessionBatch]:
        """Get session batch

        :param batch_id: batch id
        :return: session batch or None (if not found)
        """
        raise NotImplementedError()

    @staticmethod
    def release_batch(self, release_id: str) -> Optional[SessionBatch]:
        """Release current collecting batch

        :param release_id: unique release id (this is operation idempotency key)
        :return: released session batch or None if batch doesn't exist
        """
        raise NotImplementedError()

    @abstractmethod
    def update_batch_status(
        self, batch_id: str, status: SessionBatchStatus
    ) -> Optional[SessionBatch]:
        """Update batch status by batch id

        :param batch_id: batch id
        :param status: new status
        :return: updated session batch or None (if not found)
        """
        raise NotImplementedError()
