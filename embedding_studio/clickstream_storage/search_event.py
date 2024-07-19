from abc import ABC, abstractmethod

from pydantic import BaseModel

from embedding_studio.models.clickstream.session_events import SessionEvent


class SessionEventWithImportance(ABC, BaseModel):
    object_id: str

    @property
    @abstractmethod
    def event_importance(self) -> float:
        raise NotImplemented()

    @classmethod
    @abstractmethod
    def from_model(cls, event: SessionEvent) -> "SessionEventWithImportance":
        pass


class DummySessionEventWithImportance(SessionEventWithImportance):
    importance: float = 1.0

    @property
    def event_importance(self) -> float:
        return self.importance

    def from_model(
        cls, event: SessionEvent
    ) -> "DummySessionEventWithImportance":
        return DummySessionEventWithImportance(
            object_id=event.object_id,
            importance=event.meta.get("importance", 1.0),
        )
