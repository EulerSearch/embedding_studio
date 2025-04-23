from abc import ABC, abstractmethod

from pydantic import BaseModel

from embedding_studio.models.clickstream.session_events import SessionEvent


class SessionEventWithImportance(ABC, BaseModel):
    """
    Abstract base class for session events with importance scoring.

    Combines Pydantic's BaseModel with ABC for abstract methods,
    providing a foundation for events that include importance metrics.

    :param object_id: Unique identifier for the object associated with the event
    """

    object_id: str

    @property
    @abstractmethod
    def event_importance(self) -> float:
        """
        Get the importance score for this event.

        :return: A float representing the event's importance

        Example implementation:
        ```python
        @property
        def event_importance(self) -> float:
            return self.score * self.weight
        ```
        """
        raise NotImplemented()

    @classmethod
    @abstractmethod
    def from_model(cls, event: SessionEvent) -> "SessionEventWithImportance":
        """
        Create an instance from a SessionEvent model.

        :param event: The SessionEvent to convert
        :return: A new instance of SessionEventWithImportance

        Example implementation:
        ```python
        @classmethod
        def from_model(cls, event: SessionEvent) -> "CustomSessionEvent":
            return cls(
                object_id=event.object_id,
                score=event.meta.get("score", 0.5)
            )
        ```
        """


class DummySessionEventWithImportance(SessionEventWithImportance):
    """
    Concrete implementation of SessionEventWithImportance for testing or default use.

    Provides a simple importance score implementation for session events.

    :param object_id: Unique identifier for the object associated with the event
    :param importance: The importance score for this event, defaults to 1.0
    """

    importance: float = 1.0

    @property
    def event_importance(self) -> float:
        """
        Get the importance score for this event.

        :return: The importance value as a float
        """
        return self.importance

    def from_model(
        cls, event: SessionEvent
    ) -> "DummySessionEventWithImportance":
        """
        Create an instance from a SessionEvent model.

        :param event: The SessionEvent to convert
        :return: A new DummySessionEventWithImportance instance
        """
        return DummySessionEventWithImportance(
            object_id=event.object_id,
            importance=event.meta.get("importance", 1.0),
        )
