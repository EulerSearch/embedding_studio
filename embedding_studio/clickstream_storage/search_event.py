from abc import ABC, abstractmethod
from typing import Dict, Optional, Set

from pydantic import BaseModel, validator

from embedding_studio.data_storage.loaders.item_meta import ItemMeta
from embedding_studio.models.clickstream.sessions import SearchResultItem


class EventType(ABC, BaseModel):
    @property
    @abstractmethod
    def event_importance(self) -> float:
        raise NotImplemented()


class DummyEventType(EventType):
    importance: float

    @property
    def event_importance(self) -> float:
        return self.importance


class SearchResult(BaseModel):
    item: ItemMeta
    is_click: bool
    rank: Optional[float] = None
    event_type: Optional[EventType] = None
    timestamp: Optional[int] = None

    @validator("event_type", pre=True, always=True)
    def validate_event_type(cls, value, values):
        if value is not None and not isinstance(value, EventType):
            raise ValueError("Invalid event_type instance")
        return value

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_mongo(
        cls,
        result: SearchResultItem,
        event_ids: Set[str],
        item_type: type,
        event_type: type,
    ) -> "SearchResult":
        event_instance = DummyEventType(importance=1)

        return cls(
            item=item_type(**result.meta),
            is_click=result.object_id in event_ids,
            event_type=event_instance,
            timestamp=None,
        )

    @classmethod
    def from_dict(
        cls, data: dict, item_type: type, event_type: type
    ) -> "SearchResult":
        event_data: Optional[Dict] = data.get("event_type")
        event_instance = None

        if event_data is not None:
            event_instance = event_type(**event_data)

        return cls(
            item=item_type(**data["item"]),
            is_click=data["is_click"],
            rank=data["rank"],
            event_type=event_instance,
            timestamp=int(data.get("timestamp")),
        )
