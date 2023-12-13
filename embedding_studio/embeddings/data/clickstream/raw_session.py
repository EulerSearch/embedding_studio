from typing import List, Optional

from pydantic import BaseModel

from embedding_studio.embeddings.data.clickstream.query_item import QueryItem
from embedding_studio.embeddings.data.clickstream.search_event import (
    SearchResult,
)
from embedding_studio.embeddings.data.clickstream.session import (
    ClickstreamSession,
)


class RawClickstreamSession(BaseModel):
    """Class that represents clickstream session.

    :param query: provided query.
    :type query: QueryItem
    :param results: search result info
    :type results: List[SearchResult]
    :param timestamp: when session was initialized
    :type timestamp:  Optional[int]
    """

    query: QueryItem
    results: List[SearchResult]
    timestamp: Optional[int] = None
    is_irrelevant: Optional[bool] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        self.is_irrelevant = len([r for r in self.results if r.is_click]) == 0

    def __len__(self) -> int:
        return len(self.results)

    @classmethod
    def from_dict(
        cls,
        data: dict,
        query_item_type: type,
        search_result_type: type,
        item_type: type,
        event_type: type,
    ) -> "RawClickstreamSession":
        return cls(
            query=query_item_type(**data["query"]),
            results=[
                search_result_type.from_dict(i, item_type, event_type)
                for i in data["results"]
            ],
            timestamp=int(data.get("timestamp")),
        )

    def get_session(self) -> ClickstreamSession:
        return ClickstreamSession(
            query=self.query,
            events=[r.item.id for r in self.results if r.is_click],
            results=[r.item.id for r in self.results],
            ranks={r.item.id: r.rank for r in self.results},
            event_types=[r.event_type.event_importance for r in self.results],
            timestamp=self.timestamp,
        )
