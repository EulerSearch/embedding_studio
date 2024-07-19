from abc import ABC
from typing import Dict, Type, Union

from embedding_studio.clickstream_storage.input_with_items import (
    FineTuningInputWithItems,
)
from embedding_studio.clickstream_storage.query_item import QueryItem
from embedding_studio.clickstream_storage.search_event import (
    DummySessionEventWithImportance,
    SessionEventWithImportance,
)
from embedding_studio.clickstream_storage.text_query_item import TextQueryItem
from embedding_studio.data_storage.loaders.item_meta import ItemMeta
from embedding_studio.embeddings.features.feature_extractor_input import (
    FineTuningInput,
)
from embedding_studio.models.clickstream.sessions import SessionWithEvents


class ClickstreamSessionConverter(ABC):
    def __init__(
        self,
        item_type: Type[ItemMeta],
        query_item_type: Type[QueryItem] = TextQueryItem,
        fine_tuning_type: Type[FineTuningInput] = FineTuningInput,
        event_type: Type[
            SessionEventWithImportance
        ] = DummySessionEventWithImportance,
    ):
        self.item_type = item_type
        self.query_item_type = query_item_type
        self.fine_tuning_type = fine_tuning_type
        self.event_type = event_type

    def convert(
        self, session_data: Union[SessionWithEvents, Dict]
    ) -> FineTuningInputWithItems:
        obj = None
        if isinstance(session_data, dict):
            obj = SessionWithEvents(**session_data)
        else:
            obj = session_data

        items = [
            self.item_type(obect_id=result.object_id, **result.meta)
            for result in obj.search_results
        ]
        events = [self.event_type.from_model(event) for event in obj.events]
        ranks = {event.object_id: event.event_importance for event in events}
        event_types = [event.event_importance for event in events]
        event_ids = set(ranks.keys())
        return FineTuningInputWithItems(
            input=self.fine_tuning_type(
                query=self.query_item_type(
                    text=obj.search_query, **obj.search_meta
                ),
                events=[
                    result.object_id
                    for result in obj.search_results
                    if (result.object_id in event_ids)
                ],
                results=[result.object_id for result in obj.search_results],
                ranks=ranks,
                event_types=event_types,
                timestamp=obj.created_at,
            ),
            items=items,
        )
