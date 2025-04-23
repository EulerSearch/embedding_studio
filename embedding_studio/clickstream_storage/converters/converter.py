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
from embedding_studio.embeddings.features.fine_tuning_input import (
    FineTuningInput,
)
from embedding_studio.models.clickstream.sessions import SessionWithEvents


class ClickstreamSessionConverter:
    """
    Base class for converting clickstream session data to fine-tuning inputs.

    Handles the transformation of session data with search events into structured
    inputs for model fine-tuning, along with associated metadata items.

    :param item_type: The class type for item metadata
    :param query_item_type: The class type for query items, defaults to TextQueryItem
    :param fine_tuning_type: The class type for fine-tuning inputs, defaults to FineTuningInput
    :param event_type: The class type for session events with importance, defaults to DummySessionEventWithImportance
    """

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
        """
        Convert session data to a fine-tuning input with associated items.

        :param session_data: Session data as either a SessionWithEvents object or a dictionary
        :return: A FineTuningInputWithItems containing the converted data
        """
        # Handle different input types - convert dict to SessionWithEvents if needed
        obj = None
        if isinstance(session_data, dict):
            obj = SessionWithEvents(**session_data)
        else:
            obj = session_data

        # Create item metadata objects from search results
        items = [
            self.item_type(obect_id=result.object_id, **result.meta)
            for result in obj.search_results
        ]

        # Convert session events to the specified event type
        events = [self.event_type.from_model(event) for event in obj.events]

        # Create a mapping of object IDs to importance scores
        ranks = {event.object_id: event.event_importance for event in events}

        # Extract importance scores as a separate list
        event_types = [event.event_importance for event in events]

        # Create a set of object IDs that have associated events
        event_ids = set(ranks.keys())

        # Create and return the FineTuningInputWithItems object
        return FineTuningInputWithItems(
            input=self.fine_tuning_type(
                # Create a query item from the search query and metadata
                query=self.query_item_type(
                    text=obj.search_query, **obj.search_meta
                ),
                # Include only results that have associated events
                events=[
                    result.object_id
                    for result in obj.search_results
                    if (result.object_id in event_ids)
                ],
                # Include all search results
                results=[result.object_id for result in obj.search_results],
                # Include the importance score mapping
                ranks=ranks,
                # Include the list of importance scores
                event_types=event_types,
                # Use the session creation timestamp
                timestamp=obj.created_at,
            ),
            # Include the item metadata objects
            items=items,
        )
