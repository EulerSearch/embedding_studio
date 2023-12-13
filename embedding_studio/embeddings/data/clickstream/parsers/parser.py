from typing import Dict

from embedding_studio.embeddings.data.clickstream.raw_session import (
    RawClickstreamSession,
)


class ClickstreamParser(object):
    # TODO: annotate types precisely
    def __init__(
        self,
        query_item_type: type,
        search_result_type: type,
        meta_type: type,
        event_type: type,
    ):
        self.query_item_type = query_item_type
        self.search_result_type = search_result_type
        self.meta_type = meta_type
        self.event_type = event_type

    def parse(self, session_data: Dict) -> RawClickstreamSession:
        return RawClickstreamSession.from_dict(
            session_data,
            self.query_item_type,
            self.search_result_type,
            self.meta_type,
            self.event_type,
        )
