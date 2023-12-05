from typing import List

from embedding_studio.embeddings.data.clickstream.query_item import QueryItem
from embedding_studio.embeddings.data.clickstream.raw_session import (
    ClickstreamSession,
)


class QueryRetriever(object):
    """As we can't exactly predict a schema of storing queries:
    1. As text exceptly in clickstream service
    2. As ID of a record with a text
    3. As a path to an image

    We provide an ability to use any query item. So, a user can specify any.

    """

    def setup(self, clickstream_sessions: List[ClickstreamSession]):
        pass

    def __call__(self, query: QueryItem):
        return query
