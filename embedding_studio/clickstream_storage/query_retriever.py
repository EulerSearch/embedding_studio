from typing import List

from embedding_studio.clickstream_storage.query_item import QueryItem
from embedding_studio.clickstream_storage.raw_session import ClickstreamSession


class QueryRetriever(object):
    """As we can't exactly predict a schema of storing queries:
    1. As dict exceptly in clickstream service
    2. As ID of a record with a dict
    3. As a path to an image

    We provide an ability to use any query item. So, a user can specify any.

    """

    def setup(self, clickstream_sessions: List[ClickstreamSession]):
        pass

    def __call__(self, query: QueryItem):
        return query
