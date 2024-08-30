from typing import Any, Type

from embedding_studio.clickstream_storage.query_retriever import (
    QueryItemType,
    QueryRetriever,
)
from embedding_studio.clickstream_storage.text_query_item import TextQueryItem


class TextQueryRetriever(QueryRetriever):
    def get_model_class(self) -> Type[TextQueryItem]:
        return TextQueryItem

    def _convert_query(self, query: QueryItemType) -> Any:
        if not hasattr(query, "dict"):
            raise ValueError(f"Query object does not have dict attribute")
        return query.text
