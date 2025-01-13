from typing import Any, Dict, Optional, Type

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

    def _get_id(self, query: QueryItemType) -> str:
        # Ensure query.text is a valid string
        if not isinstance(query.text, str):
            raise ValueError(
                f"Expected 'query.text' to be a string, got {type(query.text)}"
            )

        return query.text

    def _get_storage_metadata(self, query: QueryItemType) -> Dict[str, Any]:
        return query.model_dump()

    def _get_payload(self, query: QueryItemType) -> Optional[Dict[str, Any]]:
        return None
