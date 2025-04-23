from typing import Any, Dict, Optional, Type

from embedding_studio.clickstream_storage.query_retriever import (
    QueryItemType,
    QueryRetriever,
)
from embedding_studio.clickstream_storage.text_query_item import TextQueryItem


class TextQueryRetriever(QueryRetriever):
    """
    Implementation of QueryRetriever for text-based queries.

    Handles retrieval operations specifically for TextQueryItem objects.
    """

    def get_model_class(self) -> Type[TextQueryItem]:
        """
        Return the TextQueryItem class as the concrete model class.

        :return: TextQueryItem class type
        """
        return TextQueryItem

    def _convert_query(self, query: QueryItemType) -> Any:
        """
        Convert the query to its text representation.

        Extracts the text content from the TextQueryItem for retrieval operations.

        :param query: The TextQueryItem to convert
        :return: The text content of the query
        :raises ValueError: If the query object doesn't have a dict attribute
        """
        if not hasattr(query, "dict"):
            raise ValueError(f"Query object does not have dict attribute")
        return query.text

    def _get_id(self, query: QueryItemType) -> str:
        """
        Use the query text as the unique identifier.

        :param query: The TextQueryItem to extract ID from
        :return: The text content as a string identifier
        :raises ValueError: If query.text is not a string
        """
        # Ensure query.text is a valid string
        if not isinstance(query.text, str):
            raise ValueError(
                f"Expected 'query.text' to be a string, got {type(query.text)}"
            )

        return query.text

    def _get_storage_metadata(self, query: QueryItemType) -> Dict[str, Any]:
        """
        Extract all fields from the query as storage metadata.

        Uses model_dump to convert the query to a dictionary for storage.

        :param query: The TextQueryItem to extract metadata from
        :return: Dictionary representation of the query
        """
        return query.model_dump()

    def _get_payload(self, query: QueryItemType) -> Optional[Dict[str, Any]]:
        """
        No additional payload information for text queries.

        :param query: The TextQueryItem to extract payload from
        :return: None as there is no payload
        """
        return None
