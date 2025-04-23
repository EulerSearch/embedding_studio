from abc import abstractmethod
from typing import Any, Dict, Optional, Type, TypeVar, Union

from pydantic import BaseModel

QueryItemType = TypeVar("QueryItemType", bound=BaseModel)


class QueryRetriever:
    """
    Abstract base class for retrieving and processing queries.

    QueryRetriever provides a standard interface for handling different types of queries.
    It supports converting queries to different formats, extracting identifiers and metadata,
    and handling optional payload information.
    """

    @abstractmethod
    def get_model_class(self) -> Type[QueryItemType]:
        """
        Return the actual model class corresponding to QueryItemType.

        :return: A Type object representing the concrete QueryItemType class

        Example implementation:
        ```python
        def get_model_class(self) -> Type[CustomQueryItem]:
            return CustomQueryItem
        ```
        """

    def __parse_dict(self, value: dict) -> QueryItemType:
        """
        Parse a dictionary into a QueryItemType instance.

        :param value: Dictionary to parse into a QueryItemType
        :return: Instance of QueryItemType constructed from the dictionary
        """
        # Get the actual model class from the generic type
        model_class = self.get_model_class()
        return model_class.model_validate(
            value
        )  # Correctly use the validate method

    @abstractmethod
    def _convert_query(self, query: QueryItemType) -> Any:
        """
        Convert the query to the appropriate format for retrieval.

        :param query: The query object to convert
        :return: The converted query in an appropriate format

        Example implementation:
        ```python
        def _convert_query(self, query: CustomQueryItem) -> str:
            return query.search_text
        ```
        """

    @abstractmethod
    def _get_id(self, query: QueryItemType) -> str:
        """
        Extract a unique identifier from the query.

        :param query: The query object to extract ID from
        :return: A string identifier for the query

        Example implementation:
        ```python
        def _get_id(self, query: CustomQueryItem) -> str:
            return f"query_{query.search_text[:20]}"
        ```
        """

    @abstractmethod
    def _get_storage_metadata(self, query: QueryItemType) -> Dict[str, Any]:
        """
        Extract metadata from the query for storage purposes.

        :param query: The query object to extract metadata from
        :return: Dictionary of metadata extracted from the query

        Example implementation:
        ```python
        def _get_storage_metadata(self, query: CustomQueryItem) -> Dict[str, Any]:
            return query.model_dump()
        ```
        """

    def _get_payload(self, query: QueryItemType) -> Optional[Dict[str, Any]]:
        """
        Extract optional payload information from the query.

        :param query: The query object to extract payload from
        :return: Dictionary of payload information or None

        Example implementation:
        ```python
        def _get_payload(self, query: CustomQueryItem) -> Optional[Dict[str, Any]]:
            return query.extra_data if hasattr(query, 'extra_data') else None
        ```
        """
        return None

    def __call__(self, query: Union[QueryItemType, Dict]) -> Any:
        """
        Make the retriever callable, converting the provided query.

        :param query: The query object or dictionary to convert
        :return: The converted query
        """
        parsed_query = query
        if isinstance(query, dict):
            parsed_query = self.__parse_dict(parsed_query)

        return self._convert_query(parsed_query)

    def get_id(self, query: Union[QueryItemType, Dict]) -> Any:
        """
        Get the ID for the provided query.

        :param query: The query object or dictionary to get ID from
        :return: The ID of the query
        """
        parsed_query = query
        if isinstance(query, dict):
            parsed_query = self.__parse_dict(parsed_query)

        return self._get_id(parsed_query)

    def get_storage_metadata(
        self, query: Union[QueryItemType, Dict]
    ) -> Dict[str, Any]:
        """
        Get storage metadata for the provided query.

        :param query: The query object or dictionary to get metadata from
        :return: Dictionary of metadata for storage
        """
        parsed_query = query
        if isinstance(query, dict):
            parsed_query = self.__parse_dict(parsed_query)

        return self._get_storage_metadata(parsed_query)

    def get_payload(
        self, query: Union[QueryItemType, Dict]
    ) -> Optional[Dict[str, Any]]:
        """
        Get payload for the provided query.

        :param query: The query object or dictionary to get payload from
        :return: Optional dictionary of payload information
        """
        parsed_query = query
        if isinstance(query, dict):
            parsed_query = self.__parse_dict(parsed_query)

        return self._get_payload(parsed_query)
