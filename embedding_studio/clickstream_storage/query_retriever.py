from abc import abstractmethod
from typing import Any, Dict, Optional, Type, TypeVar, Union

from pydantic import BaseModel

QueryItemType = TypeVar("QueryItemType", bound=BaseModel)


class QueryRetriever:
    @abstractmethod
    def get_model_class(self) -> Type[QueryItemType]:
        """Return the actual model class corresponding to QueryItemType."""

    def __parse_dict(self, value: dict) -> QueryItemType:
        # Get the actual model class from the generic type
        model_class = self.get_model_class()
        return model_class.model_validate(
            value
        )  # Correctly use the validate method

    @abstractmethod
    def _convert_query(self, query: QueryItemType) -> Any:
        pass

    @abstractmethod
    def _get_id(self, query: QueryItemType) -> str:
        pass

    @abstractmethod
    def _get_storage_metadata(self, query: QueryItemType) -> Dict[str, Any]:
        pass

    def _get_payload(self, query: QueryItemType) -> Optional[Dict[str, Any]]:
        return None

    def __call__(self, query: Union[QueryItemType, Dict]) -> Any:
        parsed_query = query
        if isinstance(query, dict):
            parsed_query = self.__parse_dict(parsed_query)

        return self._convert_query(parsed_query)

    def get_id(self, query: Union[QueryItemType, Dict]) -> Any:
        parsed_query = query
        if isinstance(query, dict):
            parsed_query = self.__parse_dict(parsed_query)

        return self._get_id(parsed_query)

    def get_storage_metadata(
        self, query: Union[QueryItemType, Dict]
    ) -> Dict[str, Any]:
        parsed_query = query
        if isinstance(query, dict):
            parsed_query = self.__parse_dict(parsed_query)

        return self._get_storage_metadata(parsed_query)

    def get_payload(
        self, query: Union[QueryItemType, Dict]
    ) -> Optional[Dict[str, Any]]:
        parsed_query = query
        if isinstance(query, dict):
            parsed_query = self.__parse_dict(parsed_query)

        return self._get_payload(parsed_query)
