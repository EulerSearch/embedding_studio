from abc import abstractmethod
from typing import Any, Dict, Type, TypeVar, Union

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

    def __call__(self, query: Union[QueryItemType, Dict]) -> Any:
        parsed_query = query
        if isinstance(query, dict):
            parsed_query = self.__parse_dict(parsed_query)

        return self._convert_query(parsed_query)
