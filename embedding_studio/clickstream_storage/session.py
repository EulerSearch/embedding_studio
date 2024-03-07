from typing import Any, Dict, List, Optional

from pydantic import BaseModel, validator
from torch import Tensor


class ClickstreamSession(BaseModel):
    """Class that represents clickstream session.

    :param query: provided query.
    :param events: ids of results (right now mostly clicks)
    :param results: ids of result items
    :param ranks: dictionary of each item ranks
    :param event_types: type of results
    :param timestamp: when srddion was initialized
    """

    query: Any
    events: List[str]
    results: List[str]
    ranks: Dict[str, Optional[float]]
    event_types: Optional[List[float]] = None
    timestamp: Optional[int] = None
    not_events: Optional[List[str]] = None
    is_irrelevant: Optional[bool] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)

        if len(self.ranks) != len(self.results):
            raise ValueError("Sizes of ranks and results are not equal")

        for id_ in self.results:
            if id_ not in self.ranks:
                raise ValueError(f"No such ID ({id_}) in provided ranks dict")

        self.not_events = [
            id_ for id_ in self.results if id_ not in self.events
        ]
        self.is_irrelevant = (
            len(self.events) == 0
        )  # TODO: will be passed later and not be calculated

    def __len__(self) -> int:
        return len(self.results)

    @validator("results", "results", pre=True, always=True)
    def preprocess_ids(cls, value):
        return [
            str(item[0])
            if isinstance(item, tuple)
            else str(item.item())
            if isinstance(item, Tensor)
            else str(item)
            for item in value
        ]
