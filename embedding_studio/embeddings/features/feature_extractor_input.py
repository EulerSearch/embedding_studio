from typing import Any, Dict, List, Optional

from pydantic import BaseModel, validator
from torch import Tensor


class FineTuningInput(BaseModel):
    """Class that represents clickstream session passes to a feature extractor.

    :param query: provided query.
    :param events: ids of results (right now mostly clicks)
    :param results: ids of result items
    :param ranks: dictionary of each item ranks
    :param event_types: type of results
    :param timestamp: when session was initialized
    :param not_events: results that have no events with
    :param is_irrelevant: clickstream session is fully irrelevant
    :param groups: dictionary of ids to group id, use if an item is split into subitems.
    """

    query: Any
    events: List[str]
    results: List[str]
    ranks: Dict[str, Optional[float]]
    event_types: Optional[List[float]] = None
    timestamp: Optional[int] = None
    is_irrelevant: Optional[bool] = None
    part_to_object_dict: Optional[Dict[str, str]] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)

        if len(self.ranks) != len(self.results):
            raise ValueError("Sizes of ranks and results are not equal")

        for id_ in self.results:
            if id_ not in self.ranks:
                raise ValueError(f"No such ID ({id_}) in provided ranks dict")

        self.is_irrelevant = (
            len(self.events) == 0
        )  # TODO: will be passed later and not be calculated

    @property
    def not_events(self) -> List[str]:
        return [id_ for id_ in self.results if id_ not in self.events]

    def __len__(self) -> int:
        return len(self.results)

    def get_object_id(self, id: str) -> str:
        if self.part_to_object_dict:
            return self.part_to_object_dict.get(id, id)
        else:
            return id

    @validator("results", "results", pre=True, always=True)
    def preprocess_ids(cls, value):
        return [
            (
                str(item[0])
                if isinstance(item, tuple)
                else (
                    str(item.item()) if isinstance(item, Tensor) else str(item)
                )
            )
            for item in value
        ]
