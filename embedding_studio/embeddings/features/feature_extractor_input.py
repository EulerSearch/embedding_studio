from typing import Any, Dict, List, Optional, Set, Union

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

    def remove_results(self, ids: Union[List[str], Set[str]]):
        if self.part_to_object_dict is not  None:
            ids_to_remove = set([part_id for part_id in self.part_to_object_dict.keys() if self.part_to_object_dict[part_id] in ids])
            self.part_to_object_dict = {part_id: object_id for part_id, object_id in self.part_to_object_dict.items() if part_id not in ids_to_remove}

        else:
            ids_to_remove = ids

        self.results = [id_ for id_ in self.results if id_ not in ids_to_remove]
        self.events = [id_ for id_ in self.events if id_ not in ids_to_remove]

        self.is_irrelevant = (
                len(self.events) == 0
        )  # TODO: will be passed later and not be calculated

        self.ranks = {id_: rank for id_, rank in self.ranks.items() if id_ not in ids_to_remove}

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
