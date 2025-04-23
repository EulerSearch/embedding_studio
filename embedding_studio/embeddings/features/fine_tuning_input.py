from typing import Any, Dict, List, Optional, Set, Union

from pydantic import BaseModel, validator
from torch import Tensor


class FineTuningInput(BaseModel):
    """Class that represents clickstream session passes to a feature extractor.

    :param query: The user query or search term that initiated the session
    :param events: List of IDs representing items that received user interactions (e.g., clicks)
    :param results: List of all result item IDs shown to the user
    :param ranks: Dictionary mapping each item ID to its rank value in search results
    :param event_types: Optional list of event type indicators for each event
    :param timestamp: Optional timestamp indicating when the session was initialized
    :param is_irrelevant: Boolean indicating if the session is fully irrelevant (no events)
    :param part_to_object_dict: Optional dictionary mapping part IDs to object IDs, used when items are split into subitems
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
        """Returns a list of result IDs that did not receive any user interaction (non-events).

        :return: List of item IDs that appeared in results but not in events
        """
        return [id_ for id_ in self.results if id_ not in self.events]

    def __len__(self) -> int:
        """Returns the number of results in this input.

        :return: Integer count of results
        """
        return len(self.results)

    def get_object_id(self, id: str) -> str:
        """Maps a part ID to its parent object ID if part_to_object_dict is defined.

        :param id: The part ID to look up
        :return: The corresponding object ID if found in mapping, otherwise returns the original ID
        """
        if self.part_to_object_dict:
            return self.part_to_object_dict.get(id, id)
        else:
            return id

    def remove_results(self, ids: Union[List[str], Set[str]]):
        """Removes specified results from this input, updating all related data structures.

        This method updates results, events, ranks, and part_to_object_dict (if present) by removing
        all entries corresponding to the specified IDs. If part_to_object_dict is present, it will
        also remove any part IDs that map to the specified object IDs.

        :param ids: List or set of IDs to remove from the results
        """
        if self.part_to_object_dict is not None:
            # If we have a mapping from parts to objects, we need to find all part_ids
            # that map to any of the object_ids in the provided 'ids' list
            ids_to_remove = set(
                [
                    part_id
                    for part_id in self.part_to_object_dict.keys()
                    if self.part_to_object_dict[part_id]
                    in ids  # Find parts belonging to objects we want to remove
                ]
            )
            # Update the mapping by keeping only entries where the part_id is not in our removal set
            self.part_to_object_dict = {
                part_id: object_id
                for part_id, object_id in self.part_to_object_dict.items()
                if part_id not in ids_to_remove
            }

        else:
            # If no part-to-object mapping exists, simply use the provided ids directly
            ids_to_remove = ids

        # Filter out the removed ids from results list
        self.results = [
            id_ for id_ in self.results if id_ not in ids_to_remove
        ]
        # Filter out the removed ids from events list
        self.events = [id_ for id_ in self.events if id_ not in ids_to_remove]

        # Update irrelevant flag based on whether there are any remaining events
        # If no events remain, mark this input as irrelevant
        self.is_irrelevant = (
            len(self.events) == 0
        )  # TODO: will be passed later and not be calculated

        # Update the ranks dictionary by removing entries for the removed ids
        self.ranks = {
            id_: rank
            for id_, rank in self.ranks.items()
            if id_ not in ids_to_remove
        }

    @validator("results", "results", pre=True, always=True)
    def preprocess_ids(cls, value):
        """
        Validator method to ensure all IDs are converted to strings regardless of input type.

        :param cls: The class being validated (automatically provided by Pydantic)
        :param value: List of values to be processed into string IDs
        :return: List of string IDs
        """
        return [
            (
                str(
                    item[0]
                )  # If item is a tuple, convert its first element to string
                if isinstance(item, tuple)
                else (
                    str(
                        item.item()
                    )  # If item is a PyTorch Tensor, extract its value and convert to string
                    if isinstance(item, Tensor)
                    else str(
                        item
                    )  # For all other types, simply convert directly to string
                )
            )
            for item in value  # Process each item in the input list
        ]
