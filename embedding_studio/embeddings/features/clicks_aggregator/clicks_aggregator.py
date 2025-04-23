from abc import ABC, abstractmethod
from typing import Dict, List


class ClicksAggregator(ABC):
    """Interface for a method of clicks aggregation in the situation if an item is split into subitems.

    This abstract base class defines the interface for aggregating click data from subitems
    into aggregated click data for parent items.
    """

    @abstractmethod
    def __call__(self, group_clicks: Dict[str, List[int]]) -> Dict[str, int]:
        """Aggregate clicks from subitems into parent items.

        :param group_clicks: dict in the format item_id - list of subitem clicks
        :return: dict in the format item_id - single aggregated click value

        Example implementation:
        ```
        def __call__(self, group_clicks: Dict[str, List[int]]) -> Dict[str, int]:
            return {group: sum(clicks) for group, clicks in group_clicks.items()}
        ```
        """
        raise NotImplemented
