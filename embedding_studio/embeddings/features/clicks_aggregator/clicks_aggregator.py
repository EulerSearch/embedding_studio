from abc import ABC, abstractmethod
from typing import Dict, List


class ClicksAggregator(ABC):
    """Interface for a method of clicks aggregation in the situation if an item is split into subitems."""

    @abstractmethod
    def __call__(self, group_clicks: Dict[str, List[int]]) -> Dict[str, int]:
        """Aggregate clicks.

        :param group_clicks: dict in the format item_id - list of subitem clicks
        :return:
            dict in the format item_id - single click value
        """
        raise NotImplemented
