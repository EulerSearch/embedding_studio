from typing import Dict, List

from embedding_studio.embeddings.features.clicks_aggregator.clicks_aggregator import (
    ClicksAggregator,
)


class MaxClicksAggregator(ClicksAggregator):
    def __call__(self, group_clicks: Dict[str, List[int]]) -> Dict[str, int]:
        """Aggregate clicks using a maximum approach: considers a group clicked if any subitem was clicked.

        This aggregator treats a parent item as clicked (value of 1) if at least one of its
        subitems has been clicked. Otherwise, the parent item is considered not clicked (value of 0).

        :param group_clicks: dict in the format item_id - list of subitem clicks
        :return: dict in the format item_id - single click value (1 if any subitem clicked, 0 otherwise)
        """
        # Aggregates clicks such that if any item in the group has been clicked, the whole group is considered clicked
        return {group: max(clicks) for group, clicks in group_clicks.items()}
