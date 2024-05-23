from typing import Dict, List

from embedding_studio.embeddings.features.clicks_aggregator.clicks_aggregator import (
    ClicksAggregator,
)


class MaxClicksAggregator(ClicksAggregator):
    def __call__(self, group_clicks: Dict[str, List[int]]) -> Dict[str, int]:
        """Aggregate clicks in the way: it's a click if there is at least one click among the group.

        :param group_clicks: dict in the format item_id - list of subitem clicks
        :return:
            dict in the format item_id - single click value
        """
        # Aggregates clicks such that if any item in the group has been clicked, the whole group is considered clicked
        return {group: max(clicks) for group, clicks in group_clicks.items()}
