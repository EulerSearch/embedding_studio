from datasets import DatasetDict


class RankingData:
    def __init__(self, clickstream: DatasetDict, items: DatasetDict):
        """Initialize a RankingData object with clickstream and items datasets.

        :param clickstream: Dataset containing user interaction data (clicks, views, etc.)
        :param items: Dataset containing information about the items being ranked
        """
        self.clickstream = clickstream
        self.items = items
