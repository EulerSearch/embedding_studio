from datasets import DatasetDict


class RankingData:
    def __init__(self, clickstream: DatasetDict, items: DatasetDict):
        self.clickstream = clickstream
        self.items = items
