from collections import defaultdict
from typing import Dict, Iterator, List, Tuple

from datasets import Dataset, DatasetDict

from embedding_studio.embeddings.data.clickstream.paired_session import (
    PairedClickstreamDataset,
)
from embedding_studio.embeddings.data.storages.storage import ItemsStorage
from embedding_studio.embeddings.features.feature_extractor_input import (
    FineTuningInput,
)
from embedding_studio.embeddings.splitters.item_splitter import ItemSplitter


class ItemsStorageSplitter:
    def __init__(self, item_splitter: ItemSplitter):
        """The class which encapsulates the functionality for splitting an item into subitmes.
        When do you need the class? For example if your data is a long text, and can't be fit in the model,
        so you can lose information after data being truncated.

        :param item_splitter: a method for items splitting.
        """
        self.item_splitter = item_splitter

    def _split_items_dataset(
        self, items_dataset: ItemsStorage, groups: dict
    ) -> Iterator[dict]:
        """Generator of subitems

        :param items_dataset: original items storage
        :param groups: after being split into subitmes part_to_object_dict dict stores id to subitem ids
        :return:
            iterator of dict with subitem id and subitem data
        """
        for row in items_dataset:
            for i, subitem in enumerate(
                self.item_splitter(row[items_dataset.item_field_name])
            ):
                subitem_id = f"{row[items_dataset.id_field_name]}:{i}"
                yield {
                    items_dataset.item_field_name: subitem,
                    items_dataset.id_field_name: subitem_id,
                }

                groups[row[items_dataset.id_field_name]].append(subitem_id)

    def __call__(
        self,
        items_dataset: DatasetDict,
        clickstream_dataset: DatasetDict,
    ) -> Tuple[DatasetDict, DatasetDict]:
        """Split items from items dataset, and adjust clickstream accordingly.

        :param items_dataset: dataset dict of original items storages
        :param clickstream_dataset: original clickstream dataset dict
        :return:
            new dataset dict of subitems storages and adjusted clickstream.
        """
        split_items_datasets_dict = dict()
        split_clickstream_datasets_dict = dict()
        for split in ["train", "test"]:
            groups = defaultdict(list)
            split_dataset = Dataset.from_generator(
                lambda: self._split_items_dataset(items_dataset, groups)
            )
            split_items_datasets_dict[split] = split_dataset

            # Change result ids and events + adding part_to_object_dict dict
            split_sessions = []
            for session in (
                clickstream_dataset[split].irrelevant
                + clickstream_dataset[split].not_irrelevant
            ):
                session_events: List[str] = []
                session_results: List[str] = []
                session_ranks: Dict[str, float] = dict()
                session_event_types: List[float] = []
                session_part_to_object_dict: Dict[str, str] = dict()

                for event_id in session.events:
                    session_events += groups[event_id]

                for i, result_id in enumerate(session.results):
                    session_results += groups[result_id]

                    for id_ in groups[result_id]:
                        session_part_to_object_dict[id_] = result_id
                        session_ranks[id_] = session.ranks[result_id]

                        if (
                            session.event_types is None
                            and session_event_types is not None
                        ):
                            session_event_types = None
                        else:
                            session_event_types.append(session.event_types[i])

                split_sessions.append(
                    FineTuningInput(
                        events=session_events,
                        results=session_results,
                        ranks=session_ranks,
                        event_types=session_event_types,
                        part_to_object_dict=session_part_to_object_dict,
                        timestamp=session.timestamp,
                    )
                )

            split_clickstream_datasets_dict[split] = PairedClickstreamDataset(
                sessions=split_sessions,
                randomize=clickstream_dataset[split].randomize,
                session_count=clickstream_dataset[split].session_count,
            )

        return DatasetDict(split_items_datasets_dict), DatasetDict(
            split_clickstream_datasets_dict
        )
