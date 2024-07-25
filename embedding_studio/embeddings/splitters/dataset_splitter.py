from collections import defaultdict
from typing import Dict, Iterator, List, Tuple

from datasets import Dataset, DatasetDict

from embedding_studio.embeddings.data.clickstream.paired_fine_tuning_inputs import (
    PairedFineTuningInputsDataset,
)
from embedding_studio.embeddings.data.items.items_set import ItemsSet
from embedding_studio.embeddings.features.fine_tuning_input import (
    FineTuningInput,
)
from embedding_studio.embeddings.splitters.item_splitter import ItemSplitter


class ItemsSetSplitter:
    def __init__(self, item_splitter: ItemSplitter):
        """The class which encapsulates the functionality for splitting an item into subitmes.
        When do you need the class? For example if your data is a long text, and can't be fit in the model,
        so you can lose information after data being truncated.

        :param item_splitter: a method for items splitting.
        """
        self.item_splitter = item_splitter

    def _split_items_dataset(
        self, items_dataset: ItemsSet, groups: dict
    ) -> Iterator[dict]:
        """Generator of subitems

        :param items_dataset: original items items_set
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

        :param items_dataset: dataset dict of original items items
        :param clickstream_dataset: original clickstream dataset dict
        :return:
            new dataset dict of subitems items and adjusted clickstream.
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
            split_fine_tuning_inputs = []
            for fine_tuning_input in (
                clickstream_dataset[split].irrelevant
                + clickstream_dataset[split].not_irrelevant
            ):
                input_events: List[str] = []
                input_results: List[str] = []
                input_ranks: Dict[str, float] = dict()
                input_event_types: List[float] = []
                input_part_to_object_dict: Dict[str, str] = dict()

                for event_id in fine_tuning_input.events:
                    input_events += groups[event_id]

                for i, result_id in enumerate(fine_tuning_input.results):
                    input_results += groups[result_id]

                    for id_ in groups[result_id]:
                        input_part_to_object_dict[id_] = result_id
                        input_ranks[id_] = fine_tuning_input.ranks[result_id]

                        if (
                            fine_tuning_input.event_types is None
                            and input_event_types is not None
                        ):
                            input_event_types = None
                        else:
                            input_event_types.append(
                                fine_tuning_input.event_types[i]
                            )

                split_fine_tuning_inputs.append(
                    FineTuningInput(
                        events=input_events,
                        results=input_results,
                        ranks=input_ranks,
                        event_types=input_event_types,
                        part_to_object_dict=input_part_to_object_dict,
                        timestamp=fine_tuning_input.timestamp,
                    )
                )

            split_clickstream_datasets_dict[
                split
            ] = PairedFineTuningInputsDataset(
                inputs=split_fine_tuning_inputs,
                randomize=clickstream_dataset[split].randomize,
                inputs_count=clickstream_dataset[split].inputs_count,
            )

        return DatasetDict(split_items_datasets_dict), DatasetDict(
            split_clickstream_datasets_dict
        )
