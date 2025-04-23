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
        """Initialize a splitter for sets of items.

        This class encapsulates functionality for splitting items into subitems and adjusting
        associated datasets accordingly. It's particularly useful when working with data that
        exceeds model input size limits and needs to be broken down.

        :param item_splitter: A method for splitting individual items
        """
        self.item_splitter = item_splitter

    def _split_items_dataset(
        self, items_dataset: ItemsSet, groups: dict
    ) -> Iterator[dict]:
        """Generate subitems from the original item dataset.

        This method processes each row in the dataset, applies the splitter to create subitems,
        assigns unique IDs to each subitem, and tracks the relationship between original items
        and their subitems in the groups dictionary.

        :param items_dataset: Original items dataset containing the items to be split
        :param groups: Dictionary to store mapping from original item IDs to subitem IDs, populated during processing
        :return: Iterator of dictionaries containing subitem IDs and data
        """
        for row in items_dataset:
            # For each item in the dataset, apply the splitter to break it into subitems
            for i, subitem in enumerate(
                self.item_splitter(row[items_dataset.item_field_name])
            ):
                # Create a unique ID for each subitem by appending an index to the original ID
                subitem_id = f"{row[items_dataset.id_field_name]}:{i}"

                # Yield a dictionary with the subitem and its ID
                yield {
                    items_dataset.item_field_name: subitem,
                    items_dataset.id_field_name: subitem_id,
                }

                # Record the relationship between original item and this subitem
                groups[row[items_dataset.id_field_name]].append(subitem_id)

    def __call__(
        self,
        items_dataset: DatasetDict,
        clickstream_dataset: DatasetDict,
    ) -> Tuple[DatasetDict, DatasetDict]:
        """Split items from the items dataset and adjust the clickstream dataset accordingly.

        This method performs two main operations:
        1. Splits each item in the items dataset into subitems based on the provided splitter
        2. Updates all references in the clickstream dataset to maintain consistency with the new subitem structure

        The method handles both training and testing splits separately but using the same process.

        :param items_dataset: Dataset dictionary containing original items with 'train' and 'test' splits
        :param clickstream_dataset: Original clickstream dataset dictionary with references to items
        :return: Tuple of (split items dataset, adjusted clickstream dataset) with original structure preserved
        """
        # Initialize dictionaries to store the split datasets
        split_items_datasets_dict = dict()
        split_clickstream_datasets_dict = dict()

        # Process each split ('train' and 'test') separately
        for split in ["train", "test"]:
            # Initialize a dictionary to track relationships between original items and subitems
            groups = defaultdict(list)

            # Create a new dataset by splitting all items in the current split
            split_dataset = Dataset.from_generator(
                lambda: self._split_items_dataset(items_dataset, groups)
            )
            split_items_datasets_dict[split] = split_dataset

            # Transform the clickstream data to reference subitems instead of original items
            split_fine_tuning_inputs = []

            # Process both irrelevant and not_irrelevant fine-tuning inputs
            for fine_tuning_input in (
                clickstream_dataset[split].irrelevant
                + clickstream_dataset[split].not_irrelevant
            ):
                # Initialize containers for the transformed data
                input_events: List[str] = []
                input_results: List[str] = []
                input_ranks: Dict[str, float] = dict()
                input_event_types: List[float] = []
                input_part_to_object_dict: Dict[str, str] = dict()

                # Replace each event ID with its corresponding subitem IDs
                for event_id in fine_tuning_input.events:
                    input_events += groups[event_id]

                # Replace each result ID with its corresponding subitem IDs and update related data
                for i, result_id in enumerate(fine_tuning_input.results):
                    # Add all subitems of this result to the results list
                    input_results += groups[result_id]

                    # For each subitem, maintain mapping back to original item and copy rank information
                    for id_ in groups[result_id]:
                        # Record which original item this subitem came from
                        input_part_to_object_dict[id_] = result_id

                        # Copy the rank from the original item to each of its subitems
                        input_ranks[id_] = fine_tuning_input.ranks[result_id]

                        # Handle event types, preserving any None values correctly
                        if (
                            fine_tuning_input.event_types is None
                            and input_event_types is not None
                        ):
                            # If original event_types became None, our output should also be None
                            input_event_types = None
                        else:
                            # Otherwise, copy the event type for this result
                            input_event_types.append(
                                fine_tuning_input.event_types[i]
                            )

                # Create a new fine-tuning input with the transformed data
                split_fine_tuning_inputs.append(
                    FineTuningInput(
                        events=input_events,
                        results=input_results,
                        ranks=input_ranks,
                        event_types=input_event_types,
                        part_to_object_dict=input_part_to_object_dict,  # Maps subitems back to original items
                        timestamp=fine_tuning_input.timestamp,  # Preserve original timestamp
                    )
                )

            # Create a new paired dataset with the transformed fine-tuning inputs
            split_clickstream_datasets_dict[
                split
            ] = PairedFineTuningInputsDataset(
                inputs=split_fine_tuning_inputs,
                # Preserve original randomization and count settings
                randomize=clickstream_dataset[split].randomize,
                inputs_count=clickstream_dataset[split].inputs_count,
            )

        # Return both transformed datasets with their original structure preserved
        return DatasetDict(split_items_datasets_dict), DatasetDict(
            split_clickstream_datasets_dict
        )
