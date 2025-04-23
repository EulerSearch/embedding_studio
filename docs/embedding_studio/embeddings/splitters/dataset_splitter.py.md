# Documentation for ItemsSetSplitter

## Overview
The ItemsSetSplitter class is designed to split a dataset of items into smaller subitems using an item splitting mechanism. This functionality is essential for managing datasets where items exceed model input size restrictions. By decomposing items into manageable chunks, it enables fine-tuning and training while ensuring each entry adheres to model constraints.

## Functionality
The method _split_items_dataset within the ItemsSetSplitter splits the original items dataset into multiple subitems. Each subitem is assigned a unique identifier derived from the original item's ID, and the relationship between original and subitem IDs is recorded in a provided groups dictionary.

## Parameters
- **items_dataset**: An original items dataset containing items to split. It must provide 'item_field_name' for item data and 'id_field_name' for the original item ID.
- **groups**: A dictionary to store mappings from original item IDs to lists of generated subitem IDs.

## Returns
- **Iterator[dict]**: The method yields dictionaries for each subitem with its corresponding unique ID.

## Inheritance
The ItemsSetSplitter class does not inherit from any other class, functioning as a standalone utility. It utilizes an instance of the ItemSplitter for data transformation.

## Usage
To utilize the ItemsSetSplitter, it is initialized with an instance of an item splitter. The items and clickstream datasets are then passed to the splitter, which returns a tuple containing the split item dataset and the updated clickstream dataset.

### Example
```python
from embedding_studio.embeddings.splitters.dataset_splitter import ItemsSetSplitter
from embedding_studio.embeddings.splitters.item_splitter import ItemSplitter

item_splitter = ItemSplitter()
splitter = ItemsSetSplitter(item_splitter)
new_items_ds, new_clickstream_ds = splitter(items_ds, clickstream_ds)

groups = {}
for subitem in splitter._split_items_dataset(items_dataset, groups):
    print(subitem)
```

### Motivation
The motivation behind the ItemsSetSplitter is to facilitate the management of large item datasets by splitting them, thereby enhancing the flexibility and efficiency of machine learning model training processes.