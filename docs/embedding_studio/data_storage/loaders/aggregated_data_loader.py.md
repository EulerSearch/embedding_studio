## Documentation for `AggregatedDataLoader`

### Functionality

The `AggregatedDataLoader` class is a DataLoader implementation that aggregates multiple data loaders, allowing data to be loaded from different sources through a single interface. It routes load requests to the appropriate loader based on the source name in the item metadata.

### Parameters

- `loaders`: Dictionary mapping source names to their respective DataLoader instances.
- `item_meta_cls`: The ItemMetaWithSourceInfo class type to use for metadata.

### Usage

- **Purpose** - The main purpose of the `AggregatedDataLoader` is to simplify the process of loading data from multiple sources by providing a unified interface.

#### Example

```python
# Example usage of AggregatedDataLoader

from embedding_studio.data_storage.loaders.aggregated_data_loader import AggregatedDataLoader
from embedding_studio.data_storage.loaders.some_data_loader import SomeDataLoader

# Create individual data loaders
loader1 = SomeDataLoader()
loader2 = SomeDataLoader()

# Create an aggregated data loader
aggregated_loader = AggregatedDataLoader(loaders={'source1': loader1, 'source2': loader2}, item_meta_cls=ItemMetaWithSourceInfo)

# Load data
items_data = [...]  # List of ItemMetaWithSourceInfo objects
combined_dataset = aggregated_loader.load(items_data)
```

## Documentation for `item_meta_cls`

### Functionality

Returns the ItemMeta class used by this loader.

### Parameters

- `None`: This method does not take any parameters.

### Usage

- **Purpose** - To provide access to the ItemMetaWithSourceInfo class type used for metadata in this loader.

#### Example

```python
item_meta_class = aggregated_data_loader.item_meta_cls
```

## Documentation for `load`

### Functionality

The `load` method of the `AggregatedDataLoader` class is responsible for loading data items from multiple sources and combining them into a single dataset. It groups the items by their source and delegates the loading process to the appropriate data loader for each source.

### Parameters

- `items_data`: List of `ItemMetaWithSourceInfo` objects identifying the items to load.

### Usage

- **Purpose** - This method allows for the aggregation of data from various sources, providing a unified dataset for further processing.

#### Example

```python
# Example usage of the load method
aggregated_loader = AggregatedDataLoader(loaders, ItemMetaWithSourceInfo)
dataset = aggregated_loader.load(items_data)
```

## Documentation for `_load_batch_with_offset`

### Functionality

Load a batch of data items from all sources starting from the given offset. This method retrieves batches from all loaders and combines them into a single batch.

### Parameters

- `offset`: The offset from where to start loading items.
- `batch_size`: The number of items to load in a single batch.
- `kwargs`: Additional parameters for customizing the batch loading process.

### Usage

- **Purpose** - To load a specified number of items from various data sources, starting from a defined position in the dataset.

#### Example

```python
# Example usage of the _load_batch_with_offset method
loader = AggregatedDataLoader(loaders, ItemMetaWithSourceInfo)
items = loader._load_batch_with_offset(offset=0, batch_size=10)
```

## Documentation for `total_count`

### Functionality

Calculates the total count of items across all loaders.

### Parameters

- `kwargs`: Additional parameters passed to each loader's total_count method.

### Usage

- **Purpose** - To retrieve the total number of items available from all data loaders combined.

#### Example

```python
# Assuming `aggregated_loader` is an instance of AggregatedDataLoader
count = aggregated_loader.total_count()
print(count)  # Outputs the total count of items across all loaders
```

## Documentation for `load_all`

### Functionality

A generator that iteratively loads all data in batches from all sources.

### Parameters

- `batch_size`: The size of each batch to load.
- `kwargs`: Additional parameters for customizing the batch loading process.

### Usage

- **Purpose** - This method overrides the base implementation to get batches from all loaders for each offset.

#### Example

```python
for batch in aggregated_data_loader.load_all(batch_size=10):
    process_batch(batch)
```