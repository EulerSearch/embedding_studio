## Documentation for DataLoader

### Functionality
DataLoader is an abstract base class that defines the interface for data loaders. It provides abstract methods for loading full datasets and batched items, as well as a property for specifying metadata types. Implementations must override the following:
- `item_meta_cls`
- `load`
- `load_items`
- `_load_batch_with_offset`

### Purpose and Motivation
This class serves as a blueprint for creating data loader objects that manage datasets from various sources. It standardizes the loading interface for efficient data retrieval and processing.

### Inheritance
DataLoader inherits from Python's ABC (Abstract Base Class) module, enforcing implementation of abstract methods in subclasses.

### Method Documentation

#### `item_meta_cls`

##### Functionality
This abstract property returns the class used for handling metadata in a DataLoader. Implementations must override this property to return a valid ItemMeta subclass.

##### Parameters
None.

##### Usage
- Purpose: Specify the metadata class associated with the data loader.

###### Example
```python
@property
def item_meta_cls(self):
    return FileItemMeta
```

---

#### `DataLoader.load`

##### Functionality
This method loads a full dataset based on the provided item metadata. It is expected to aggregate the data by calling lower-level methods (e.g., `load_items`) and transform them into a Dataset object.

##### Parameters
- `items_data`: List of ItemMeta objects that indicate which items to load. Each ItemMeta holds the necessary metadata for data retrieval.

##### Returns
A Dataset object containing the loaded data. The Dataset is typically created by aggregating the data and metadata into a dictionary and converting it via `Dataset.from_dict`.

##### Usage
- Purpose: To fully load and assemble a dataset from individual data items, using their metadata.

###### Example
```python
def load(self, items_data: List[ItemMeta]) -> Dataset:
    downloaded_items = self.load_items(items_data)
    data_dict = {
        "id": [item.id for item in downloaded_items],
        "text": [item.data for item in downloaded_items],
        "metadata": [item.meta.dict() for item in downloaded_items]
    }
    return Dataset.from_dict(data_dict)
```

---

#### `DataLoader.load_items`

##### Functionality
This abstract method loads individual items from the provided metadata list. Implementations should return a list of DownloadedItem objects containing the loaded data and their corresponding metadata.

##### Parameters
- `items_data`: List of ItemMeta objects identifying the items to load.

##### Returns
- A list of DownloadedItem objects with loaded data and metadata.

##### Usage
- Purpose: Load each item based on the provided metadata.

###### Example
```python
# Example usage:
downloaded_items = loader.load_items(items_data)
```

---

#### `_load_batch_with_offset`

##### Functionality
Loads a batch of data items starting from a given offset. It retrieves a slice of metadata using provided keyword arguments and then loads the corresponding items.

##### Parameters
- `offset`: The starting index for loading items.
- `batch_size`: Number of items to load in one batch.
- `**kwargs`: Additional parameters for metadata slicing.

##### Return
- A list of DownloadedItem instances.

##### Usage
- Purpose: Process large datasets in manageable chunks.

###### Example
```python
# Example implementation of the method
# within a DataLoader subclass

def _load_batch_with_offset(self, offset, batch_size, **kwargs):
    # Retrieve metadata for the batch
    item_metas = self._get_metadata_slice(offset, batch_size, **kwargs)
    # Load items based on metadata
    return self.load_items(item_metas)
```

---

#### `total_count`

##### Functionality
Retrieves the total number of items available in the dataset. Returns an integer count if accessible, otherwise returns None.

##### Parameters
- `**kwargs`: Optional keyword arguments to customize the count.

##### Usage
- Purpose: To fetch the total item count from the loader.

###### Example
```python
count = data_loader.total_count()
if count is not None:
    print(f"Total count: {count}")
else:
    print("Count is not available.")
```

---

#### `load_all`

##### Functionality
A generator method to iteratively load dataset batches using `_load_batch_with_offset`. It yields each batch for further processing, allowing large datasets to be managed in smaller chunks.

##### Parameters
- `batch_size`: The number of items in each batch.
- `**kwargs`: Additional keyword arguments for batch loading.

##### Usage
- Purpose: Manage large datasets by processing them in batches.

###### Example
```python
for batch in data_loader.load_all(batch_size=100):
    process(batch)
```