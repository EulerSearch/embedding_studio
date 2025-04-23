# Documentation for Upsertion Methods

## handle_failed_items

### Functionality
Handles failed items during the upsertion process by logging the error, appending each failed DataItem along with a traceback, and updating the task status accordingly.

### Parameters
- **failed_items**: `List[Tuple[DataItem, str]]` -- Tuples of the failed item and its corresponding traceback.
- **task**: `BaseDataHandlingTask` -- The upsertion task object from the database.
- **exception**: `Exception` -- The exception that occurred during the process.
- **task_crud**: `CRUDBase` -- The CRUD handler for persisting task updates.

### Usage
Purpose: Manage failures in upsertion by recording error details and propagating critical errors based on configuration.

#### Example
```python
try:
    # upsert actions
    pass
except Exception as e:
    handle_failed_items(failed_items, task, e, task_crud)
```

---

## upsert_batch

### Functionality
Processes a batch of data items by downloading, splitting, running inference, and uploading vectors. It logs each stage and handles errors by marking tasks as failed when necessary.

### Parameters
- **batch**: `List of DataItems` to process.
- **data_loader**: `DataLoader` instance for downloading data.
- **items_splitter**: `ItemSplitter` instance for splitting items.
- **preprocessor**: `Preprocessor` instance to format data.
- **inference_client**: `TritonClient` instance for inference.
- **collection**: Target collection to upload vectors.
- **batch_index**: Index of the current batch.
- **task**: Task object representing the upsertion process.
- **task_crud**: `CRUDBase` instance to update task status.

### Usage
Process a batch of items using `upsert_batch`. The function downloads data, splits and preprocesses content, performs inference, and uploads generated vectors. It provides detailed logging and handles errors gracefully.

#### Example
```python
batch = [...]  
data_loader = DataLoader()  
items_splitter = ItemSplitter()  
preprocessor = ItemsDatasetDictPreprocessor()  
inference_client = TritonClient()  
collection = Collection()  
batch_index = 0  
task = get_task()  
task_crud = get_task_crud()  

upsert_batch(batch, data_loader, items_splitter, preprocessor,
             inference_client, collection, batch_index, task, task_crud)
```

---

## process_upsert

### Functionality
Processes an upsertion task in batches by downloading, preprocessing, splitting, inferring, and uploading vectors. Handles errors at each stage and updates the task status.

### Parameters
- **task**: Upsertion task containing items and status info.
- **collection**: Target collection for vector uploads.
- **data_loader**: Loader to fetch item details.
- **items_splitter**: Splits downloaded items into parts.
- **preprocessor**: Prepares items for inference processing.
- **inference_client**: Client to run inference on item parts.
- **task_crud**: CRUD handler to update the task in storage.

### Usage
Purpose: Process and upsert items in a task by batching operations.

#### Example
```python
process_upsert(
    task,
    collection,
    data_loader,
    items_splitter,
    preprocessor,
    inference_client,
    task_crud
)
```