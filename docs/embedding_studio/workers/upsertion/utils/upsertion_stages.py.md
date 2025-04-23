# Documentation

## `download_items`

### Functionality
Download a list of items using the provided DataLoader. Each DataItem is transformed into a download item via the data_loader's item_meta_cls and then downloaded using load_items. If any error occurs, a DownloadException is raised.

### Parameters
- `items`: List of DataItem instances containing the necessary information.
- `data_loader`: DataLoader instance with an item_meta_cls attribute and a load_items method to process items.

### Usage
- **Purpose**: Retrieve and download items for further processing.

#### Example
```python
downloaded_items = download_items(items, data_loader)
```

---

## `split_items`

### Functionality
Split each item into parts after applying a preprocessor. It applies the provided ItemSplitter to the preprocessed data of each item. It also gathers a mapping from object IDs to parts and collects errors from failed splits.

### Parameters
- `items`: List of DownloadedItem instances to split.
- `item_splitter`: Instance of ItemSplitter used to split items.
- `preprocessor`: Instance of ItemsDatasetDictPreprocessor used to preprocess item data.

### Return Value
Returns a tuple with three elements:
1. A list of parts produced by splitting the items.
2. A dictionary mapping object IDs to lists of part indexes.
3. A list of tuples, each containing a failed DownloadedItem and its error traceback.

### Usage
- **Purpose**: Automate the splitting of downloaded items while handling errors gracefully.

#### Example
```python
parts, obj_to_parts, failed = split_items(downloaded_items, splitter, preprocessor)
```

---

## `run_inference`

### Functionality
Run inference on a list of data items in batches using the specified TritonClient. The function returns a numpy array of inference results.

### Parameters
- `items_data`: List of data on which inference is run.
- `inference_client`: Client that handles inference.

### Usage
Split the input data into batches based on the batch size from settings. Use the inference_client to obtain results for each batch and combine them using numpy.vstack.

#### Example
```python
vectors = run_inference(data_items, triton_client)
```

---

## `upload_vectors`

### Functionality
Uploads computed vectors for a list of items to a collection. The function prepares object parts by extracting vector segments and computing an average vector. After assembling the objects, it upserts them to the collection. Depending on configuration, it may also delete improved objects from the collection.

### Parameters
- `items`: List of DownloadedItem instances, each with metadata used to build object details.
- `vectors`: Numpy array containing the vectors to be uploaded. Each row corresponds to a vector part.
- `object_to_parts`: Dictionary mapping object IDs to a list of indices that indicate vector parts associated with each item.
- `collection`: Collection instance where objects are upserted. It supports operations like upsert, find, and delete.

### Usage
- **Purpose**: To upload and manage vector data by creating object parts, computing an average vector, and updating the collection accordingly.

#### Example
```python
items = [item1, item2, ...]
vectors = np.array([...])
object_to_parts = {"item_id": [0, 1, 2]}
collection = get_collection()

upload_vectors(items, vectors, object_to_parts, collection)
```