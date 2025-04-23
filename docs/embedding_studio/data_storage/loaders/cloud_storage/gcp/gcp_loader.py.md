## Documentation for `read_from_gcp`

### Functionality

Downloads a file from a GCP Cloud Storage bucket into an in-memory file-like object (io.BytesIO).

### Parameters

- `bucket`: The name of the GCP bucket. Must be a non-empty string.
- `file`: The file name to be downloaded. Must be a non-empty string.
- `client`: An initialized GCP storage client instance.

### Usage

Use this function to retrieve files stored in Google Cloud Storage. The returned BytesIO object contains the file's contents, which can then be read or processed as needed.

#### Example

```python
from google.cloud import storage

# Initialize the client
client = storage.Client()

# Download file from GCP
file_stream = read_from_gcp("my_bucket", "my_file.txt", client)

# Process file contents
content = file_stream.read()
```

---

## Documentation for GCPDataLoader

### Functionality

Loads items from Google Cloud Storage with a retry mechanism. Supports credential configuration and custom data schema.

### Parameters

- `retry_config`: Optional retry strategy configuration.
- `features`: Optional schema for loaded data.
- `kwargs`: Additional credential parameters.

### Usage

- **Purpose**: Retrieve data from GCP with robust error handling.
- **Motivation**: Simplify access to cloud-stored data.
- **Inheritance**: Extends DataLoader interface.

#### Example

```python
loader = GCPDataLoader(
    retry_config=custom_retry, features=custom_features,
    project_id="my-project", credentials_path="/path/to/creds"
)
```

---

## Documentation for GCPDataLoader.item_meta_cls

### Functionality

Returns the class used for item metadata. This metadata class is used to represent file metadata in GCP Cloud Storage.

### Return Value

- Returns: GCPFileMeta class that defines file metadata.

### Usage

**Purpose:** Retrieve file metadata class for processing items loaded from GCP Cloud Storage.

#### Example

```python
from embedding_studio.data_storage.loaders.cloud_storage.gcp.gcp_loader import GCPDataLoader

loader = GCPDataLoader(...)
meta_class = loader.item_meta_cls
print(meta_class)
```

---

## Documentation for GCPDataLoader._get_default_retry_config

### Functionality

This method defines the default retry configuration for GCP operations. It creates a RetryConfig with default, credentials, and download_data parameters derived from project settings. It ensures consistent retry behavior for operations like reading credentials and downloading data.

### Parameters

No parameters.

### Returns

- RetryConfig: A configuration object containing:
  - default: Base retry parameters.
  - credentials: Retry parameters for GCP credentials read.
  - download_data: Retry parameters for data download.

### Usage

Use this method to obtain a default retry configuration when no custom configuration is provided to the GCPDataLoader.

#### Example

```python
config = GCPDataLoader._get_default_retry_config()
# Utilize 'config' to manage retry logic in GCP operations.
```

---

## Documentation for GCPDataLoader._read_from_gcp

### Functionality

This method wraps the read_from_gcp function with a retry mechanism to safely download a file from GCP Cloud Storage. It calls the standard read_from_gcp function using a GCP storage client.

### Parameters

- **client**: An initialized GCP storage client used to access the bucket.
- **bucket**: The name of the GCP bucket from which the file is downloaded.
- **file**: The name of the file to download.

### Usage

This internal method is used by the GCPDataLoader class to perform retried downloads of files from GCP Cloud Storage.

#### Example

Assuming you have a valid GCP storage client, you can invoke the method as follows:

```python
from google.cloud import storage

client = storage.Client.from_service_account_json(
    'path/to/credentials.json'
)

data = loader._read_from_gcp(client, 'my_bucket', 'data.txt')
```

---

## Documentation for GCPDataLoader._get_client

### Functionality

This method returns an initialized GCP storage client. It checks the configuration from the GCPDataLoader credentials. If system-provided credentials are enabled, it creates an anonymous client. Otherwise, it loads the service account JSON from the specified path.

### Parameters

This method does not take any parameters.

### Returns

- GCP storage.Client: The client instance for accessing GCP Cloud Storage.

### Usage

This method is used internally by GCPDataLoader to establish a connection with GCP for performing file operations. Users generally initialize GCPDataLoader with appropriate credentials and rely on this method to obtain the client.

#### Example

```python
loader = GCPDataLoader(credentials_path="/path/to/credentials.json")
client = loader._get_client()
```

---

## Documentation for GCPDataLoader._get_item

### Functionality

Retrieves and returns the content of an io.BytesIO file object. This method is designed to handle the file content extraction process, allowing for conversion or parsing as needed.

### Parameters

- `file`: An io.BytesIO object containing the file content. This object is expected to represent the downloaded file from GCP Cloud Storage.

### Return Value

Returns the file object directly. This method can be extended in the future to include parsing or conversion of the file content.

### Usage

- **Purpose**: Extracts raw data from a file object obtained from GCP. It serves as a simple hook for additional data processing if needed.

#### Example

Suppose you have downloaded a file from GCP and stored it in a variable `file_content`:

```python
loader = GCPDataLoader()
data = loader._get_item(file_content)
# `data` now holds the raw file content
```

---

## Documentation for GCPDataLoader._get_data_from_gcp

### Functionality

Retrieves data from GCP Cloud Storage. The method downloads files using metadata provided as a list of GCPFileMeta objects. It caches downloaded items to avoid redundant downloads and handles errors based on the ignore_failures flag.

### Parameters

- files: List of GCPFileMeta objects, each containing bucket and file information.
- ignore_failures: Boolean flag. If True, continues processing subsequent files upon errors; otherwise, raises an exception.

### Return Value

Returns an iterable of tuples, each containing a data dictionary and its corresponding GCPFileMeta object.

### Usage

- Purpose: Download and process file data from a GCP bucket.

#### Example

```python
loader = GCPDataLoader(...)
for data, meta in loader._get_data_from_gcp(files):
    process_data(data)
```

---

## Documentation for GCPDataLoader._process_file_meta

### Functionality

Process a single file metadata to download the file and prepare data objects. The method downloads a file from GCP Cloud Storage, caches it if not already downloaded, and yields tuples of the data dictionary along with the associated file metadata.

### Parameters

- `gcp_client`: The GCP storage client configured for downloading files.
- `file_meta`: Metadata for the specific file to be processed.
- `ignore_failures`: Boolean flag; if True, the method continues after failures; otherwise it raises exceptions.
- `uploaded`: A cache dictionary keyed by a tuple (bucket, file) to store previously downloaded data.

### Yield

Generates tuples containing:
- A data dictionary representing the file's content.
- The associated file metadata (GCPFileMeta).

### Usage

- **Purpose**: Processes file metadata and facilitates downloading of data from GCP. This is used internally by the loader to handle individual files.

#### Example

Assuming you have a valid GCP client and file metadata object:

```python
for data, meta in gcp_loader._process_file_meta(gcp_client, file_meta, False, {}):
    print(data)
```

---

## Documentation for GCPDataLoader._download_and_get_item

### Functionality

This method downloads a file using a provided GCP Cloud Storage client. If the file is not found in the cache dictionary "uploaded", it downloads it and then caches the downloaded data.

### Parameters

- gcp_client: An initialized GCP Storage client for downloading files.
- file_meta: Metadata for the file, including bucket and filename.
- uploaded: A dictionary used as a cache mapping (bucket, file) to the downloaded data.

### Usage

- **Purpose**: Retrieve file data from GCP while avoiding redundant downloads through caching.

#### Example

Imagine `file_meta` with bucket "data-bucket" and file "example.txt". The method downloads the file via `gcp_client` if not cached, and caches it for future reference.

---

## Documentation for GCPDataLoader._yield_item_objects

### Functionality

This method yields data objects and their metadata from a downloaded item. If the item is a list and the file metadata includes an index, the method iterates over each subitem. Otherwise, it wraps the item with its metadata and returns it.

### Parameters

- `item`: The downloaded or retrieved data. It can be a list or a single data object.
- `file_meta`: An instance of GCPFileMeta holding metadata like bucket, file, and index.

### Usage

- **Purpose**: To generate data entries by pairing data objects with their associated GCP metadata for further processing.

#### Example

Suppose you have a downloaded item and corresponding metadata, you can iterate over the yielded objects as follows:

```python
for data_object, meta in loader_instance._yield_item_objects(item, file_meta):
    process(data_object, meta)
```

---

## Documentation for GCPDataLoader._create_item_object

### Functionality

Creates a dictionary item object from provided data and includes file metadata. It initializes the object with the `item_id` set from `file_meta.id`. When `features` is defined and `item` is a dictionary, the object is updated with the contents of `item`. Otherwise, the `item` is assigned to the `item` key.

### Parameters

- `item`: The data content of the item. Can be a dict or other type.
- `file_meta`: An instance of GCPFileMeta containing metadata. Must include an `id` attribute.

### Return

Returns a tuple (`item_object`, `file_meta`) where `item_object` is a dictionary with an `item_id` key and additional item details.

### Usage

This method is invoked internally to yield formatted item objects, typically used in the `_yield_item_objects` generator.

#### Example

```python
data, meta = loader._create_item_object({"a": 1, "b": 2}, file_meta)
print(data["item_id"])  # prints file_meta.id
```

---

## Documentation for GCPDataLoader.load

### Functionality

Loads data as a Hugging Face Dataset by retrieving files from GCP Cloud Storage and processing them with a generator function.

### Parameters

- items_data: List[GCPFileMeta] - A list of metadata objects that specify the files to be loaded from GCP Cloud Storage.

### Usage

- **Purpose** - To load and convert data stored in GCP Cloud Storage into a Hugging Face Dataset format.

#### Example

```python
from embedding_studio.data_storage.loaders.cloud_storage.gcp.gcp_loader import GCPDataLoader

# Initialize loader with desired features
loader = GCPDataLoader(features=features)

# Load dataset using list of GCPFileMeta objects
dataset = loader.load(items_metadata)
```

---

## Documentation for GCPDataLoader._generate_dataset_from_gcp

### Functionality

Generates dataset entries from GCP data by iterating over the results of an internal data fetching method. It retrieves item data and metadata from GCP and yields the item dictionary for each file.

### Parameters

- `files`: List of GCPFileMeta objects that contain the metadata necessary to locate and load files from GCP Cloud Storage.

### Returns

- An iterable of tuples, each containing:
  - A dictionary with the loaded item data, and
  - The corresponding GCPFileMeta metadata.

### Usage

- **Purpose**: Lazily generate dataset entries to be consumed by a Hugging Face Dataset.

#### Example

Suppose you have an instance of GCPDataLoader named `loader` and a list of GCPFileMeta objects called `file_list`. You can create a dataset as follows:

```python
dataset = Dataset.from_generator(
    lambda: loader._generate_dataset_from_gcp(file_list)
)
```

This approach dynamically loads data from GCP as needed.

---

## Documentation for GCPDataLoader.load_items

### Functionality

Loads specific items from GCP Cloud Storage and returns them as a list of DownloadedItem objects. Each DownloadedItem contains an item id, the loaded data, and associated metadata from GCPFileMeta.

### Parameters

- items_data: List[GCPFileMeta] -- List of metadata objects that specify the bucket and file names for the items to load.

### Usage

- **Purpose**: Load designated files from GCP and wrap them into DownloadedItem objects for further processing.

#### Example

Assuming you have a list of GCPFileMeta objects named `items_data` and a GCPDataLoader instance called `loader`, you can load items as follows:

```python
items = loader.load_items(items_data)
```

---

## Documentation for GCPDataLoader._download_blob

### Functionality

Downloads a blob from GCP Cloud Storage and returns its content as an io.BytesIO object. It wraps the GCP Cloud Storage API call and ensures the content pointer is reset to the beginning.

### Parameters

- blob: A `Blob` object that represents a file in a GCP bucket.

### Returns

- An io.BytesIO object containing the contents of the blob.

### Usage

- **Purpose:** Internally invoked to fetch file content from GCP Cloud Storage.

#### Example

```python
content = gcp_data_loader._download_blob(blob)
# Now content.seek(0) has been already called.
```

---

## Documentation for GCPDataLoader._list_blobs

### Functionality

Lists all blobs in a specific GCP Cloud Storage bucket. This method uses a GCP client to connect to the given bucket and returns all blobs available in that bucket.

### Parameters

- `bucket`: The name of the GCP Cloud Storage bucket from which to list blobs.

### Return Value

- A list of Blob objects found in the specified bucket.

### Usage

This method is used internally to access all blobs for further processing. It is not typically called directly.

#### Example

Suppose you want to retrieve blobs from a bucket named "my-bucket":

```python
blobs = instance._list_blobs("my-bucket")
```

---

## Documentation for GCPDataLoader._load_batch_with_offset

### Functionality

This method loads a batch of files from GCP starting at a given offset. It retrieves up to the specified batch size files. If any file fails to load, it logs the error and continues processing.

### Parameters

- `offset`: Starting index for loading files.
- `batch_size`: Number of files to load.
- `kwargs`: Additional arguments; must include the `bucket` name.

### Usage

Use this method internally to process large datasets in batches.

#### Example

```python
batch = loader._load_batch_with_offset(0, 10, bucket="my_bucket")
# batch is a list of DownloadedItem objects.
```

---

## Documentation for GCPDataLoader.load_all

### Functionality

Generates batches of DownloadedItem objects from GCP Cloud Storage. Data is loaded using the _load_batch_with_offset method, yielding each batch until no more data is available. This function enables handling large datasets in manageable chunks.

### Parameters

- `batch_size`: An integer indicating the number of items per batch.
- `kwargs`: A dictionary for additional parameters; must include:
  - `buckets`: A list of GCP bucket names from which data will be loaded.

### Usage

- **Purpose** - To iteratively load and process files from GCP Cloud Storage.

#### Example

For example, using a batch size of 50 with two buckets:

```python
loader = GCPDataLoader()
for batch in loader.load_all(batch_size=50, buckets=['bucket1', 'bucket2']):
    process(batch)
```

---

## Documentation for GCPDataLoader.total_count

### Functionality

Calculates the total number of files across specified GCP buckets. It retrieves blobs from each bucket provided in the `buckets` parameter and sums them to return the total count.

### Parameters

- `buckets`: List of GCP bucket names to count files from.
- Additional keyword arguments may be provided, but only `buckets` is used.

### Usage

- **Purpose** - To determine the overall workload or verify the presence of data in the specified GCP buckets.

#### Example

```python
loader = GCPDataLoader(...)
count = loader.total_count(buckets=['bucket1', 'bucket2'])
print('Total items:', count)
```