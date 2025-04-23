## Documentation for `read_from_s3`

### Functionality

Reads a file from S3 and returns its content wrapped in a BytesIO object. In case the file is not found (HTTP 404), it logs an error and returns None. Other S3 errors raise the original exception.

### Parameters

- `client`: Boto3 S3 client used to download the file.
- `bucket`: Name of the S3 bucket.
- `file`: The key or path of the file in the bucket.

### Usage

- **Purpose**: Download a file from S3 as a BytesIO stream.

#### Example

```python
from boto3 import client

s3_client = client("s3")
file_stream = read_from_s3(s3_client, "mybucket", "file.txt")
if file_stream:
    content = file_stream.read()
    print(content)
```

---

## Documentation for `AwsS3DataLoader`

### Functionality

The AwsS3DataLoader class is a DataLoader implementation that loads data items from AWS S3 buckets. It supports retry strategies for handling failures during downloads and allows custom authentication methods such as role-based access, direct keys, or anonymous access.

### Inheritance

Inherits from DataLoader.

### Motivation

This class is designed to efficiently load data from AWS S3 by providing mechanisms for error handling and flexible authentication. Its retry capabilities ensure robustness in unstable network conditions.

### Parameters

- `retry_config`: Configuration for retry logic during operations.
- `features`: Expected dataset features schema.
- `kwargs`: Additional parameters for AWS S3 credentials configuration.

### Usage

- **Purpose** - Load and process data from AWS S3 buckets with reliable error handling and customizable authentication.

#### Example

```python
loader = AwsS3DataLoader(retry_config=my_retry, features=my_features)
data = loader.load_data('bucket_name', 'file_key')
```

---

## Documentation for `AwsS3DataLoader.item_meta_cls`

### Functionality

Returns the class used to represent metadata for files stored in AWS S3. This property returns the `S3FileMeta` class, which encapsulates metadata associated with S3 objects.

### Parameters

This is a property and does not accept any parameters.

### Usage

- **Purpose**: To obtain the metadata class type configured for the S3 data loader.

#### Example

```python
from embedding_studio.data_storage.loaders.cloud_storage.s3.s3_loader import AwsS3DataLoader

# create an instance (provide necessary parameters)
loader = AwsS3DataLoader()
meta_cls = loader.item_meta_cls
print(meta_cls)
```

---

## Documentation for `AwsS3DataLoader._get_default_retry_config`

### Functionality

Creates a default retry configuration for S3 operations. It sets up retry parameters for credential acquisition and data downloads using predefined timeout and attempt values from the settings.

### Parameters

This method does not accept any parameters.

### Usage

- **Purpose:** Provides a default retry configuration when no custom configuration is specified.

#### Example

```python
config = AwsS3DataLoader._get_default_retry_config()
print(config)
```

---

## Documentation for `AwsS3DataLoader._read_from_s3`

### Functionality

Reads a file from an AWS S3 bucket using retry capabilities. This method calls a helper function to download the file and returns a BytesIO object with its contents.

### Parameters

- `client`: A boto3 S3 client with valid AWS credentials.
- `bucket`: S3 bucket name as a non-empty string.
- `file`: File key or path in the bucket as a non-empty string.

### Usage

- **Purpose**: Reliably download a file from S3 with automatic retries. 
- Decorated with @retry_method to handle intermittent failures.

#### Example

```python
client = s3_loader._get_client("task_id")
file_obj = s3_loader._read_from_s3(client, "my-bucket", "folder/data.txt")
```

---

## Documentation for `AwsS3DataLoader._get_client`

### Functionality

Obtains an S3 client with proper credentials. If access keys are not provided and `use_system_info` is False, an anonymous client is created. Alternatively, role-based credentials are assumed. The method is retried upon failure for reliability.

### Parameters

- `task_id`: Unique identifier for the session, used as the role session name.

### Usage

- **Purpose**: Create a boto3 S3 client configured with either direct or role-based credentials.

#### Example

```python
# Example usage of _get_client
loader = AwsS3DataLoader(credentials=aws_creds)
client = loader._get_client(task_id="session123")
```

---

## Documentation for `AwsS3DataLoader._get_item`

### Functionality

Processes a downloaded file stored as a BytesIO object and returns it. By default, it returns the file unmodified. This method acts as a hook for subclasses to override for custom processing, such as decoding JSON or other file formats.

### Parameters

- `file`: An io.BytesIO object containing the downloaded file content.

### Return Value

- The processed data item. By default, the original BytesIO object is returned.

### Usage

- **Purpose**: To convert raw S3 file data into a format that can be further processed by the application.

#### Example

```python
# Example subclass overriding _get_item

def _get_item(self, file: io.BytesIO) -> dict:
    import json
    data = json.loads(file.read().decode('utf-8'))
    return data
```

---

## Documentation for `AwsS3DataLoader._get_data_from_s3`

### Functionality

Main method to download and process data from AWS S3. It accepts a list of S3FileMeta objects representing files to download and process. It connects to AWS S3 using a generated client, downloads files, and uses caching to avoid redundant downloads.

### Parameters

- `files`: A list of S3FileMeta objects with metadata for each file.
- `ignore_failures`: Boolean flag. If True, continues on failure; otherwise, raises an exception.

### Usage

- **Purpose**: Retrieve and process data from AWS S3 with error handling.

#### Example

```python
files = [
    S3FileMeta(bucket="mybucket", file="file1.csv"),
    S3FileMeta(bucket="mybucket", file="file2.csv")
]

data = list(loader._get_data_from_s3(files))
for d, meta in data:
    print(d)
```

---

## Documentation for `AwsS3DataLoader._process_file_meta`

### Functionality

Processes a single file metadata to download a file from AWS S3 and prepare its associated data objects. It uses the provided S3 client to download data (if not cached), handles errors based on the ignore_failures flag, and yields tuples containing a data dictionary and its corresponding file metadata.

### Parameters

- `s3_client`: The boto3 S3 client configured to access AWS S3.
- `file_meta`: An instance of S3FileMeta containing metadata for the file.
- `ignore_failures`: A boolean flag that, if True, prevents the raising of exceptions on failure.
- `uploaded`: A dictionary cache for downloaded files, using a tuple of (bucket, file) as the key.

### Usage

- **Purpose**: To download a file from S3 based on its metadata and yield processed data objects for further use.

#### Example

Assuming `file_meta` is a valid S3FileMeta object and `s3_client` is set up:

```python
for data, meta in loader._process_file_meta(
    s3_client, file_meta, ignore_failures=True, uploaded={}
):
    print(data, meta)
```

---

## Documentation for `AwsS3DataLoader._download_and_get_item`

### Functionality

This method downloads a file from S3 if it is not already cached. It uses the provided S3 client to fetch the file as a stream and processes it into a usable data item. The downloaded item is then stored in a cache dictionary to avoid redundant downloads.

### Parameters

- `s3_client`: Configured boto3 S3 client for S3 operations.
- `file_meta`: S3FileMeta object containing bucket and file path info.
- `uploaded`: Cache dictionary with keys as (bucket, file) and values as the downloaded item.

### Usage

- **Purpose**: Retrieve and cache file data from AWS S3 storage.

#### Example

Assuming `s3_client` is configured and `file_meta` is an S3FileMeta:

```python
uploaded = {}
item = loader._download_and_get_item(s3_client, file_meta, uploaded)
```

---

## Documentation for `AwsS3DataLoader._yield_item_objects`

### Functionality

Yields data objects based on item data and its S3 metadata. It handles both lists and single items.

### Parameters

- `item`: Downloaded or retrieved item data.
- `file_meta`: S3FileMeta object with item metadata.

### Usage

- **Purpose**: Generates tuples of item dictionary and metadata. Use this method to iterate over items from S3 files.

#### Example

Assume `item` is a list:

```python
for obj, meta in loader._yield_item_objects(item, file_meta):
    print(obj, meta)
```

---

## Documentation for `AwsS3DataLoader._create_item_object`

### Functionality

This method creates a dictionary object from the item data and associates it with its S3 metadata. If the loader's features property is not set or if the item is not a dictionary, the item is stored under the key "item". Otherwise, the item dictionary is merged into the resulting dictionary.

### Parameters

- `item`: The data content to be converted into a dictionary.
- `file_meta`: An S3FileMeta object containing metadata (e.g., the item's id).

### Usage

- **Purpose**: Encapsulates raw item data and metadata into a uniform tuple for further processing.

#### Example

Suppose an item and metadata are provided as follows:

```python
item = {"name": "example"}
file_meta.id = "123"
```

The method returns a tuple like:

```python
({"item_id": "123", "name": "example"}, file_meta)
```

---

## Documentation for `AwsS3DataLoader._generate_dataset_from_s3`

### Functionality

This generator function converts a list of S3 file metadata into a stream of item dictionaries. It iterates over S3 data and yields each item, making it suitable for dataset creation.

### Parameters

- `files`: List of S3FileMeta objects representing the files to load.

### Usage

- **Purpose**: Transforms S3 file metadata into a dataset generator, enabling easy loading into a Hugging Face Dataset.

#### Example

```python
s3_loader = AwsS3DataLoader(
    retry_config=my_retry_config,
    features=my_features
)
dataset = Dataset.from_generator(
    lambda: s3_loader._generate_dataset_from_s3(s3_files),
    features=my_features
)
```

---

## Documentation for `AwsS3DataLoader.load`

### Functionality

This method loads data from S3 files into a Hugging Face Dataset. It converts a list of S3FileMeta objects into a dataset using the Dataset.from_generator function. The method reads file contents and builds a dataset with the provided features.

### Parameters

- `items_data (List[S3FileMeta])`: A list of S3FileMeta objects describing the files to load from AWS S3.

### Return Value

- `Dataset`: A Hugging Face Dataset object containing the loaded data.

### Usage

- **Purpose**: To efficiently load and structure data from AWS S3 for analysis or machine learning tasks.

#### Example

```python
loader = AwsS3DataLoader(features=features)
dataset = loader.load(items_data)
```

---

## Documentation for `AwsS3DataLoader.load_items`

### Functionality

Loads individual items from S3 files and returns a list of DownloadedItem objects containing both data and metadata. Unlike load(), this method creates a list rather than a Dataset.

### Parameters

- `items_data (List[S3FileMeta])`: List of S3 file metadata to load.

### Usage

- **Purpose**: To load individual items from S3 and wrap them in a DownloadedItem structure.

#### Example

```python
# Initialize loader with necessary configuration
loader = AwsS3DataLoader(retry_config=retry_conf, features=features)
items = loader.load_items(items_data)
```

---

## Documentation for `AwsS3DataLoader._load_batch_with_offset`

### Functionality

This method loads a batch of files from AWS S3 using pagination. It connects to S3, retrieves files starting at a specified offset, and returns a list of DownloadedItem objects containing the file key, content, and metadata.

### Parameters

- `offset`: The starting offset used as the token for pagination.
- `batch_size`: The maximum number of files to load in the batch.
- `**kwargs`: Additional keyword arguments, including the key `bucket` for the S3 bucket name.

### Usage

- **Purpose:** Internally used to load files in manageable chunks from an S3 bucket with retry and error handling.

#### Example

```python
loader = AwsS3DataLoader(...)
batch = loader._load_batch_with_offset(0, 10, bucket="my-s3-bucket")
for item in batch:
    print(item.id, item.data)
```

---

## Documentation for `AwsS3DataLoader.load_all`

### Functionality

This method is a generator that continuously retrieves data batches from one or more S3 buckets. Each iteration yields a batch of downloaded items, processing them in manageable chunks.

### Parameters

- `batch_size`: Defines the number of items in each batch.
- `**kwargs`: Additional keyword arguments. Should include:
  - `buckets`: List of S3 bucket names to load files from.

### Usage

- **Purpose**: Load large S3 datasets in smaller batches, improving memory usage and enabling incremental processing.

#### Example

```python
loader = AwsS3DataLoader()
for batch in loader.load_all(batch_size=10, buckets=["my-bucket"]):
    # Process each batch of downloaded items
    process(batch)
```

---

## Documentation for `AwsS3DataLoader.total_count`

### Functionality

Returns the total count of items available in the S3 bucket. The base method returns None because S3 does not provide an efficient way to count objects without listing them.

### Parameters

- `kwargs`: Additional parameters for configuration, such as the S3 bucket name.

### Usage

- **Purpose**: Retrieve the item count when available using S3 APIs.

#### Example

```python
# Example override implementation
def total_count(self, **kwargs) -> Optional[int]:
    try:
        response = self._get_client(str(uuid.uuid4())).list_objects_v2(
            Bucket=kwargs['bucket'], MaxKeys=0
        )
        return response.get('KeyCount', 0)
    except Exception:
        return None
```
