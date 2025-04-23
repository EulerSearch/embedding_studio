# Documentation for BucketFileMeta

## Overview

The `BucketFileMeta` class extends `ItemMeta` to manage metadata for files stored in bucket-based systems (e.g., S3). It maintains essential information such as the bucket name, file path, and an optional index, ensuring unique identification of each item in storage.

### Parameters

- **bucket**: Name of the storage bucket.
- **file**: File path within the bucket.
- **index** (Optional[int]): Sub-item index (e.g., chunk or line).
- **object_id** (Optional[str]): Explicit ID for the item.
- **payload** (Optional[Dict[str, Any]]): Additional metadata.

### Functionality

The `derived_id` method returns a unique identifier for a file by concatenating the bucket name and file path. If an index is specified, it is appended to differentiate sub-items within the file.

### Returns

- A string that uniquely identifies a file within the bucket-based storage system.

### Usage

Use the `derived_id` method to generate unique keys for files stored in systems like S3. This ensures the identifier is unique across buckets and file paths.

#### Example

```python
meta = BucketFileMeta(
    bucket="example-bucket",
    file="path/to/file.txt",
    index=0,
    object_id="12345",
    payload={"size": 1024}
)
print(meta.derived_id)  # Outputs: example-bucket/path/to/file.txt:0
```

For a `BucketFileMeta` object with bucket "my_bucket" and file "data/file.txt", the `derived_id` method would return "my_bucket/data/file.txt" if no index is set. If the index is 2, it would return "my_bucket/data/file.txt:2".