## Documentation for AwsS3TextLoader

### Functionality

The AwsS3TextLoader class provides functionality for loading and decoding text files from AWS S3 storage. It extends the AwsS3DataLoader class to handle text-specific operations with configurable encoding.

### Parameters

- retry_config: Configuration for retry strategies during network calls.
- features: Dataset features that are expected.
- encoding: Character encoding for text decoding (default: "utf-8").
- kwargs: Additional keyword arguments, such as AWS credentials.

### Usage

```python
from embedding_studio.data_storage.loaders.cloud_storage.s3.s3_text_loader import AwsS3TextLoader

loader = AwsS3TextLoader(
    encoding="utf-8",
    aws_access_key_id="YOUR_KEY",
    aws_secret_access_key="YOUR_SECRET"
)
text_content = loader.load_items([
    S3FileMeta(bucket="my-bucket", file="document.txt")
])
```

## Documentation for AwsS3TextLoader._get_item

### Functionality

This method processes a downloaded BytesIO file by reading its data.

- file (io.BytesIO): A binary stream of the file from S3 that is to be decoded into a text string.

### Return Value

- str: The decoded text content from the BytesIO stream.

### Usage

- **Purpose:** Convert binary content from S3 into a text string by applying the loader's decoding mechanism.

#### Example

```python
from embedding_studio.data_storage.loaders.cloud_storage.s3.s3_text_loader import AwsS3TextLoader
import io

loader = AwsS3TextLoader(
    encoding="utf-8",
    aws_access_key_id="YOUR_KEY",
    aws_secret_access_key="YOUR_SECRET"
)

with open("document.txt", "rb") as f:
    text = loader._get_item(io.BytesIO(f.read()))

print(text)
``` 

### Parameters

- file (io.BytesIO): A binary stream of the file from S3 that is to be decoded into a text string.

### Return Value

- str: The decoded text content from the BytesIO stream.