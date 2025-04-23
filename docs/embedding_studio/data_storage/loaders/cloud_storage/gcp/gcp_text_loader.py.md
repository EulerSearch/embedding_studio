## Documentation for `GCPTextLoader`

### Functionality
GCPTextLoader loads and decodes text files stored in Google Cloud Platform's Cloud Storage. It reads byte streams and converts them into strings using a specified encoding. The method `_get_item` specifically extracts text from a given `io.BytesIO` file object and decodes it using the loader's specified encoding, ensuring that the text content is properly read from cloud storage.

### Inheritance
This class inherits from GCPDataLoader, which handles GCP credentials, connection, and retry logic. It builds on GCPDataLoader's functionality to specialize in text processing.

### Motivation
The main purpose of GCPTextLoader is to abstract the complexity of accessing and reading text files from GCP Cloud Storage. It integrates retry strategies and custom decoding to simplify text extraction.

### Parameters
- `retry_config`: Optional retry strategy configuration.
- `features`: Optional schema defining the expected data.
- `encoding`: Character encoding for the text (default: "utf-8").
- `kwargs`: Additional arguments for GCP credentials.
- `file`: An `io.BytesIO` object containing the text data to be decoded.

### Usage
- **Purpose**: Load and decode text files stored in GCP and retrieve the text data from files stored in Google Cloud Platform's Cloud Storage.

#### Example
```python
from io import BytesIO
from embedding_studio.data_storage.loaders.cloud_storage.gcp.gcp_text_loader import GCPTextLoader

# Initialize loader with optional retry configuration
loader = GCPTextLoader(retry_config=None, encoding="utf-8")

# Suppose 'file' is an io.BytesIO object from GCP Storage
data = loader._get_item(BytesIO(b"example text"))
print(data)
```