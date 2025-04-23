# Documentation for ExperimentsManagerWithAzureBlobBackend

## Functionality
The `ExperimentsManagerWithAzureBlobBackend` class wraps mlflow for managing fine-tuning experiments that use Azure Blob Storage to store artifacts robustly. It also employs retry logic to handle transient failures when performing operations like deletion of blobs.

### Inheritance
Inherits from `ExperimentsManager`.

### Motivation
Designed to combine mlflow experiment tracking with cloud-scale storage in Azure Blob Storage. This approach promotes scalability and fault tolerance in handling experiment data.

### Main Purposes
- Manage experiment tracking with mlflow.
- Store experiment artifacts in Azure Blob Storage.
- Employ retries to overcome transient backend errors.

## Method: is_retryable_error

### Functionality
Checks if an exception is retryable based on its HTTP status code. It returns True if the exception is an instance of `HttpResponseError` and its status code is between 500 and 599, indicating a transient server error that can be retried; otherwise, it returns False.

### Parameters
- `e`: Exception - The error instance to evaluate for retryability.

### Usage
- **Purpose**: Validate whether an error should be retried following a transient server failure.

#### Example
Suppose `error` is an instance of `HttpResponseError` with status 503.

```python
manager = ExperimentsManagerWithAzureBlobBackend(...)
manager.is_retryable_error(error)
```
Returns: `True`

## Method: _delete_model

### Functionality
This method deletes model artifact files stored in Azure Blob Storage for a given MLflow run. It initializes a `BlobServiceClient` using the provided Azure Blob credentials and extracts the blob path from the run's artifact URI. The method then deletes the blob and returns True if the operation is successful. In case the blob does not exist, it logs an error and returns False.

### Parameters
- `run_id`: The MLflow run identifier used to retrieve run information.
- `experiment_id`: The experiment identifier. Currently, this parameter is unused and reserved for potential future use.

### Usage
- **Purpose**: To remove stored model artifacts related to a specific run. This ensures that outdated or unnecessary files are cleaned from Azure Blob Storage.

#### Example
Assuming valid Azure Blob credentials and a valid run ID, you can delete model artifacts as follows:

```python
result = experiments_manager._delete_model("run123", "exp456")
if result:
    print("Deletion succeeded")
else:
    print("Deletion failed")
```

### Example Initialization
```python
import mlflow
from azure.storage.blob import BlobServiceClient

# Define Azure credentials
creds = AzureBlobCredentials(
    account_name="your_account",
    account_key="your_key",
    container_name="your_container"
)

manager = ExperimentsManagerWithAzureBlobBackend(
    tracking_uri="http://mlflow.server",
    azure_blob_credentials=creds,
    main_metric="accuracy",
    plugin_name="example_plugin",
    accumulators=[]
)

# Example usage: delete an artifact for a run.
manager._delete_model(run_id="run123", experiment_id="exp123")
```