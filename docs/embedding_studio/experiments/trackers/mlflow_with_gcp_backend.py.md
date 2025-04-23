## Documentation for `ExperimentsManagerWithGCPBackend`

### Functionality
This class provides a wrapper over the mlflow package to manage fine-tuning experiments with a Google Cloud backend. It integrates MLFlow tracking with GCP services, managing artifacts via Google Cloud Storage and applying retry strategies on API errors.

### Parameters
- `tracking_uri`: URL of the MLFlow server.
- `gcp_credentials`: GCP credentials (Pydantic model with project and bucket details).
- `main_metric`: Primary metric name for evaluating experiments.
- `plugin_name`: Name of the fine-tuning method used.
- `accumulators`: List of metric accumulators for logging.
- `is_loss`: Boolean indicating if the main metric represents a loss.
- `n_top_runs`: Number of top parameter groups to consider.
- `requirements`: Extra requirements for logging models to MLflow.
- `retry_config`: Configuration for retry policies.

### Usage
Purpose: To manage MLFlow experiments by leveraging GCP storage for artifact management, while extending core experiment functionalities.

Inheritance: Inherits from `ExperimentsManager`, adding GCP backend integration.

#### Example

```python
manager = ExperimentsManagerWithGCPBackend(
    tracking_uri="http://mlflow.server",
    gcp_credentials=GCPCredentials(
        project_id="your_project_id",
        bucket_name="your_bucket_name",
        client_email="your_email",
        private_key="your_private_key",
        private_key_id="your_key_id"
    ),
    main_metric="accuracy",
    plugin_name="fine_tuning",
    accumulators=[],
    is_loss=False,
    n_top_runs=10,
    requirements=None,
    retry_config=None
)
```

---

## Documentation for `ExperimentsManagerWithGCPBackend.is_retryable_error`

### Functionality

This method checks if an exception encountered from Google Cloud Storage is retryable. Specifically, if the exception is a GoogleAPIError with HTTP status code 500 or above, it logs an error message and returns True. Otherwise, it returns False.

### Parameters

- `e`: Exception instance to check. Usually a GoogleAPIError object.

### Usage

- **Purpose** - Determine if a GCP error is retryable for retry logic.

#### Example

```python
from google.api_core.exceptions import GoogleAPIError

# Example of using is_retryable_error
error = GoogleAPIError('Server error', code=500)
if instance.is_retryable_error(error):
    # implement retry mechanism
    pass
```

---

## Documentation for `ExperimentsManagerWithGCPBackend._delete_model`

### Functionality

This method deletes model artifact files stored in Google Cloud Storage. It uses the MLflow run information to obtain the artifact URI, extracts the object path using the provided GCP bucket name, and deletes the file from the bucket using the GCP client.

### Parameters

- `run_id`: Identifier for the experiment run used to obtain model info.
- `experiment_id`: Identifier for the experiment. Although not currently used, it may be used for logging or future extensions.

### Usage

- **Purpose**: Remove a stored model's files after experiment cleanup.

#### Example

```python
manager = ExperimentsManagerWithGCPBackend(
    tracking_uri,
    gcp_credentials,
    main_metric,
    plugin_name,
    accumulators
)
success = manager._delete_model(run_id="run_123", experiment_id="exp_456")
if success:
    print("Model deleted successfully.")
```