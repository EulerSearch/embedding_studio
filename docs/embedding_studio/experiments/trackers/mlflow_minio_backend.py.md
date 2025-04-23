# Documentation for `ExperimentsManagerWithMinIOBackend`

## Overview
`ExperimentsManagerWithMinIOBackend` is a wrapper around MLFlow designed for fine-tuning experiments with a MinIO backend. It connects to the MLFlow server and manages models stored in a MinIO bucket, integrating error handling and retry policies specific to MinIO operations.

### Parameters
- `tracking_uri` (str): URL of the MLFlow server.
- `minio_credentials` (dict): Credentials to connect to MinIO.
- `main_metric` (str): Primary metric to determine the best model.
- `plugin_name` (str): Name of the fine-tuning method.
- `accumulators` (list): List of metrics accumulators for logging.
- `is_loss` (bool): Flag indicating if the main metric should be minimized.
- `n_top_runs` (int): Maximum number of top runs to consider.
- `requirements` (list): Extra requirements for MLFlow model logging.
- `retry_config` (dict): Retry policy configuration.

### Usage
- **Purpose**: Manage fine-tuning experiments with a MinIO storage backend. This class inherits from `ExperimentsManager` to reuse common tracking features.

#### Example
```python
manager = ExperimentsManagerWithMinIOBackend(
    "http://mlflow-server",
    minio_credentials,
    "accuracy",
    "fine-tuning",
    [MetricsAccumulator()],
    is_loss=False,
    n_top_runs=10
)
manager._delete_model("run-id", "experiment-id")
```

## `ExperimentsManagerWithMinIOBackend.is_retryable_error`

### Functionality
Checks if an exception is a retryable error based on server error conditions. If the error is of type `ServerError` and its status code is between 500 and 599, it is considered retryable.

### Parameters
- `e` (Exception): The exception encountered during an operation.

### Returns
- `bool`: True if the error is retryable; otherwise, False.

### Usage
- **Purpose**: To determine if an error from MinIO operations might be temporary and thus eligible for a retry.

#### Example
```python
try:
    perform_operation()
except Exception as e:
    if manager.is_retryable_error(e):
        retry_operation()
    else:
        handle_failure()
```

## `ExperimentsManagerWithMinIOBackend._delete_model`

### Functionality
Deletes model files stored on MinIO using the MLFlow artifact URI. It extracts the object path from the artifact URI and calls the MinIO client to remove the object from the designated bucket. The method logs the outcome and returns True if the deletion was successful, or False if an error occurred.

### Parameters
- `run_id` (str): The MLFlow run identifier to locate the model.
- `experiment_id` (str): The MLFlow experiment identifier (not used in deletion).

### Usage
- **Purpose**: To remove stored model files on MinIO for a given MLFlow run.

#### Example
Given a valid `run_id` and `experiment_id`, the method retrieves run info, computes the object path, and attempts to remove the model from the bucket. A successful deletion returns True, while a `NoSuchKey` error returns False.