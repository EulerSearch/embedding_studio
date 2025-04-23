# Documentation for `ExperimentsManagerWithAmazonS3Backend`

## Functionality
The `ExperimentsManagerWithAmazonS3Backend` class provides a thin wrapper over the MLflow package to track fine-tuning experiments while integrating with Amazon S3 for artifact management. It extends and enhances the basic features of `ExperimentsManager`, ensuring robust handling of model artifacts stored on Amazon S3 along with MLflow tracking. The class employs retry mechanisms to gracefully handle network or connection errors.

### Inheritance
- Inherits from `ExperimentsManager`, offering experiment tracking capabilities and model management.

### Parameters
- `tracking_uri`: URL of the MLflow server.
- `s3_credentials`: `S3Credentials` instance for connecting to Amazon S3.
- `main_metric`: Name of the primary metric used for model selection.
- `plugin_name`: Name of the fine-tuning plugin.
- `accumulators`: List of `MetricsAccumulator` for logging experiment metrics.
- `is_loss`: Boolean flag indicating if the main metric is a loss value.
- `n_top_runs`: Number of top runs considered for further tuning.
- `requirements`: Additional requirements for model logging (optional).
- `retry_config`: Optional configuration for retrying operations.

### Usage
- **Purpose:** Integrates MLflow experiment tracking with the Amazon S3 backend, ensuring proper handling of model artifacts.

#### Example
```python
from embedding_studio.experiments.trackers.mlflow_with_aws_s3_backend \
    import ExperimentsManagerWithAmazonS3Backend, S3Credentials
from embedding_studio.experiments.metrics_accumulator import MetricsAccumulator
from embedding_studio.workers.fine_tuning.utils.config import RetryConfig

s3_creds = S3Credentials(
    endpoint="s3.amazonaws.com", access_key="ABC", secret_key="XYZ",
    bucket_name="my-bucket", secure=True
)
accs = [MetricsAccumulator(...)]
manager = ExperimentsManagerWithAmazonS3Backend(
    tracking_uri="http://mlflow.server", s3_credentials=s3_creds,
    main_metric="accuracy", plugin_name="fine_tuning", accumulators=accs,
    is_loss=False, n_top_runs=10, requirements=None,
    retry_config=RetryConfig(...)
)
```

## Documentation for `ExperimentsManagerWithAmazonS3Backend._delete_model`

### Functionality
The `_delete_model` method deletes model files stored in an Amazon S3 bucket by identifying the S3 object using the MLflow artifact URI and invoking the S3 `delete_object` API.

### Parameters
- `run_id`: A string specifying the MLflow run identifier.
- `experiment_id`: A string specifying the experiment identifier. (Note: This parameter is not directly used for deletion.)

### Returns
- `bool`: True if deletion is successful, else False.

### Usage
- **Purpose:** Removes model artifacts from S3 storage to free up unused space or remove deprecated models.

#### Example
```python
manager = ExperimentsManagerWithAmazonS3Backend(...)
success = manager._delete_model("run123", "exp456")
if success:
    print("Deletion successful.")
```