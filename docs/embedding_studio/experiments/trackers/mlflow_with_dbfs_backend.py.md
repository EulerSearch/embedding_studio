# Documentation for `ExperimentsManagerWithDBFSBackend`

## Functionality

This class provides a specialized wrapper over the MLflow package, designed for managing fine-tuning experiments using a DBFS backend on Databricks. It integrates with MLflow tracking and leverages a DBFS client for file operations, such as deleting model artifacts, facilitating a smooth experiment management workflow.

### Main Purposes and Motivation

- Integrate MLflow tracking with Databricks DBFS for managing experiments.
- Enable direct file operations (e.g., deleting model files) on DBFS.
- Provide a foundation for handling experiment data in Databricks environments with an emphasis on fine-tuning tasks.

## Method: `ExperimentsManagerWithDBFSBackend._delete_model`

### Functionality

Deletes model files stored on DBFS. It retrieves the MLflow run details, extracts the DBFS path from the artifact URI, and requests deletion of the files.

### Parameters

- `run_id` (str): Unique identifier for the MLflow run.
- `experiment_id` (str): Identifier for the associated experiment.

### Usage

- **Purpose**: Remove DBFS model artifacts after a run's completion.

#### Example

```python
result = experiments_manager._delete_model(
    "run123", "exp456"
)
if result:
    print("Deleted successfully")
else:
    print("File not found")
```

### Inheritance

`ExperimentsManagerWithDBFSBackend` is a subclass of `ExperimentsManager`, extending its functionalities to work specifically with a DBFS backend.