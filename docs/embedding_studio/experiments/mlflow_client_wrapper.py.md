# Documentation for MLflowClientWrapper

### Functionality
This class serves as a wrapper around the MLflow client to provide enhanced operations with built-in retry functionality. It simplifies managing models, experiments, and runs by handling temporary errors (especially MLflow RestExceptions) securely.

### Motivation
The wrapper addresses limitations in the native MLflow client, especially related to error handling during heavy model logging or unstable network conditions. The retry feature improves reliability and user experience when interacting with the tracking server.

### Inheritance
MLflowClientWrapper does not inherit from any custom classes; it extends Python's base object. Its design focuses on augmenting MLflow client capabilities without modifying the original behavior.

## Documentation for `MLflowClientWrapper.tracking_uri`

### Functionality
Returns the MLflow tracking URI associated with the client wrapper. This URI specifies the MLflow server used for tracking experiments.

### Parameters
This property does not accept any parameters.

### Usage
Use this property to retrieve the MLflow tracking URI as a string. It is useful for confirming or debugging the server configuration.

#### Example
```python
client_wrapper = MLflowClientWrapper(tracking_uri='http://localhost:5000')
print(client_wrapper.tracking_uri)
```

## Documentation for `MLflowClientWrapper.requirements`

### Functionality
This property returns the list of requirement strings used for logging models with MLflow.

### Parameters
This is a read-only property and does not accept parameters.

### Usage
- **Purpose** - Retrieve dependency details for model logging.

#### Example
```python
from embedding_studio.experiments.mlflow_client_wrapper import MLflowClientWrapper

client = MLflowClientWrapper("http://mlflow.server")
reqs = client.requirements
print(reqs)
```

## Documentation for `MLflowClientWrapper._get_default_retry_config`

### Functionality
Creates a default retry configuration for MLflow operations. It builds a `RetryConfig` object using default settings from the global configuration. Each MLflow operation is assigned specific retry parameters like maximum attempts and wait time in seconds.

### Parameters
This method does not require any parameters.

### Returns
- A `RetryConfig` object with default retry settings for MLflow operations, including:
  - log_metric
  - log_param
  - log_model
  - load_model
  - delete_model
  - search_runs
  - list_artifacts
  - end_run
  - get_run
  - search_experiments
  - delete_experiment
  - rename_experiment
  - create_experiment
  - get_experiment

### Usage
Call this method to retrieve default retry settings for handling intermittent communication failures with the MLflow tracking server.

#### Example
```python
from embedding_studio.experiments.mlflow_client_wrapper import MLflowClientWrapper
retry_config = MLflowClientWrapper._get_default_retry_config()
print(retry_config)
```

## Documentation for `MLflowClientWrapper._get_base_requirements`

### Functionality
Generates the base requirements needed for model logging by running the "poetry export" command in the current directory. It returns a list of dependency strings exported from the project configuration.

### Parameters
This method does not accept any parameters.

### Usage
- **Purpose**: Extract base dependencies from the project's poetry configuration to ensure correct dependency versions for model logging.

#### Example
```python
client = MLflowClientWrapper(tracking_uri="http://127.0.0.1")
requirements = client._get_base_requirements()
```
*Note: If the poetry export command fails, an empty list is returned.*

## Documentation for `MLflowClientWrapper._get_model_exists_filter`

### Functionality
Returns a filter string to find runs with uploaded models. The returned string "metrics.model_uploaded = 1" is used in MLflow run queries to filter runs where the model has been successfully uploaded.

### Parameters
This method does not accept any parameters.

### Usage
Call this method to obtain the filter expression when listing MLflow runs. It helps narrow down the results to only those runs that contain an uploaded model.

#### Example
```python
client = MLflowClientWrapper(tracking_uri="http://localhost")
filter_str = client._get_model_exists_filter()
# Use filter_str when querying runs in MLflow.
```

## Documentation for `MLflowClientWrapper._get_artifact_url`

### Functionality
Generates a URL to access artifacts stored in MLflow. It constructs the URL using the provided run ID and artifact path, which can then be used to retrieve the artifact from the MLflow server.

### Parameters
- `run_id`: ID of the run containing the artifact. (str)
- `artifact_path`: Path to the artifact within the run. (str)

### Returns
- A string representing the URL for accessing the specified artifact.

### Usage
- **Purpose**: Generate a URL to fetch a specific artifact from MLflow.

#### Example
```python
client = MLflowClientWrapper(tracking_uri="http://mlflow.server")
url = client._get_artifact_url(run_id="abc123", artifact_path="model/data/model.pth")
print(url)
```

## Documentation for `MLflowClientWrapper._check_artifact_exists`

### Functionality
Checks if an artifact exists within a run. It lists artifacts and matches the provided artifact path.

### Parameters
- `run_id`: The ID of the run to check for the artifact.
- `artifact_path`: The path to the artifact within the run.

### Usage
- **Purpose**: Validate existence of an artifact in a run to ensure it is uploaded or available.

#### Example
```python
client = MLflowClientWrapper(tracking_uri="http://mlflow.server")
exists = client._check_artifact_exists("run_id_example", "artifact/path")
```

## Documentation for `MLflowClientWrapper._get_experiment_id_by_name`

### Functionality
This method retrieves the experiment ID given an experiment name. It leverages the utility function `get_experiment_id_by_name` to perform the lookup.

### Parameters
- `experiment_name`: A string specifying the name of the experiment.

### Usage
- **Purpose**: Obtain the unique identifier for an experiment by its name.

#### Example
```python
wrapper = MLflowClientWrapper(tracking_uri="http://localhost:5000")
exp_id = wrapper._get_experiment_id_by_name("SampleExp")
```

## Documentation for `MLflowClientWrapper._get_run_id_by_name`

### Functionality
This method retrieves the run ID for a given experiment ID and run name by querying the MLflow backend. It helps in identifying a run within an experiment when only the name is known.

### Parameters
- `experiment_id` (str): ID of the experiment containing the run.
- `run_name` (str): Name of the run to search for.

### Usage
- **Purpose**: Lookup the run identifier using the experiment ID and run name, typically used when further run details are needed.

#### Example
```python
client = MLflowClientWrapper()
run_id = client._get_run_id_by_name("exp123", "run_example")
if run_id is not None:
    print("Run found:", run_id)
else:
    print("Run not found")
```

## Documentation for `MLflowClientWrapper.is_retryable_error`

### Functionality
Determines if the provided exception is a retryable MLflow error. This method always returns False.

### Parameters
- `e`: Exception instance to be checked for retryability.

### Usage
- **Purpose** - Decide if an operation should be retried based on the encountered exception.

#### Example
```python
wrapper = MLflowClientWrapper("http://mlflow_server")
if wrapper.is_retryable_error(Exception("error")):
    # Implement retry logic
    pass
```

## Documentation for `get_runs`

### Functionality
This method retrieves runs for a given experiment. If the models_only flag is set to True, it returns only runs that have an associated model. Otherwise, it returns all runs for the specified experiment as a pandas DataFrame.

### Parameters
- `experiment_id`: A string representing the ID of the experiment.
- `models_only`: A boolean flag indicating whether to return only runs with an uploaded model.

### Usage
Use this method to query runs from an experiment. For example:

```python
wrapper = MLflowClientWrapper(tracking_uri="http://your_mlflow_server")
runs_df = wrapper.get_runs(experiment_id="exp_123", models_only=True)
print(runs_df.head())
```

## Documentation for `MLflowClientWrapper._download_model_by_run_id`

### Functionality
Downloads and loads an embedding model from a given MLflow run. It builds the model URI using the provided run ID and logs the process. The method uses MLflow's PyTorch loader to return the model.

### Parameters
- `run_id`: A string representing the ID of the run containing the model.

### Usage
This method is used internally to fetch and load a logged model from MLflow by its run ID. It is useful when you need to retrieve a model for further evaluation or inference.

#### Example
```python
client = MLflowClientWrapper(tracking_uri="http://localhost:5000")
model = client._download_model_by_run_id("1234567890abcdef")
```

## Documentation for `MLflowClientWrapper._delete_model`

### Functionality
This method attempts to delete a model for a given run. Since MLflow does not provide a direct API to remove a logged model, the method logs a warning and returns False.

### Parameters
- `run_id`: ID of the run containing the model to delete.
- `experiment_id`: ID of the experiment associated with the run.

### Usage
- **Purpose**: Use this method in workflows where you need to mark a model as deleted. It serves as a placeholder for a custom deletion implementation.

#### Example
```python
wrapper = MLflowClientWrapper("http://tracking.server")
success = wrapper._delete_model("run_123", "exp_456")
if not success:
    print("Model deletion is not implemented.")
```

## Documentation for `MLflowClientWrapper._get_run`

### Functionality
This method retrieves run information for a given run ID using MLflow. It calls `mlflow.get_run` and handles any `RestException` that occurs. If a run is not found (HTTP 404), it logs an error; otherwise, it raises the exception.

### Parameters
- `run_id` (str): The ID of the run to retrieve.

### Return Value
- Returns an instance of `mlflow.entities.Run` containing the run details.

### Usage
- **Purpose**: To obtain detailed information of an MLflow run for processing or inspection.

#### Example
```python
client_wrapper = MLflowClientWrapper(tracking_uri="http://mlflow.server")
run = client_wrapper._get_run("some_run_id")
print(run)
```

## Documentation for `MLflowClientWrapper._set_model_as_deleted`

### Functionality
Mark a model as deleted in MLflow by updating model-related metrics. It logs `model_deleted` (set to 1) and `model_uploaded` (set to 0) to indicate the model is deleted. This update helps in managing model lifecycle within MLflow.

### Parameters
- `run_id`: ID of the run containing the model.
- `experiment_id`: ID of the experiment for the given run.

### Usage
- **Purpose** - Mark a model as deleted in MLflow for proper lifecycle management.

#### Example
```python
client = MLflowClientWrapper(tracking_uri="http://localhost:5000")
client._set_model_as_deleted(run_id="abc123", experiment_id="exp456")
```

## Documentation for `MLflowClientWrapper.get_model_url_by_run_id`

### Functionality
Retrieves the download URL for the model artifact from a finished MLflow run. The method searches for a run that meets criteria for model existence and returns the URL for the artifact. If no valid run is found, the method returns None.

### Parameters
- `run_id` (str): The unique identifier of an MLflow run that contains a model.

### Returns
- Optional[str]: The URL to download the model artifact, or None if the model is not found.

### Usage
- **Purpose**: To obtain the URL required for downloading a model from a completed MLflow run.

#### Example
```python
wrapper = MLflowClientWrapper(tracking_uri)
model_url = wrapper.get_model_url_by_run_id("run_id")
```

## Documentation for `MLflowClientWrapper._archive_experiment`

### Functionality
Archives an experiment by renaming it with an "_archive" suffix. This method uses the MLflow client's rename_experiment function to update the experiment's name, effectively marking it as archived.

### Parameters
- `experiment_id` (str): The ID of the experiment to be archived.

### Usage
- **Purpose**: Archive an experiment to move it to a legacy state. This is useful when an experiment should no longer be active but kept for record purposes.

#### Example
```python
client = MLflowClientWrapper(tracking_uri="http://localhost")
client._archive_experiment("12345")
```

## Documentation for `MLflowClientWrapper._delete_experiment`

### Functionality
Deletes an experiment from MLflow. This method calls the MLflow API to remove an experiment using the supplied experiment ID, with built-in retry mechanisms to handle transient failures.

### Parameters
- `experiment_id`: A string representing the ID of the experiment to delete.

### Usage
- **Purpose**: Use this function to remove obsolete or unwanted experiments.

#### Example
```python
wrapper = MLflowClientWrapper(tracking_uri="http://localhost")
wrapper._delete_experiment("12345")
```

## Documentation for `MLflowClientWrapper.get_params_by_run_id`

### Functionality
This method retrieves fine-tuning parameters from a specific MLflow run. It searches for run data using the provided run ID and returns the parsed fine-tuning parameters as a FineTuningParams object, or None if not found.

### Parameters
- `run_id`: A string representing the unique ID of the MLflow run.

### Returns
- An instance of FineTuningParams if the run exists, otherwise None.

### Usage
- **Purpose**: Fetch fine-tuning parameters by specifying the run ID.

#### Example
```python
from embedding_studio.experiments.mlflow_client_wrapper import MLflowClientWrapper

mlflow_wrapper = MLflowClientWrapper('http://localhost:5000')
params = mlflow_wrapper.get_params_by_run_id('12345')
if params is not None:
    print('Parameters obtained:', params)
```

## Documentation for `MLflowClientWrapper.get_iteration_by_id`

### Functionality
This method retrieves fine-tuning iteration info from a run using its run ID. It searches all experiments to locate the matching run and extracts the experiment name. The method then parses the experiment name to generate fine-tuning iteration details.

### Parameters
- `run_id`: Identifier of the MLflow run (str).

### Usage
- **Purpose**: Retrieve fine-tuning iteration info from a run.

#### Example
```python
wrapper = MLflowClientWrapper(tracking_uri="http://localhost:5000")
iteration = wrapper.get_iteration_by_id("<run_id>")
```

## Documentation for `MLflowClientWrapper.get_experiment_id`

### Functionality
This method retrieves the artifact URL for a given MLflow run. It searches all runs for the provided run_id, filters out only finished runs, and returns the URL to the model artifact if found. If no matching run is found or the run does not contain a model, it returns None.

### Parameters
- `run_id`: ID of the MLflow run whose artifact URL is to be retrieved.

### Usage
- **Purpose:** Locate and retrieve the artifact URL of a run, which points to the stored model data.

#### Example
```python
wrapper = MLflowClientWrapper(tracking_uri="http://localhost:5000")
artifact_url = wrapper.get_experiment_id("1234567890")
if artifact_url:
    print(f"Artifact URL: {artifact_url}")
else:
    print("Run not found or model not available.")
```