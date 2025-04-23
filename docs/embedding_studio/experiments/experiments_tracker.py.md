## Documentation for `ExperimentsManager`

### Functionality

ExperimentsManager is a wrapper over the MLflow client that enables fine-tuning experiment management. It automates experiment creation, run tracking, and metrics logging for fine-tuning processes using MLflow.

### Motivation

The class is designed to simplify the setup and monitoring of fine-tuning experiments. By integrating with MLflow, it provides a structured way to track hyper-parameter tuning and experiment performance, ensuring consistency and reliability in logging important metrics.

### Inheritance

ExperimentsManager extends the MLflowClientWrapper class. This inheritance allows it to leverage standardized MLflow functionalities while offering customized features specific to fine-tuning experiments.

---

## Documentation for `ExperimentsManager.from_wrapper`

### Functionality

Creates a new ExperimentsManager instance from an existing MLflowClientWrapper. This method reuses the tracking URI, requirements, and retry configuration from the given wrapper, while adding additional parameters for fine-tuning.

- **main_metric**: Name of the main metric to evaluate runs.
- **plugin_name**: Fine-tuning method name used for naming experiments.
- **accumulators**: List of MetricsAccumulator for logging metrics.
- **is_loss**: Boolean flag. True if the main metric is a loss.
- **n_top_runs**: Number of top runs considered for tuning steps.

### Usage

- **Purpose**: Initialize an ExperimentsManager with settings from a given MLflowClientWrapper and specified tuning parameters.

#### Example

```python
from embedding_studio.experiments.experiments_tracker import (
    ExperimentsManager
)
from embedding_studio.experiments.mlflow_client_wrapper import (
    MLflowClientWrapper
)

# Assume mlflow_wrapper is an instance of MLflowClientWrapper
mgr = ExperimentsManager.from_wrapper(
    mlflow_wrapper,
    main_metric="accuracy",
    plugin_name="ExamplePlugin",
    accumulators=[],
    is_loss=False,
    n_top_runs=10
)
```

---

## Documentation for `ExperimentsManager.is_loss`

### Functionality

Returns True if the main metric is a loss metric (i.e., lower is better), and False otherwise. This property is set during initialization and used for comparing experiment run results.

### Parameters

This is a property method and does not accept any parameters apart from the implicit self.

### Usage

- **Purpose**: Determine whether the main metric represents a loss, so that lower values indicate better performance.

#### Example

Assuming you have an ExperimentsManager instance named exp_manager:

```python
if exp_manager.is_loss:
    print("Loss metric: lower values are better.")
else:
    print("Non-loss metric: higher values are better.")
```

---

## Documentation for `ExperimentsManager._fix_name`

### Functionality

Prefix a name with the plugin name. It is used to enforce a naming convention in the experiments tracking system.

### Parameters

- **name**: Base name to fix.

### Usage

- **Purpose**: To standardize experiment names by adding the plugin prefix.

#### Example

Suppose self._plugin_name is set to "MyPlugin" and name is "Experiment1". Calling _fix_name("Experiment1") returns "MyPlugin / Experiment1".

---

## Documentation for `ExperimentsManager.is_initial_run`

### Functionality

Check whether the given `run_id` corresponds to the initial run, which is treated as the baseline model run. The method retrieves the designated initial run ID and compares it with the provided run ID.

### Parameters

- **run_id** (str): Identifier of the run to be checked.

### Usage

- **Purpose**: Verify if a specific run is the initial run.

#### Example

```python
experiments_manager = ExperimentsManager(
    tracking_uri, main_metric, plugin_name, accumulators
)
if experiments_manager.is_initial_run("abc123"):
    print("This is the initial run.")
```

---

## Documentation for `ExperimentsManager.get_initial_run_id`

### Functionality

Returns the run ID of the initial model run by the manager. It retrieves the run ID using the experiment name and a fixed run name.

### Parameters

None.

### Returns

- **str**: The ID of the initial run.

### Usage

This method is part of the ExperimentsManager class. It is used to fetch the ID of the initial run from the MLflow server.

#### Example

```python
run_id = manager.get_initial_run_id()
```

---

## Documentation for `ExperimentsManager.has_initial_model`

### Functionality

This method checks whether an initial model exists by verifying that an initial experiment is present and has an associated run with the model artifact.

### Parameters

This method does not take any parameters.

### Usage

Call this method to confirm if the initial model was uploaded to MLflow. A return value of True indicates that the model is available; False means it is missing.

#### Example

```python
if experiments_manager.has_initial_model():
    print("Initial model exists")
else:
    print("No initial model found")
```

---

## Documentation for `ExperimentsManager.upload_initial_model`

### Functionality

Uploads the initial model instance to the MLflow server. This method finishes any ongoing iteration, manages experiment creation or restoration, and logs the model artifact if it has not been previously uploaded.

### Parameters

- **model**: An instance of EmbeddingsModelInterface representing the model to be logged. It should conform to the MLflow model logging API.

### Usage

Use this method to log your project's initial model. The method handles experiment setup and avoids duplicate uploads.

#### Example

```python
model = SomeEmbeddingsModel(...)
manager = ExperimentsManager(tracking_uri, main_metric, plugin_name, accumulators)
manager.upload_initial_model(model)
```

---

## Documentation for `ExperimentsManager.get_initial_model_run_id`

### Functionality

Retrieves the run ID associated with the initial model. If no initial model is available, logs an error and returns None.

### Parameters

This method does not accept external parameters.

### Usage

- **Purpose**: To obtain the unique run ID for the initial model in an experiment, allowing the system to track and reference it.

#### Example

```python
manager = ExperimentsManager(...)
run_id = manager.get_initial_model_run_id()
if run_id:
    print("Run ID:", run_id)
else:
    print("No initial model found.")
```

---

## Documentation for `ExperimentsManager.get_experiments`

### Functionality

This method retrieves all experiments that are related to the current plugin. It leverages MLflow's experiment search feature and filters experiments based on a name prefix that includes the plugin name and a predefined experiment prefix.

### Parameters

This method does not require any parameters.

### Returns

- A list of MLflow Experiment objects. Each Experiment corresponds to a fine-tuning iteration experiment from the current plugin.

### Usage

- **Purpose**: Retrieve a list of experiments that are associated with the plugin's fine-tuning process.

#### Example

```python
experiments = experiments_manager.get_experiments()
for experiment in experiments:
    print(experiment.name)
```

---

## Documentation for `ExperimentsManager.get_previous_iteration_id`

### Functionality

This method returns the ID of the previous iteration from logged MLflow experiments. It first checks if a current tuning iteration is set. If not, it logs a warning and returns None. Otherwise, it selects the experiment with the latest creation time as the previous iteration.

### Parameters

- None

### Return Value

- Returns the experiment ID as a string if found, or None if not.

### Usage

Obtain the previous iteration ID from an instance of ExperimentsManager. Use the returned ID to reference prior tuning iterations.

#### Example

```python
manager = ExperimentsManager(...)
prev_id = manager.get_previous_iteration_id()
if prev_id:
    print("Previous iteration ID:", prev_id)
else:
    print("No previous iteration found.")
```

---

## Documentation for `ExperimentsManager.get_last_iteration_id`

### Functionality

Returns the experiment ID of the last iteration before the current one. It excludes the current tuning iteration and selects the experiment with the latest creation time.

### Parameters

None.

### Usage

- **Purpose**: To fetch the identifier of a previous experiment.
- **Return Value**: A string representing the experiment ID, or None if not found.

#### Example

Assume you have an ExperimentsManager instance called manager. You can obtain the last iteration ID by:

```python
last_id = manager.get_last_iteration_id()
```

This value can be used to compare or reference past experiment states.

---

## Documentation for `ExperimentsManager.get_last_finished_iteration_id`

### Functionality

Retrieves the ID of the last finished iteration from the recorded experiments. It skips the current iteration or unset iteration and selects the most recent experiment based on creation time among those whose runs have all reached a FINISHED or FAILED status.

### Parameters

- self: Instance of ExperimentsManager. Uses internal properties to filter experiments.

### Returns

- Optional[str]: The ID of the last finished iteration, or None if no finished iteration is found.

### Usage

- **Purpose**: Determine the identifier of the most recent iteration that has completed all runs.

#### Example

```python
manager = ExperimentsManager(tracking_uri, main_metric, plugin_name, accumulators)
last_id = manager.get_last_finished_iteration_id()
if last_id:
    print("Last finished iteration:", last_id)
else:
    print("No finished iteration found")
```

---

## Documentation for `ExperimentsManager.delete_previous_iteration`

### Functionality

This method deletes the models for the previous fine-tuning iteration. It retrieves the previous experiment id, filters finished runs, deletes their corresponding models, archives the experiment, and finally deletes the experiment. If no previous iteration is found, a warning is logged.

### Parameters

This method does not require any parameters.

### Usage

- **Purpose**: Cleans up resources from a completed iteration, removing outdated models and experiments.

#### Example

```python
manager.delete_previous_iteration()
```

---

## Documentation for `ExperimentsManager._set_experiment_with_name`

### Functionality

This method sets the current experiment using a provided experiment name. It calls `mlflow.set_experiment` to either create a new experiment or retrieve an existing one, and then saves the experiment ID for tracking fine-tuning iterations.

### Parameters

- **experiment_name**: A string representing the name of the experiment to be set.

### Usage

Use this method to initialize or switch to a specific experiment by name during a fine-tuning session.

#### Example

```python
manager._set_experiment_with_name("My Experiment")
```

---

## Documentation for `ExperimentsManager._set_experiment_with_id`

### Functionality

Sets the current experiment in MLFlow using an experiment ID. It updates the tuning iteration ID and registers the experiment instance for further MLFlow operations.

### Parameters

- **experiment_id**: The unique identifier for the experiment.

### Usage

- **Purpose**: Switch to a specific experiment by providing its MLFlow experiment ID.

#### Example

Simple call:

```python
obj._set_experiment_with_id("123")
```

---

## Documentation for `ExperimentsManager.set_iteration`

### Functionality

Starts a new fine-tuning session by setting the current experiment based on the provided iteration details. If the plugin name does not match, an error is logged. If a previous iteration is active, it is finished before starting a new one.

### Parameters

- **iteration**: Instance of FineTuningIteration containing iteration info.

### Usage

- **Purpose**: Update the current fine-tuning iteration experiment.

#### Example

Assuming 'iteration' is a FineTuningIteration instance:

```python
manager.set_iteration(iteration)
```

---

## Documentation for `ExperimentsManager.finish_iteration`

### Functionality

This method marks the end of the current iteration in a fine-tuning process. It resets the tuning iteration to the initial experiment and updates the experiment id accordingly.

### Parameters

None.

### Returns

None.

### Usage

- **Purpose**: Ends the current iteration and resets the system to the base experiment state.

#### Example

Assuming an instance of ExperimentsManager named manager, the following call ends the current iteration.

```python
manager.finish_iteration()
```

---

## Documentation for `ExperimentsManager.get_run_by_id`

### Functionality

Retrieves run details by its ID using the MLflow API. It searches for runs and filters the results to return run information that matches the provided ID.

### Parameters

- **run_id**: Unique identifier for the run.
- **models_only**: Boolean flag indicating if only runs with models should be returned (default: False).

### Usage

- **Purpose**: To obtain detailed run information after logging, useful for diagnostics and monitoring.

#### Example

```python
from embedding_studio.experiments.experiments_tracker import ExperimentsManager

# Create an instance (fill in required arguments)
manager = ExperimentsManager(...)

# Retrieve run information by its ID
run_info = manager.get_run_by_id("run_id_example", models_only=True)
```

---

## Documentation for `ExperimentsManager._get_best_previous_run_id`

### Functionality

Get the best run id from the previous iteration. It compares the initial experiment id with the previous iteration id to decide if a valid run exists. If the initial id matches the previous iteration id or if no previous iteration exists, then it returns (None, True).

### Parameters

None.

### Returns

- **run_id**: Optional string, best run identifier.
- **is_initial**: Boolean, True if it is the first experiment iteration.

### Usage

- **Purpose**: Retrieve best run details from the previous experiment iteration.

#### Example

```python
run_id, is_initial = experiments_manager._get_best_previous_run_id()
if is_initial:
    print("No previous run available.")
else:
    print(f"Best previous run: {run_id}")
```

---

## Documentation for `ExperimentsManager.get_best_current_run_id`

### Functionality

Returns the best run ID for the current tuning iteration. If the current iteration is initial, it returns (None, True). Otherwise, it returns the best run ID along with False.

### Parameters

None.

### Returns

Tuple (run_id, is_initial) where:
- **run_id**: The best run identifier or None.
- **is_initial**: True if no previous iteration exists.

### Usage

- **Purpose**: Retrieve the best run ID for the current experiment iteration.

#### Example

```python
run_id, is_initial = manager.get_best_current_run_id()
if is_initial:
    print("No previous run available.")
else:
    print("Best run ID:", run_id)
```

---

## Documentation for `ExperimentsManager._start_run`

### Functionality

Starts a new MLflow run using fine-tuning parameters. This method sets the run parameters, retrieves a run ID based on the run's name and tuning iteration, and then starts an MLflow run via `mlflow.start_run`.

### Parameters

- **params** (FineTuningParams): Fine-tuning parameters for the run. It should include an identifier and necessary configuration details.

### Usage

- **Purpose**: Initialize a new MLflow run associated with a tuning iteration.

#### Example

```python
manager = ExperimentsManager(tracking_uri, main_metric, plugin_name, ...)
params = FineTuningParams(id="run_001", ...)
manager._start_run(params)
```

---

## Documentation for `ExperimentsManager._save_params`

### Functionality

This method saves all experiment parameters to MLflow. It logs parameters from two sources: the tuning iteration settings and the current run's parameters. List values are converted to comma-separated strings for logging.

### Parameters

No external parameters. This method uses internal state:
- **_tuning_iteration**: Parameters for the tuning iteration.
- **_run_params**: Parameters for the current run.

### Usage

- **Purpose**: Logs all experiment parameters to MLflow to facilitate tracking of experiment configurations and results.

#### Example

Assuming an instance called manager:

```python
manager._save_params()
```

---

## Documentation for `ExperimentsManager._set_run`

### Functionality

This method initializes an experiment run using the provided fine-tuning parameters. It calls `_start_run` and then checks if the run already exists. For a new run, it sets `run_id`, saves parameters, and logs an initial metric ('model_uploaded' set to 0).

### Parameters

- **params** (FineTuningParams): Fine-tuning parameters to initiate the run.

### Return Value

- Returns True if a new run is started, otherwise False.

### Usage

- **Purpose**: Sets up or resumes a run based on its existence.

#### Example

```python
run_started = experiments_manager._set_run(params)
if run_started:
    print("Started a new run.")
else:
    print("Run already exists.")
```

---

## Documentation for `ExperimentsManager.set_run`

### Functionality

Starts a new run using provided fine-tuning parameters. If a run already exists, it is finished before starting a new run. If the current iteration is initial, an error is raised to avoid misconfiguration.

### Parameters

- **params**: An instance of FineTuningParams containing the tuning parameters for the run.

### Usage

- **Purpose**: Initiate a new experiment run for a given iteration.

#### Example

Assuming `params` is a FineTuningParams object:

```python
manager = ExperimentsManager(
    tracking_uri,
    main_metric,
    plugin_name,
    accumulators
)
finished = manager.set_run(params)
```

---

## Documentation for `ExperimentsManager.finish_run`

### Functionality

Ends the current run in a fine-tuning experiment. It clears all accumulators and resets run parameters.

### Parameters

- **as_failed**: Boolean flag. If True, ends the run with a "FAILED" status.

### Usage

Use this method to terminate an active run and clear its state.

#### Example

```python
manager.finish_run(as_failed=True)
```

---

## Documentation for `ExperimentsManager.download_initial_model`

### Functionality

Downloads the initial embeddings model using MLflow. This method fetches the model associated with the "initial_model" run from the initial experiment.

### Parameters

This method does not require any parameters.

### Returns

- An instance of EmbeddingsModelInterface representing the model.

### Usage

- **Purpose**: Retrieve the initial model for starting experiments.

#### Example

```python
model = experiments_manager.download_initial_model()
```

---

## Documentation for `ExperimentsManager.download_model_by_run_id`

### Functionality

This method downloads an embeddings model based on the provided run ID. It utilizes an internal download function and logs any errors encountered during the process.

### Parameters

- **run_id**: A string representing the unique identifier of a run. It is used to locate and download the corresponding model.

### Usage

- **Purpose**: Retrieve an embeddings model for a given run ID. If downloading fails, the method returns None while logging the exception.

#### Example

```python
model = experiments_manager.download_model_by_run_id("123abc")
if model is not None:
    # Use the model for inference or further processing
    pass
else:
    # Handle the case where the model wasn't downloaded
    pass
```

---

## Documentation for `ExperimentsManager.download_model`

### Functionality

This method retrieves an embedding model using an experiment name and a run name. It first finds the experiment's ID based on the provided experiment name and then searches for the run ID within that experiment. If either the experiment or run is not found, the method logs a message and returns None. Otherwise, it downloads the model using a helper function.

### Parameters

- **experiment_name**: Name of the experiment to look up.
- **run_name**: Name of the run from which to download the model.

### Usage

- **Purpose**: To fetch an embedding model related to a specified run in an experiment.

#### Example

```python
# Initialize the experiments manager
manager = ExperimentsManager(...)

# Download the model from experiment and run
model = manager.download_model("experiment_A", "run_1")
```

---

## Documentation for `ExperimentsManager.download_last_model`

### Functionality

Downloads the best model from the last iteration of fine-tuning. If no valid run is found, the method returns the initial model to ensure that a model is always available.

### Parameters

This method does not accept any parameters.

### Usage

- **Purpose**: Retrieve the embedding model from the best previous run for evaluation or inference.

#### Example

```python
manager = ExperimentsManager(...)
model = manager.download_last_model()
```

---

## Documentation for `ExperimentsManager.download_best_model`

### Functionality

This method retrieves the best embedding model from a given experiment by selecting the run with the highest quality metric. If no valid run is found, it falls back to returning the initial model.

### Parameters

- **experiment_id**: ID of the experiment from which to retrieve the best model.

### Usage

- **Purpose**: Fetch the best model from a previous iteration's completed runs. If no finished run is available, the method returns the initial model.

#### Example

```python
manager = ExperimentsManager(
    tracking_uri, main_metric, plugin_name, accumulators
)
model = manager.download_best_model("exp123")
```

---

## Documentation for `ExperimentsManager._set_model_as_deleted`

### Functionality

Marks a model as deleted in an experiment by logging metrics using MLflow. The method updates the run metrics to indicate that the model has been deleted and ensures subsequent model retrieval returns the proper model.

### Parameters

- **run_id**: ID of the run containing the model.
- **experiment_id**: ID of the experiment containing the run.

### Usage

- **Purpose**: Use this method to mark an uploaded model as no longer active. It logs metrics to indicate deletion and reset uploading.

#### Example

Assuming you have valid run and experiment IDs, you can call:

```python
manager._set_model_as_deleted("run_123", "exp_456")
```

---

## Documentation for `ExperimentsManager.get_last_model_url`

### Functionality

Retrieves the URL for the best model from the previous iteration. This method calls an internal function to obtain the best previous run ID and then fetches the artifact URL associated with that run. If the run is marked as initial or no valid run is found, the method returns None.

### Parameters

None.

### Usage

Use this method when you need to retrieve the URL of the best model artifact from a previous tuning iteration. This is useful for comparisons or version management.

#### Example

```python
url = experiments_manager.get_last_model_url()
if url:
    print("Best model URL:", url)
else:
    print("No model available.")
```

---

## Documentation for `ExperimentsManager.get_current_model_url`

### Functionality

This method fetches the URL for the best model in the current iteration of experiments. It calls `get_best_current_run_id` to determine the appropriate run and then retrieves the model URL using the artifact path. If the run is initial or no valid run exists, it returns None after logging a warning.

### Parameters

This method does not accept external parameters.

### Returns

- Optional[str]: The URL of the best model or None if not available.

### Usage

- **Purpose**: Retrieve the URL of the best model from the current iteration for evaluation or deployment.

#### Example

Assuming an ExperimentsManager instance named experiment_manager, you can fetch the URL as follows:

```python
model_url = experiment_manager.get_current_model_url()
if model_url:
    print("Current model URL:", model_url)
else:
    print("Model URL not found.")
```

---

## Documentation for `ExperimentsManager.model_is_uploaded`

### Functionality

This method checks if a model is uploaded for the current run. It queries the MLflow run records using a model existence filter. The method returns True if there is a record of an uploaded model, and False otherwise. It is decorated with a retry mechanism to handle transient errors.

### Parameters

This method does not take any parameters.

### Returns

- bool: True if a model is uploaded, False otherwise.

### Usage

- **Purpose**: Verify the existence of an uploaded model for the current run.

#### Example

```python
experiments_manager = ExperimentsManager(...)

if experiments_manager.model_is_uploaded():
    print("Model is available.")
else:
    print("Model not uploaded yet.")
```

---

## Documentation for `ExperimentsManager.delete_model`

### Functionality

Deletes a model for a specified run in an experiment. The method first determines which experiment to use: if `experiment_id` is not provided, it defaults to the current tuning iteration. It then checks if the run exists and has a model uploaded. If no model is found, a warning is logged; otherwise, it attempts to delete the model and marks it as deleted.

### Parameters

- **run_id**: ID of the run containing the model.
- **experiment_id**: Optional; ID of the experiment holding the run. If not provided, the current tuning iteration is used. Note that deletion is not allowed for the initial model experiment.

### Usage

- **Purpose**: To remove a model from an experiment, useful for cleanup and management of experimental runs.

#### Example

```python
manager.delete_model("run123", "exp456")
```

---

## Documentation for `_save_model`

### Functionality

Uploads a model to MLflow. The method uses `mlflow.pytorch.log_model` to log the given model along with its pip requirements, and logs a "model_uploaded" metric. It also prints an info log message when the upload finishes.

### Parameters

- **model**: An instance of EmbeddingsModelInterface representing the model to be uploaded.

### Usage

- **Purpose**: Saves the fine-tuned model by uploading it to MLflow. This is typically called internally by the save_model method.

#### Example

Assuming `model` implements EmbeddingsModelInterface:

```python
experiments_manager._save_model(model)
```

---

## Documentation for `ExperimentsManager.save_model`

### Functionality

Saves a fine-tuned embedding model to MLflow. If `best_only` is True, only saves when the model shows improved quality over the previous best.

### Parameters

- **model**: Instance of EmbeddingsModelInterface representing the model to be saved.
- **best_only**: Boolean flag indicating whether to save only the best performing model (default: True).

### Usage

- **Purpose**: To record a fine-tuned model in an MLflow experiment. The method validates the current tuning iteration and determines if the new model surpasses the current best based on quality.

#### Example

```python
experiments_manager.save_model(model_instance, best_only=True)
```

---

## Documentation for `ExperimentsManager.get_top_params_by_experiment_id`

### Functionality

Retrieves the best hyperparameter settings from previous tuning iterations for a given experiment. It filters finished runs, orders them by a specified metric, and returns a list of FineTuningParams instances.

### Parameters

- **experiment_id** (str): Identifier of the experiment.

### Returns

- Optional[List[FineTuningParams]]: A list of the top N fine-tuning parameter objects. Returns None if the experiment matches the initial iteration or if no finished runs are found.

### Usage

- **Purpose**: Use this method to select promising tuning configurations from previous experiments for further iterations.

#### Example

```python
top_params = exp_manager.get_top_params_by_experiment_id("12345")
```

---

## Documentation for `ExperimentsManager.get_top_params`

### Functionality

Retrieve the top N previously tuned parameter sets from the latest fine-tuning iteration. Internally, the method obtains the previous iteration ID and uses it to fetch the top runs based on the main metric. This selection is guided by whether a lower or higher value of the metric indicates better performance.

### Return Value

Returns a list of FineTuningParams objects representing the best parameter configurations from prior experiments. If no parameters are found, it returns None.

### Usage

Use this method to obtain the top-performing parameter settings to initialize further fine-tuning or evaluation processes.

#### Example

```python
# Retrieve the best tuning parameters
params = experiments_manager.get_top_params()
if params is not None:
    for config in params:
        print(config)
```

---

## Documentation for `ExperimentsManager.save_metric`

### Functionality

This method accumulates metric values and logs each metric using MLflow in the context of fine-tuning experiments. It accepts a MetricValue object, passes it to every registered accumulator, and logs the resulting metrics.

### Parameters

- **metric_value**: A MetricValue instance carrying the metric data to be logged.

### Usage

- **Purpose**: To record performance metrics during experiment runs.

#### Example

Assuming you have an ExperimentsManager instance named exp_manager and a MetricValue instance called mv, you can log a metric by calling:

```python
exp_manager.save_metric(mv)
```

---

## Documentation for `ExperimentsManager.get_quality`

### Functionality

For a tuning experiment run, this method retrieves metrics from MLflow and extracts the value based on the main metric. A ValueError is raised if the run id is missing or if the experiment is the initial experiment. The returned value is a float representing the quality metric.

### Parameters

None.

### Usage

- **Purpose**: Evaluate the run quality by returning the main metric value.

#### Example

```python
quality = experiments_manager.get_quality()
print("Run quality:", quality)
```

---

## Documentation for `ExperimentsManager._get_best_quality`

### Functionality

This method retrieves the best run from a given experiment based on its quality metric. It filters for finished runs and selects the run with the best score (lowest if the metric is a loss, highest otherwise).

### Parameters

- **experiment_id**: ID of the experiment to search for finished runs.

### Usage

- **Purpose**: To determine the best run by evaluating a quality metric from an experiment.
- Returns a tuple where the first element is the run ID (or None if no valid run is found) and the second is the quality value.

#### Example

Suppose you have an experiment with ID "12345":

```python
run_id, quality = manager._get_best_quality("12345")
if run_id is not None:
    print(f"Best quality: {quality} on run {run_id}")
else:
    print("No finished runs found.")
```

---

## Documentation for `ExperimentsManager.get_best_quality`

### Functionality

This method retrieves the best quality run from the current fine-tuning iteration. It returns a tuple containing the run ID and the best metric value. For loss metrics, the best value is computed as the minimum, while for other metrics it is the maximum.

### Parameters

None.

### Usage

- **Purpose**: To obtain the optimal run and its quality from the current fine-tuning experiment.

#### Example

```python
run_id, quality = manager.get_best_quality()
print(f"Best run: {run_id} with quality: {quality}")
```