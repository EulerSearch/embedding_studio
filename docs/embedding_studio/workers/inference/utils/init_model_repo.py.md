# Merged Documentation

## Documentation for `init_model_repo_for_plugin`

### Functionality
This function initializes the model repository for a specific plugin. If no initial model is found, it uploads the initial model, sets up vector database collections, and converts the model for Triton usage if needed.

### Parameters
- `model_repo`: A string representing the model repository path or name.
- `plugin_name`: The name of the plugin whose model is being initialized.

### Usage
- **Purpose**: Prepare the plugin's model repository for inference.

#### Example
```python
from embedding_studio.workers.inference.utils.init_model_repo import init_model_repo_for_plugin

model_repo = "path/to/model_repo"
plugin_name = "example_plugin"
init_model_repo_for_plugin(model_repo, plugin_name)
```

---

## Documentation for `OnStartMiddleware`

### Functionality
OnStartMiddleware is a middleware to manage worker boot processes. It ensures that all required plugins have their models initialized in the model repository. It triggers plugin discovery and model setup.

### Inheritance
This class inherits from `dramatiq.Middleware`.

### Motivation
The middleware simplifies the inference workflow by automating plugin discovery and model repository initialization.

### Usage
Configure the middleware to run on worker boot. Example:
```python
middleware = OnStartMiddleware()
middleware.after_worker_boot(broker, worker)
```

---

## Documentation for `OnStartMiddleware.after_worker_boot`

### Functionality
Initializes the model repository for all defined inference plugins. On worker boot, it triggers plugin discovery and model initialization. For each plugin in the inference settings, it acquires a lock, calls initialization, and then releases the lock. It also creates an "initialization_complete.flag" file in the repository if needed.

### Parameters
- `broker`: The broker associated with the worker booting up.
- `worker`: The worker instance that is starting.

### Usage
- **Purpose**: Ensures every inference plugin has its models set up in the model repository before processing jobs.

#### Example
Assuming valid broker and worker objects:
```python
middleware = OnStartMiddleware()
middleware.after_worker_boot(broker, worker)
```