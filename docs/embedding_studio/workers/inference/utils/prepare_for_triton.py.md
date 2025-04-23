# Documentation for `mark_same_query_and_items` and `convert_for_triton`

## `mark_same_query_and_items`

### Functionality
Marks that the query and items models are the same by creating a flag file in the query model directory. This flag indicates that both the query and items models are identical.

### Parameters
- `query_model_path` (str): The path to the query model directory. A file named "same_query" will be created at this path to signal that the models are the same.

### Usage
- **Purpose**: To flag that a single model is used for both queries and items.

#### Example
Simple usage example:
```python
mark_same_query_and_items("/path/to/query/model")
```

---

## `convert_for_triton`

### Functionality
Prepares and deploys a model for use with the Triton Inference Server. It dynamically selects GPU, processes, and saves both query and items models. Additionally, it creates configuration files for Triton.

### Parameters
- `model`: (EmbeddingsModelInterface) The model interface that provides access to query and items models.
- `plugin_name`: The name used for creating directories and files.
- `model_repo`: The file path to the repository where model versions are stored.
- `model_version`: The version number of the model to be saved.
- `embedding_model_id`: A unique identifier for the model.
- `embedding_studio_path`: Optional; default is `/embedding_studio`, the root path for the studio.

### Usage
- **Purpose**: Deploys a traced model with corresponding configuration for Triton Inference Server.

#### Example
```python
convert_for_triton(model, "plugin_example", "/path/to/repo", 1, "model123", "/embedding_studio")
```