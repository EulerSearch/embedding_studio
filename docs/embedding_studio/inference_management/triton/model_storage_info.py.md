# Documentation for `DeployedModelInfo`

### Functionality
The `DeployedModelInfo` class stores essential details for models deployed in the Triton Inference Server. It captures the plugin name, model type, an optional embedding model identifier, and the version string. This class standardizes model identification and validates model configuration, ensuring that deployment settings follow Python identifier naming rules and simplifying configuration management by leveraging Pydantic's data validation.

### Inheritance
`DeployedModelInfo` inherits from Pydantic's BaseModel, providing automatic validation, serialization, and error handling for its attributes.

### Usage
- **Purpose:** Use this class to configure and verify model deployment information in a consistent manner across the system.

#### Example
Initialize a model info instance:

```python
instance = DeployedModelInfo(
    plugin_name="MyPlugin",
    model_type="query",
    embedding_model_id="emb001",
    version="1"
)
```

---

## Documentation for `DeployedModelInfo.name`

### Functionality
Generates a standardized model name string using model attributes. If `embedding_model_id` is provided, returns the string in the format: `{plugin_name}_{embedding_model_id}_{model_type}`. Otherwise, it returns the string in the format: `{plugin_name}_{model_type}`.

### Parameters
None. This is a computed property that does not require additional parameters.

### Usage
- **Purpose**: Provides a consistent and descriptive name for deployed models.

#### Example
Simple usage example:

```python
d = DeployedModelInfo(
    plugin_name="PluginA",
    embedding_model_id="emb001",
    model_type="query"
)
print(d.name)  # Output: "PluginA_emb001_query"
```

---

## Documentation for `DeployedModelInfo.validate_plugin_name`

### Functionality
Validates that the plugin name follows Python identifier naming rules. It ensures the name starts with a letter or underscore and contains only letters, digits, and underscores. A ValueError is raised if the name is invalid.

### Parameters
- `value`: The plugin name string to be validated.
- `info`: Validation context information (FieldValidationInfo).

### Usage
- **Purpose**: Guarantees that plugin names conform to standard Python identifier rules, ensuring consistency in model identification.

#### Example
```python
from pydantic import BaseModel, FieldValidationInfo

class DeployedModelInfo(BaseModel):
    plugin_name: str

    @field_validator("plugin_name")
    def validate_plugin_name(cls, value: str, info: FieldValidationInfo) -> str:
        # validation logic
        return value
```

---

## Documentation for `DeployedModelInfo.validate_version`

### Functionality
Validates the version string of a deployed model. It ensures the version is either numeric or matches the archived version identifier. If not, it raises a ValueError.

### Parameters
- `value`: The version string to validate.
- `info`: Validation context with Pydantic details.

### Usage
Use this method to guarantee model versions satisfy required rules. Valid versions are digits (e.g., "1", "2") or the archived identifier (e.g., "_archived").

#### Example
A valid call might use version "3" or "_archived". An invalid version triggers a ValueError.

---

# Documentation for `ModelStorageInfo`

### Functionality
The `ModelStorageInfo` class contains details about a model's storage configuration in the Triton Inference Server. It provides properties for retrieving the model name and paths.

### Parameters
- `model_repo`: Base directory for Triton models.
- `embedding_studio_path`: Path for embedding studio imports.
- `deployed_model_info`: Instance of `DeployedModelInfo` with model deployment details.

### Inheritance
Inherits from Pydantic's BaseModel for data validation and serialization.

### Usage
- **Purpose**: To generate and manage file paths and naming conventions for model deployment in Triton. Ensures a consistent and validated storage layout.

#### Example
```python
from embedding_studio.inference_management.triton.model_storage_info import ModelStorageInfo
storage_info = ModelStorageInfo(...)
print(storage_info.model_path)
```

---

## Documentation for `ModelStorageInfo.archived_version_name`

### Functionality
Returns a constant string identifier used to mark archived model versions.

### Parameters
No parameters.

### Usage
- **Purpose** - Obtain the constant that indicates archived model versions.

#### Example
```python
archived_ver = ModelStorageInfo.archived_version_name()
```

---

## Documentation for `ModelStorageInfo.model_name`

### Functionality
This property retrieves the complete model name from the embedded deployed model info. It is used in constructing paths in the Triton Inference Server.

### Parameters
This method does not take any additional parameters; it relies on the information stored in the `ModelStorageInfo` instance.

### Usage
- **Purpose**: To obtain the full model name needed for organizing and accessing model directories.

#### Example
Assume you have a `ModelStorageInfo` instance:

```python
model_info = ModelStorageInfo(...)
full_name = model_info.model_name
```

---

## Documentation for `ModelStorageInfo.model_path`

### Functionality
Returns the absolute path to the model directory in the Triton model repository. It constructs the path by joining the base model repository (directory in model_repo) with the model name derived from the deployed model info.

### Parameters
This property does not accept any parameters.

### Usage
This property is used to retrieve the complete path where a model is stored in the Triton Inference Server. It ensures consistent path creation based on the provided base directory and model name.

#### Example
If `model_repo` is set to "/models" and the deployed model name is "MyPlugin_query", accessing the property will return "/models/MyPlugin_query".

---

## Documentation for `ModelStorageInfo.model_version_path`

### Functionality
Computes the absolute path to the specific version directory of a model. It combines the base model repository path with the model name to form a valid path for model version storage on the Triton Inference Server.

### Parameters
This property does not accept any parameters.

### Usage
Use this property to obtain the full directory path where a specific version of the model is stored. This is useful for loading or managing model versions within the Triton environment.

#### Example
Assuming you have an instance of `ModelStorageInfo` called `model_info`, you can retrieve the version path as follows:

```python
version_dir = model_info.model_version_path
```