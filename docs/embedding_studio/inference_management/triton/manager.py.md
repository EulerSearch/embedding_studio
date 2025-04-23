## Documentation for `TritonModelStorageManager`

### Functionality

The TritonModelStorageManager class serves as an abstract base class for managing the storage and deployment of models in Triton Inference Server. It provides a framework for preparing, saving, and configuring models by managing directory structure, configuration files, and model artifacts.

### Parameters

- `storage_info`: Contains information about the model storage path and naming specifications.
- `do_dynamic_batching`: Enables dynamic batching support if set to True.

### Usage

- **Purpose** - Provides reusable methods and an interface for model storage management, ensuring that model deployment is standardized.

#### Example

Subclass the TritonModelStorageManager and implement the abstract methods, such as `_get_model_artifacts` and `_generate_triton_config_model_info`, to define model-specific logic.

---

## Documentation for `TritonModelStorageManager._get_model_artifacts`

### Functionality

Returns a list of artifact filenames required for model deployment in Triton Inference Server. Subclasses must implement this method to specify all necessary files for a complete model setup.

### Parameters

- None

### Usage

- **Purpose** - Declare the filenames that should be present in the model version directory for successful deployment.

#### Example

```python
def _get_model_artifacts(self) -> List[str]:
    return ["model.pt", "config.json"]
```

---

## Documentation for `TritonModelStorageManager.is_model_deployed`

### Functionality

This method checks if the model is deployed by verifying the existence of all required artifact files in the model version path. It returns True if every artifact is found, and False otherwise.

### Parameters

No parameters are required for this method.

### Returns

- bool: True if all required artifacts exist, False otherwise.

### Usage

- **Purpose**: Confirm that a model has been fully deployed by ensuring all necessary artifacts are present.

#### Example

```python
manager = MyTritonModelStorageManager(storage_info, True)
if manager.is_model_deployed():
    print("Model is deployed")
else:
    print("Model deployment is incomplete")
```

---

## Documentation for `TritonModelStorageManager._setup_folder_directory`

### Functionality

This method creates the required directory structure for the model. It ensures that the path specified in the storage information exists before model deployment.

### Parameters

- None

### Usage

- **Purpose** - Used during model deployment to prepare the necessary folder where model artifacts are stored.

#### Example

```python
manager = TritonModelStorageManager(storage_info, do_dynamic_batching=True)
manager._setup_folder_directory()
```

---

## Documentation for `TritonModelStorageManager._generate_triton_config_model_info`

### Functionality

Generates the model information section of a Triton configuration. Subclasses must override this abstract method to define model name, platform, and maximum batch size for the configuration.

### Parameters

None.

### Usage

Override this method to output configuration lines for the Triton model. Each line corresponds to a setting required by the Triton server.

#### Example

```python
 def _generate_triton_config_model_info(self) -> List[str]:
     return [
         'name: "{}"'.format(self._storage_info.model_name),
         'platform: "pytorch_libtorch"',
         "max_batch_size: 16",
     ]
```

---

## Documentation for `TritonModelStorageManager._generate_triton_config_model_input`

### Functionality

Generates the model inputs section for a Triton config. It takes a dictionary mapping input names to PyTorch tensors and returns a list of formatted configuration lines that describe each input's name, data type, and shape.

### Parameters

- `example_inputs`: A dictionary where keys are input names (str) and values are example PyTorch tensors. The tensor's dtype is converted to a Triton data type, and its shape (except batch dim) specifies dimensions.

### Usage

- **Purpose**: Prepares the input configuration for a model to be deployed on Triton Inference Server.

#### Example

Given a dictionary:

```python
example_inputs = {
    "input": torch.randn(1, 3, 224, 224)
}
```

Calling `_generate_triton_config_model_input(example_inputs)` may return:

```python
[
    "input [",
    "  {",
    "    name: \"input\"",
    "    data_type: TYPE_CODE",
    "    dims: [3, 224, 224]",
    "  }",
    "]"
]
```

---

## Documentation for `TritonModelStorageManager._generate_triton_config_model_output`

### Functionality

Generates the output configuration section for Triton inference by running a PyTorch model on provided example inputs, capturing output shape and type.

### Parameters

- `model`: The PyTorch model whose outputs are used to determine the configuration.
- `example_inputs`: Dictionary mapping input names to example torch.Tensor.
- `named_inputs`: Boolean flag deciding if inputs must be passed as named arguments to the model. Defaults to False.

### Usage

Designed to obtain output configuration entries that reflect the model outputs. It expects a PyTorch model to be executed on example inputs and generates a list of Triton configuration lines based on the output properties such as dtype and shape.

#### Example

Assuming a model outputting a tuple of tensors, the configuration might include multiple output entries:

```python
config_lines = [
    "output [",
    "  {",
    '    name: "output0"',
    "    data_type: TYPE",
    "    dims: [D1, D2]",
    "  },",
    "  {",
    '    name: "output1"',
    "    data_type: TYPE",
    "    dims: [D1, D2]",
    "  }",
    "]"
]
```

---

## Documentation for `TritonModelStorageManager._generate_triton_config_inference_mode`

### Functionality

Generates the instance group section for Triton inference mode. This function sets the execution device based on GPU availability and returns configuration lines defining the instance group.

### Parameters

None.

### Usage

**Purpose** - Specifies instance group details for deploying a model with Triton. It sets the instance count, device kind (GPU or CPU), and lists GPU indices if available.

#### Example

```python
manager = TritonModelStorageManager(storage_info)
config = manager._generate_triton_config_inference_mode()
print("\n".join(config))
```

---

## Documentation for `TritonModelStorageManager._generate_triton_config_dynamic_batching`

### Functionality

This method generates the dynamic batching section for the Triton configuration. When dynamic batching is enabled (i.e., when `do_dynamic_batching` is True), it returns a list of configuration lines. Otherwise, it returns an empty list.

### Returns

- List[str]: Configuration lines for dynamic batching if enabled, or an empty list if disabled.

### Usage

Use this method to insert dynamic batching settings into a full Triton configuration file.

#### Example

```python
config_lines = manager._generate_triton_config_dynamic_batching()
for line in config_lines:
    print(line)
```

---

## Documentation for `TritonModelStorageManager._generate_triton_config_model_versions`

### Functionality

Generates the model version policy section of the Triton configuration. Returns configuration lines that specify the version policy stating that the two most recent model versions are active.

### Parameters

None

### Usage

- **Purpose**: Supplies the version policy for a Triton model. This ensures that the two latest versions are utilized.

#### Example

```python
manager = TritonModelStorageManager(storage_info)
config_lines = manager._generate_triton_config_model_versions()
for line in config_lines:
    print(line)
```

---

## Documentation for `TritonModelStorageManager._generate_extra`

### Functionality

This method generates additional Triton config parameters. It returns a list of configuration lines that may toggle specific features, such as disabling the JIT executor.

### Parameters

None.

### Usage

**Purpose** - To include extra configuration settings in the Triton config file for fine tuning server behavior.

#### Example

A sample configuration appended may be:

```plaintext
parameters: {
  key: "ENABLE_JIT_EXECUTOR"
  value: {
    string_value: "false"
  }
}
```

---

## Documentation for `_generate_triton_config`

### Functionality

Generates the complete Triton configuration string by assembling model info, inputs, outputs, inference mode, dynamic batching, model versions, and extra parameters.

### Parameters

- `model`: The PyTorch model to configure.
- `example_inputs`: A dictionary mapping input names to example tensors.
- `named_inputs`: Boolean flag; if True, inputs are passed as named arguments.

### Usage

Combines various configuration sections into a single string. The result is a configuration for the Triton Inference Server.

#### Example

```python
config = manager._generate_triton_config(
    model, example_inputs, named_inputs=False
)
```

---

## Documentation for `TritonModelStorageManager._setup_triton_config`

### Functionality

Creates and writes the complete Triton configuration file for a model. It generates the config by combining various configuration sections and writes the output to a 'config.pbtxt' file in the model path.

### Parameters

- `model`: The PyTorch model to be configured.
- `example_inputs`: A dictionary mapping input names to example tensors.
- `named_inputs`: (Optional) Boolean flag indicating whether to pass inputs as named arguments.

### Usage

**Purpose** - Prepares the configuration file required by the Triton Inference Server to deploy the model by merging multiple config parts.

#### Example

```python
manager._setup_triton_config(model, example_inputs, named_inputs=True)
```

---

## Documentation for `TritonModelStorageManager._save_model`

### Functionality

Saves a PyTorch model in a format compatible with Triton Inference Server. This abstract method must be implemented by subclasses to handle the specific serialization requirements for different model types.

### Parameters

- `model`: The PyTorch model to save.
- `example_inputs`: A dictionary mapping input names to example tensors.

### Usage

- **Purpose**: Define model-specific serialization logic for Triton.

#### Example

```python
def _save_model(self, model: nn.Module, example_inputs: Dict[str, torch.Tensor]):
    model_path = os.path.join(self._storage_info.model_version_path, "model.pt")
    torch.save(model.state_dict(), model_path)
```

---

## Documentation for `TritonModelStorageManager.save_model`

### Functionality

Sets up the model deployment folder and writes the Triton configuration file. If the model is not yet deployed, it creates necessary directories and saves the model using a subclass implementation for serialization.

### Parameters

- `model`: The PyTorch model to deploy.
- `example_inputs`: A dictionary mapping input names to example tensors.
- `named_inputs`: Boolean flag to enable passing inputs as named arguments.

### Usage

- **Purpose** - Deploy a model to Triton Inference Server by creating the model directory and configuration file.

#### Example

```python
manager = MyTritonModelStorageManager(storage_info)
manager.save_model(model, example_inputs, named_inputs=True)
```