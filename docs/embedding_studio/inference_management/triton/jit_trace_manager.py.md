## Documentation for `JitTraceTritonModelStorageManager`

### Functionality

The `JitTraceTritonModelStorageManager` class serializes a PyTorch model by applying JIT tracing with an example input. It produces a static computational graph optimized for inference and prepares the model for deployment on Triton Inference Server.

### Parameters

- `storage_info`: Config details for model storage and naming.
- `do_dynamic_batching`: Boolean flag that enables dynamic batching.

### Inheritance

Inherits from `TritonModelStorageManager` and reuses directory setup and configuration generation methods.

### Usage

- **Purpose**: To trace models into static graphs for efficient inference, which is ideal for cases without dynamic control flow.

#### Example

Instantiate `JitTraceTritonModelStorageManager` and use its `_save_model` method with the model and sample input to produce a traced model ready for Triton deployment.

---

### Method: `_get_model_artifacts`

#### Functionality

Returns a list of artifact filenames expected in a model directory. For a JIT-traced model, it includes only the traced model file.

#### Parameters

This method does not take any parameters.

#### Return Value

A list of strings representing model artifact filenames. It returns `["model.pt"]`.

#### Usage

Call this method to verify the required model file for Triton. It ensures that only the JIT-traced model file is used.

##### Example

```python
artifacts = manager._get_model_artifacts()
# Expected output: ["model.pt"]
```

---

### Method: `_generate_triton_config_model_info`

#### Functionality

Generates Triton configuration lines for the PyTorch LibTorch backend. This method sets the model name, platform, and maximum batch size using the storage info provided to the manager.

#### Parameters

This method does not require parameters. It retrieves configuration settings from the instance's `_storage_info` attribute.

#### Usage

**Purpose**: To provide a configuration snippet for Triton when serving a JIT-traced PyTorch model.

##### Example

```python
manager = JitTraceTritonModelStorageManager(storage_info, True)
config = manager._generate_triton_config_model_info()
print(config)
```

---

### Method: `_save_model`

#### Functionality

This method saves a PyTorch model using JIT tracing. It converts the model into an optimized serialized form that can be loaded by Triton's LibTorch backend.

#### Parameters

- `model`: The PyTorch `nn.Module` to trace.
- `example_inputs`: A dictionary mapping input names to example tensors.

#### Usage

- **Purpose**: Trace and save a model in a format optimized for inference with Triton.

##### Example

Assuming a properly prepared model and inputs:

```python
manager._save_model(model, {"input": input_tensor})
```