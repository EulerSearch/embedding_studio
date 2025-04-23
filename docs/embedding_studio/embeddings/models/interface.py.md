## Documentation for `EmbeddingsModelInterface`

### Overview

EmbeddingsModelInterface is an abstract class that standardizes the interface for embedding models used in fine-tuning tasks. It handles both query and item representations, supporting multi-domain scenarios.

### Main Purposes

- Provide a consistent way to access model parameters for query and item components.
- Facilitate fine-tuning by requiring concrete implementations of core model input and parameter methods.
- Offer a base structure that enforces uniform method signatures across various model implementations.

### Motivation

This interface simplifies managing models that deal with two entities: queries and items. By having standard abstract methods, it ensures that any subclass will implement the necessary functionalities for training, inference, and fine-tuning within the PyTorch Lightning framework.

### Inheritance

EmbeddingsModelInterface inherits from `pytorch_lightning.LightningModule`, which integrates it into the PyTorch Lightning ecosystem, making it easier to leverage training, validation, and distributed execution features.

### Method Documentation

#### `get_query_model_params`

**Functionality:** This method provides an iterator over the parameters of the query model. It is used to access the model's parameters for fine-tuning and inference configuration.

**Parameters:** None.

**Usage:**
- **Purpose:** Retrieve model parameters for the query branch.

**Example:**
```python
def get_query_model_params(self) -> Iterator[Parameter]:
    return self.query_model.parameters()
```

---

#### `get_items_model_params`

**Functionality:** This method returns an iterator over the parameters of the items model. It provides access to the parameters that are used for fine-tuning the items component of an embedding model.

**Parameters:** None.

**Usage:**
- **Purpose:** Retrieve items model parameters for optimization or tracing.

**Example:**
```python
def get_items_model_params(self) -> Iterator[Parameter]:
    return self.items_model.parameters()
```

---

#### `is_named_inputs`

**Functionality:** Returns a boolean indicating if the model expects named inputs.

**Parameters:** None. This is a property, not a method.

**Usage:** Use this property to check if the model requires named inputs.

**Example:**
```python
@property
def is_named_inputs(self) -> bool:
    return True  # When inputs are specified as a named dictionary
```

---

#### `get_query_model_inputs`

**Functionality:** Returns a dictionary of input tensors for the query model, typically used for model tracing.

**Parameters:**
- `device`: The device to place tensors on. If None, the model's device is used.

**Usage:**
- **Purpose:** Generate sample inputs for the query model during tracing.

**Example:**
```python
def get_query_model_inputs(self, device=None) -> Dict[str, Tensor]:
    inputs = self.tokenizer("example query", return_tensors="pt")
    device = device if device else self.device
    return {k: v.to(device) for k, v in inputs.items()}
```

---

#### `get_items_model_inputs`

**Functionality:** This method provides example inputs for the items model, mainly used for model tracing.

**Parameters:**
- `device`: (Optional) Device to place the tensors on. If None, the model's device is used.

**Usage:**
- **Purpose:** To generate a dictionary of input tensors for tracing the items model.

**Example:**
```python
def example_usage(model):
    inputs = model.get_items_model_inputs(device="cuda")
    print(inputs)
```

---

#### `get_query_model_inference_manager_class`

**Functionality:** This method returns the Triton model storage manager class for query model inference.

**Parameters:** None.

**Usage:**
- **Purpose:** Specify the Triton model storage manager class used for query model inference.

**Example:**
```python
def get_query_model_inference_manager_class(self) -> Type[TritonModelStorageManager]:
    return JitTraceTritonModelStorageManager
```

---

#### `get_items_model_inference_manager_class`

**Functionality:** Returns the class for managing items model inference in Triton.

**Parameters:** None.

**Usage:**
- **Purpose:** Provide the items model inference manager class for Triton.

**Example:**
```python
def get_items_model_inference_manager_class(self) -> Type[TritonModelStorageManager]:
    return SomeTritonModelStorageManager
```

---

#### `fix_query_model`

**Functionality:** Fixes a specific number of layers in the query model by freezing them.

**Parameters:**
- `num_fixed_layers`: Number of layers to freeze from the bottom of the model.

**Usage:**
- **Purpose:** Freeze layers to prevent weight updates during training.

**Example:**
```python
def fix_query_model(self, num_fixed_layers: int):
    if len(self.query_model.encoder.layers) <= num_fixed_layers:
        raise ValueError(
            f"Number of fixed layers ({num_fixed_layers}) >= "
            f"number of existing layers ({len(self.query_model.encoder.layers)})"
        )
    self.query_model.embeddings.requires_grad = False
    for i in range(num_fixed_layers):
        self.query_model.encoder.layers[i].requires_grad = False
```

---

#### `unfix_query_model`

**Functionality:** This method unfreezes all layers of the query model by enabling gradients.

**Parameters:** None.

**Usage:** Use this method to re-enable gradient computation for all layers of the query model.

**Example:**
```python
def unfix_query_model(self):
    self.query_model.embeddings.requires_grad = True
    for layer in self.query_model.encoder.layers:
        layer.requires_grad = True
```

---

#### `fix_item_model`

**Functionality:** Freeze a given number of layers in the item model by setting their `requires_grad` attribute to False.

**Parameters:**
- `num_fixed_layers`: An integer specifying the number of layers to freeze.

**Usage:**
- **Purpose:** Freeze the initial layers of the item model during fine-tuning.

**Example:**
```python
def fix_item_model(self, num_fixed_layers: int):
    if len(self.items_model.encoder.layers) <= num_fixed_layers:
        raise ValueError(
            f"Number of fixed layers ({num_fixed_layers}) is greater than or "
            f"equal to total layers "
            f"({len(self.items_model.encoder.layers)})")
    self.items_model.embeddings.requires_grad = False
    for i in range(num_fixed_layers):
        self.items_model.encoder.layers[i].requires_grad = False
```

---

#### `unfix_item_model`

**Functionality:** Unfixes all layers of the item model by enabling gradients.

**Parameters:** None.

**Usage:** Enable fine-tuning by unfreezing the item model.

**Example:**
```python
def unfix_item_model(self):
    self.items_model.embeddings.requires_grad = True
    for layer in self.items_model.encoder.layers:
        layer.requires_grad = True
```

---

#### `forward_query`

**Functionality:** Processes a query through the query model to produce an embedding tensor.

**Parameters:**
- `query`: Input query which may be text, features, or another format.

**Returns:** A FloatTensor containing the embedding of the query.

**Usage:** Converts the given query into an embedding vector.

**Example:**
```python
def forward_query(self, query: str) -> FloatTensor:
    if len(query) == 0:
        logger.warning("Provided query is empty")
    tokenized = self.tokenize(query)
    return self.query_model(
        input_ids=tokenized["input_ids"].to(self.device),
        attention_mask=tokenized["attention_mask"].to(self.device)
    )
```

---

#### `forward_items`

**Functionality:** Processes a list of items through the items model to generate embedding tensors.

**Parameters:**
- `items`: List of items, must not be empty.

**Usage:** Compute embeddings in a single forward pass.

**Example:**
```python
def forward_items(self, items: List[str]) -> FloatTensor:
    if len(items) == 0:
        raise ValueError("items list must not be empty")
    tokenized = self.tokenize(items)
    return self.items_model(
        input_ids=tokenized["input_ids"].to(self.device),
        attention_mask=tokenized["attention_mask"].to(self.device)
    )
```