## Documentation for `TritonClient`

### Functionality

TritonClient is an abstract base class designed to interact with the Triton Inference Server. It provides foundational features for checking model readiness and managing inference operations.

### Motivation and Purpose

The class serves as a bridge between your application and the Triton server. It encapsulates connection details, model information, and retry policies, allowing subclasses to focus on inference logic.

### Inheritance

TritonClient inherits from Python's ABC (Abstract Base Class) in the abc module. This requires any subclass to implement the abstract methods necessary for executing inference operations.

### Key Attributes

- **url**: The Triton server URL for connecting to the inference server.
- **plugin_name**: Name of the plugin associated with the model.
- **embedding_model_id**: Identifier for the deployed model.
- **same_query_and_items**: Flag indicating if query and items models are identical.
- **client**: Instance of the Triton InferenceServerClient managing server communication.
- **retry_config**: Configuration for the retry policy during connection attempts.

### Usage

Subclasses extending TritonClient should implement methods that build upon its connection and readiness check functionalities.

#### Example

Below is a simple example of extending TritonClient for a custom use case:

```python
from embedding_studio.embeddings.inference.triton.client import TritonClient

class MyTritonClient(TritonClient):
    def __init__(self, url, plugin_name, embedding_model_id):
        super().__init__(url, plugin_name, embedding_model_id)
    
    def run_inference(self, data):
        # Implement your inference logic using self.client
        pass
```

---

## Documentation for `TritonClient.is_model_ready`

### Functionality

Checks if all required models on the Triton server are deployed and ready. If the query and items models are different, both are verified separately.

### Parameters

- None

### Usage

- **Purpose**: Ensure models are ready before making inference requests.

#### Example

Assume a concrete implementation of TritonClient:

```python
client = YourTritonClient(...)
  
if client.is_model_ready():
    # Proceed with inference
    result = client.infer(...)
else:
    # Handle the error
```

---

## Documentation for `TritonClient._is_model_ready`

### Functionality

This method checks if a specified model (query or items) is ready on the Triton server. It sends a ModelReady request using the model's name and returns a boolean indicating the model's readiness.

### Parameters

- `is_query`: Boolean that specifies whether to check the query model (True) or the items model (False).

### Usage

- **Purpose**: Verify that a model is ready before performing inference.

#### Example

```python
client = SomeTritonClientSubClass(url, plugin_name, embedding_model_id)
if client._is_model_ready(is_query=True):
    # Proceed with inference operations
```

---

## Documentation for `TritonClient._get_default_retry_config`

### Functionality

Creates a default retry configuration for connection and inference attempts with the Triton Inference Server. It generates a RetryConfig object using default parameters from global settings and adds specific retry parameters for query and items inference operations.

### Parameters

None.

### Usage

- **Purpose**: Generate a RetryConfig object with pre-defined retry settings to control retries for Triton client operations.

#### Example

```python
# Retrieve the default retry configuration
retry_config = TritonClient._get_default_retry_config()

# Use retry_config in client initialization or operations
```

---

## Documentation for `TritonClient._prepare_query`

### Functionality

Prepares input for a query embedding request to the Triton Inference Server. Subclasses must implement this method to convert raw query data into a list of InferInput objects suitable for inference.

### Parameters

- `query`: Raw query data to be embedded. The method should handle various data types (e.g., text, images).

### Usage

- Implement this method in a subclass of TritonClient.
- Return a list of grpcclient.InferInput objects that contain the prepared data for model inference.

#### Example

For text data:

```python
def _prepare_query(self, query: str) -> List[grpcclient.InferInput]:
    encoded_text = self.tokenizer.encode(query, max_length=128, truncation=True)
    text_tensor = np.array([encoded_text], dtype=np.int64)
    infer_input = grpcclient.InferInput("input_ids", text_tensor.shape, "INT64")
    infer_input.set_data_from_numpy(text_tensor)
    return [infer_input]
```

---

## Documentation for `TritonClient._prepare_items`

### Functionality

This abstract method converts raw items data into a list of grpcclient.InferInput objects. It prepares inputs for embedding items on the Triton server. Implementations must convert raw data into a properly formatted list based on the data type.

### Parameters

- `data`: Raw items data to be transformed for inference. The type can vary, so the implementation should handle the conversion accordingly.

### Usage

- **Purpose**: Used to prepare input items for batch embedding. Subclasses must implement this method for specific data types.

#### Example

Example for text items:

```python
def _prepare_items(self, data: List[str]) -> List[grpcclient.InferInput]:
    # Processing each text item for embedding
    batch_tensors = []
    for item in data:
        encoded_text = self.tokenizer.encode(item, max_length=128, truncation=True)
        batch_tensors.append(encoded_text)
    padded_batch = pad_sequences(batch_tensors, maxlen=128, padding='post')
    batch_tensor = np.array(padded_batch, dtype=np.int64)
    infer_input = grpcclient.InferInput('input_ids', batch_tensor.shape, 'INT64')
    infer_input.set_data_from_numpy(batch_tensor)
    return [infer_input]
```

---

## Documentation for `TritonClient.forward_query`

### Functionality

Sends a query to the Triton Inference Server and returns a numpy array containing the computed query embedding. Internally, it prepares the input and dispatches an inference request to the server.

### Parameters

- `query`: Data to be embedded. This parameter should be in a format that can be processed by the client's `_prepare_query` method.

### Returns

- `np.ndarray`: The embedding output from the server.

### Usage

- **Purpose**: Retrieve query embeddings from the Triton model server.

#### Example

```python
client = TritonClient(url, plugin_name, embedding_model_id)
embedding = client.forward_query(query_data)
```

---

## Documentation for `TritonClient.forward_items`

### Functionality

This method sends a list of items to the Triton server and receives the corresponding embedding outputs as a numpy array.

### Parameters

- `items`: List[Any] - List of data items to be embedded by the model.

### Usage

- **Purpose**: Batch embedding multiple items using Triton inference.

#### Example

```python
items = ["hello", "world"]
embeddings = client.forward_items(items)
```

---

## Documentation for `TritonClient._send_query_request`

### Functionality

This helper method sends a query request to the Triton Inference Server using retry logic. It calls the client's infer method and returns the model output as a NumPy array.

### Parameters

- `inputs`: List of prepared grpcclient.InferInput objects representing the query request input.

### Usage

- **Purpose**: Issues a query inference call to the Triton server with retry logic, ensuring resilience against temporary connection errors.

#### Example

```python
inputs = client._prepare_query(query)
embedding = client._send_query_request(inputs)
```

Ensure that the input tensor is properly prepared and the output tensor is named "output" in your model configuration.

---

## Documentation for `TritonClient._send_items_request`

### Functionality

Sends an items request to the Triton inference server with retry logic. Selects either the items model or the query model based on configuration, and returns the model output as a numpy array.

### Parameters

- `inputs`: List[grpcclient.InferInput] - A list of prepared InferInput objects for the inference request.

### Usage

- **Purpose**: Performs an inference request on the items model while handling transient errors using retry logic.

#### Example

Suppose you have prepared your inference inputs, then:

```python
output = client._send_items_request(inputs)
```

where `client` is an instance of TritonClient and `inputs` is a list of InferInput objects.

---

## Documentation for `TritonClientFactory`

### Functionality

Factory for creating TritonClient instances with common configuration parameters. This class standardizes client creation for different embedding models.

### Motivation

This factory simplifies client instantiation by bundling shared parameters such as URL, plugin name, and retry settings. It avoids redundant code and improves consistency.

### Inheritance

TritonClientFactory does not explicitly extend any class but declares an abstract method `get_client`. Subclasses must override this method.

### Usage

- **Purpose**: Provide a uniform way to create TritonClient instances for various embedding models.

#### Example Implementation

```python
class TextTritonClientFactory(TritonClientFactory):
    def __init__(self, url, plugin_name, tokenizer, retry_config=None):
        super().__init__(url, plugin_name, retry_config=retry_config)
        self.tokenizer = tokenizer

    def get_client(self, embedding_model_id, **kwargs):
        return TextTritonClient(
            url=self.url,
            plugin_name=self.plugin_name,
            embedding_model_id=embedding_model_id,
            tokenizer=self.tokenizer,
            retry_config=self.retry_config,
            **kwargs
        )
```

---

## Documentation for `TritonClientFactory.get_client`

### Functionality

This method is a factory function for creating a new instance of a TritonClient subclass. It uses common configuration parameters such as URL, plugin name, and retry policies, and returns a client configured for a specific model version.

### Parameters

- `embedding_model_id` (str): The deployed ID of the model used for inference.
- `**kwargs`: Additional keyword arguments passed to the client constructor.

### Usage

This abstract method should be implemented by subclasses to create a client instance. The overriding method will typically call a specific client class, for example, `TextTritonClient`, with the provided parameters.

#### Example

```python
def get_client(self, embedding_model_id: str, **kwargs):
    return TextTritonClient(
        url=self.url,
        plugin_name=self.plugin_name,
        embedding_model_id=embedding_model_id,
        retry_config=self.retry_config,
        **kwargs
    )
```