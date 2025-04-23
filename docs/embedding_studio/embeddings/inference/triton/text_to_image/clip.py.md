# Documentation for CLIPModelTritonClient and CLIPModelTritonClientFactory

## CLIPModelTritonClient Class

### Functionality
A specialized Triton client for CLIP models. This class uses a tokenizer and an image preprocessing function to prepare text and image data for inference via a Triton Inference Server. It extends from the base TritonClient to support dual input types.

### Purpose
Facilitate flexible inference for multi-modal data by using text tokenization and image transformation pipelines.

### Inheritance
Inherits from TritonClient, which provides core inference client functionalities.

### Parameters
- `url`: URL of the Triton Inference Server.
- `plugin_name`: Name of the plugin/model for inference tasks.
- `embedding_model_id`: Identifier for the deployed CLIP model.
- `tokenizer`: Tokenizer for processing text queries using transformers.
- `transform`: Optional callable to preprocess image inputs.
- `retry_config`: Optional configuration for retrying failed inferences.

### Example
Creating an instance with necessary components:
```python
clip_client = CLIPModelTritonClient(
    url="http://server",
    plugin_name="clip",
    embedding_model_id="clip_model_123",
    tokenizer=my_tokenizer,
    transform=my_transform_func,
    retry_config=None
)
```

## `CLIPModelTritonClient._prepare_query` Method

### Functionality
Tokenizes a text query for preparing inputs for the Triton server. The method uses the provided tokenizer to process the input string and removes the attention mask. It then converts tokens into Triton InferInput objects for inference.

### Parameters
- `query`: A string containing the text to be tokenized for inference.

### Usage
- **Purpose**: Prepares a text query by tokenizing it and wrapping the tokens in Triton InferInput objects for the server.

#### Example
If the query is "Example text", the method tokenizes the text, removes the attention mask, and creates the corresponding inference inputs for the Triton model.

## `CLIPModelTritonClient._prepare_items` Method

### Functionality
This method processes a batch of image inputs for Triton inference. It accepts a list of images that can be PIL.Image objects or numpy arrays originating from cv2. The method converts OpenCV images from BGR to RGB, applies a custom transformation if provided, and ensures the image format is a numpy array with type float32.

### Parameters
- `data`: A list of images. Each image can be a PIL.Image or a numpy array. It represents individual images to be prepared for inference.

### Usage
- **Purpose**: Convert a list of images into a batch suitable for Triton inference. The images are stacked into a batch and wrapped in an `InferInput` for processing.

#### Example
Assume `client` is an instance of `CLIPModelTritonClient` and `image1` and `image2` are valid image objects:
```python
processed_items = client._prepare_items([image1, image2])
```
This returns a list with one `InferInput` containing the batched image data.

## `CLIPModelTritonClientFactory` Class

### Functionality
This factory class creates instances of `CLIPModelTritonClient`, a specialized TritonClient for handling inference tasks on CLIP models using text-to-image processing. It sets common configurations and customizes the tokenizer and image transformation functions.

### Inheritance
Inherits from `TritonClientFactory`.

### Parameters
- `url`: URL of the Triton Inference Server.
- `plugin_name`: Name of the plugin/model for inference.
- `transform`: Function to preprocess images for the model.
- `model_name`: CLIP model name (default: `clip-ViT-B-32`).
- `tokenizer_name`: Tokenizer to use (default to a specific version).
- `retry_config`: Configuration for retry policies.

### Usage
This factory class simplifies the instantiation of `CLIPModelTritonClient`. It provides a common configuration for different model versions, reducing redundancy in client initialization. Use the `get_client` method to create and configure a new client instance.

#### Example
```python
from embedding_studio.embeddings.inference.triton.text_to_image.clip import CLIPModelTritonClientFactory

# Create a factory instance with the necessary configuration.
factory = CLIPModelTritonClientFactory(
    url="http://triton.server",
    plugin_name="clip_plugin",
    transform=my_transform
)

# Get a specific client using the deployed model ID and extra params.
client = factory.get_client("model_id_123", custom_param=value)

# Use the client for inference.
```