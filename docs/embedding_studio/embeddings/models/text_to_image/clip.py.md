## Documentation for `TextToImageCLIPModel`

### Functionality
This class wraps a SentenceTransformer CLIP model to produce embeddings for text queries and image items. It splits the model into separate text and vision components, facilitating text-to-image retrieval in a unified embedding space.

### Motivation
The design of TextToImageCLIPModel is motivated by the need to align textual and visual semantics. Separating the processing of text and images allows for efficient matching between text queries and image items.

### Inheritance
TextToImageCLIPModel inherits from EmbeddingsModelInterface, which standardizes the interface for extracting embeddings.

### Parameters
- `clip_model`: A SentenceTransformer instance representing a CLIP model. It provides access to both text and vision components required by the model.

### Usage
- **Purpose**: To enable text-to-image search by generating shared embeddings for both text queries and image items.

#### Example
```python
from sentence_transformers import SentenceTransformer
from embedding_studio.embeddings.models.text_to_image.clip import TextToImageCLIPModel

# Initialize the model
model = TextToImageCLIPModel(SentenceTransformer('clip-ViT-B-32'))

# Get embeddings for text and image
text_embedding = model.get_query_model()(input_text)
image_embedding = model.get_items_model()(input_image)
```

---

## Documentation for `TextToImageCLIPModel.get_query_model`

### Functionality
Returns the text model that processes queries by encoding input text into a shared embedding space with image items. This enables text queries to be compared with image embeddings.

### Parameters
None.

### Usage
- **Purpose** - Retrieve the component that encodes text queries.

#### Example
```python
from sentence_transformers import SentenceTransformer
embedding_model = TextToImageCLIPModel(SentenceTransformer("clip-ViT-B-32"))
query_model = embedding_model.get_query_model()
```

---

## Documentation for `TextToImageCLIPModel.get_items_model`

### Functionality
Returns the vision model component used for processing image items in the text-to-image search module.

### Parameters
This method does not take any parameters.

### Usage
- **Purpose**: Retrieve the model for embedding image data. It is intended for processing image items.

#### Example
```python
model = TextToImageCLIPModel(clip_model)
vision_model = model.get_items_model()
```

---

## Documentation for `TextToImageCLIPModel.get_query_model_params`

### Functionality
Returns an iterator over the parameters of the text model, which is used to process query inputs.

### Parameters
This method does not require any parameters since it operates on the internal text model.

### Usage
- **Purpose**: Retrieve the parameters of the text model component for training, fine-tuning, or analysis.

#### Example
```python
model = TextToImageCLIPModel(...)
params = model.get_query_model_params()
for param in params:
    print(param.shape)
```

---

## Documentation for `TextToImageCLIPModel.get_items_model_params`

### Functionality
Returns an iterator over the parameters of the vision model used for processing image items. This allows access to model parameters for training, evaluation, and debugging.

### Parameters
This method does not take any external parameters.

- Returns: An iterator over torch.nn.Parameter objects representing the vision model's parameters.

### Usage
- **Purpose** - To retrieve the image processing model parameters for optimization or analysis.

#### Example
```python
model = TextToImageCLIPModel(clip_model)
params = model.get_items_model_params()
for p in params:
    print(p.shape)
```

---

## Documentation for `TextToImageCLIPModel.is_named_inputs`

### Functionality
Indicates whether the model uses a named inputs scheme. For CLIP models, the text and vision modules expect different input formats, so named inputs are not employed. This property always returns False.

### Parameters
None.

### Usage
Use this property to determine the input scheme of the model. In CLIP, it confirms that a uniform named input structure is not used.

#### Example
```python
model = TextToImageCLIPModel(...)
print(model.is_named_inputs)
```

---

## Documentation for `TextToImageCLIPModel.get_query_model_inputs`

### Functionality
This method creates example input for tracing the text model. It tokenizes a sample text and returns a dictionary with the key "input_ids", which holds a tensor of tokenized text data.

### Parameters
- `device`: Optional. Specifies the device to place the tensors on. If not provided, the model's default device is used.

### Usage
Use this method to obtain fixed example inputs needed during model tracing or when exporting the model.

#### Example
```python
model = TextToImageCLIPModel(clip_model)
inputs = model.get_query_model_inputs(device=torch.device('cpu'))
print(inputs['input_ids'])
```

---

## Documentation for `TextToImageCLIPModel.get_items_model_inputs`

### Functionality
This method provides example inputs for the vision model, typically used for model tracing. It prepares a sample image by resizing, normalizing, and converting it to a tensor. If no image is provided, a default image from the package is loaded and processed.

### Parameters
- `image`: Optional PIL Image to be used as input. If None, a default image is loaded from the package.
- `device`: Optional device on which to place the tensor. If not provided, the model's device is used.

### Usage
- **Purpose**: Prepare input for the vision model, useful during model tracing or inference preparation.

#### Example
```python
from sentence_transformers import SentenceTransformer

# Initialize the CLIP model
clip_model = SentenceTransformer('clip-ViT-B-32')

# Create the TextToImageCLIPModel instance
model = TextToImageCLIPModel(clip_model)

# Get example inputs for the vision model
inputs = model.get_items_model_inputs()
```

---

## Documentation for `TextToImageCLIPModel.get_query_model_inference_manager_class`

### Functionality
This method returns the Triton model storage manager class used for managing inference of the text (query) model. The returned class handles model tracing with JIT for deployment with Triton.

### Parameters
None.

### Usage
- **Purpose** - To obtain the inference manager class for the text model.

#### Example
```python
model = TextToImageCLIPModel(clip_model)
manager_class = model.get_query_model_inference_manager_class()
manager = manager_class(model.get_query_model())
```

---

## Documentation for `TextToImageCLIPModel.get_items_model_inference_manager_class`

### Functionality
Returns the Triton model inference manager class for handling vision model inference. It uses the JitTraceTritonModelStorageManager to manage the storage and inference configuration.

### Parameters
None.

### Usage
- **Purpose**: Manage vision model inference within Triton.

#### Example
```python
clip_model = SentenceTransformer("clip-ViT-B-32")
model = TextToImageCLIPModel(clip_model)
manager_cls = model.get_items_model_inference_manager_class()
manager = manager_cls(model.get_items_model(), ...)
```

---

## Documentation for `TextToImageCLIPModel.fix_query_model`

### Functionality
This method freezes the embeddings and a specified number of encoder layers in the text model during fine-tuning. Freezing is achieved by setting the `requires_grad` flag to False, which prevents parameter updates during training.

### Parameters
- `num_fixed_layers`: Number of layers to freeze from the bottom of the text model.

### Usage
- **Purpose**: Freeze layers in the query model to control fine-tuning granularity.

#### Example
For instance, if the text model has 12 layers, use:
```python
model.fix_query_model(4)
```
to freeze the first 4 layers during training.

---

## Documentation for `TextToImageCLIPModel.unfix_query_model`

### Functionality
Unfreezes all layers of the text model by setting the `requires_grad` attribute to True for both the embeddings and all encoder layers. This enables gradient updates during fine-tuning after previously freezing layers.

### Parameters
None.

### Usage
- **Purpose**: Allows the text model to learn by re-enabling gradient computation after being fixed.

#### Example
```python
model = TextToImageCLIPModel(clip_model)
model.unfix_query_model()
```

---

## Documentation for `TextToImageCLIPModel.fix_item_model`

### Functionality
Freeze the lower layers of the vision model to prevent updates during training. This is done by setting the requires_grad attribute of the embeddings and the specified number of encoder layers to False.

### Parameters
- `num_fixed_layers`: The number of layers to freeze from the bottom of the vision model. If this number is greater than or equal to the total number of layers, a ValueError is raised.

### Usage
- **Purpose**: Use this method during fine-tuning to keep selected layers fixed while allowing the remaining layers to learn.

#### Example
Assume `model` is an instance of TextToImageCLIPModel:
```python
model.fix_item_model(3)
```

---

## Documentation for `TextToImageCLIPModel.unfix_item_model`

### Functionality
This method enables gradient updates for all layers in the vision model. It sets the requires_grad attribute of the embeddings and encoder layers to True, allowing them to be updated during training.

### Parameters
This method does not accept any parameters.

### Usage
- **Purpose**: To unfreeze all layers of the vision model for further training or fine-tuning.

#### Example
Assuming you have an instance of the model named `clip_model`, simply call:
```python
clip_model.unfix_item_model()
```

---

## Documentation for `TextToImageCLIPModel.tokenize`

### Functionality
Tokenizes a text query for processing by the text model. This method converts an input string into a dictionary of tensors, applying padding, truncation, and setting the maximum length as defined by the underlying tokenizer.

### Parameters
- `query`: A string containing the text query to tokenize.

### Usage
- **Purpose**: Convert a text query into a tokenized format that can be fed into the text model during inference.

#### Example
```python
tokens = model.tokenize("example query")
print(tokens)
```

---

## Documentation for `TextToImageCLIPModel.forward_query`

### Functionality
This method processes a text query using the text model. It tokenizes the query with the model's tokenizer and then obtains the query embedding by passing tokens through the text model. A warning is logged when an empty query is provided.

### Parameters
- `query`: A string representing the text query to encode.

### Usage
- **Purpose** - To generate an embedding tensor from a text query for retrieval or matching tasks in text-to-image search.

#### Example
```python
query = "A breathtaking landscape during sunrise"
embedding = model.forward_query(query)
```

---

## Documentation for `TextToImageCLIPModel.forward_items`

### Functionality
This method processes a list of image tensors through the vision model. It returns an embedding tensor that represents the images.

### Parameters
- `items`: List of image tensors to encode.

### Usage
- **Purpose**: Encode a batch of images into embedding tensors for further processing.

#### Example
```python
embeddings = model.forward_items([img1, img2])
```