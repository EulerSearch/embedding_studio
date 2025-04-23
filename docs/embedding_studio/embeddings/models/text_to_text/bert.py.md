# Documentation for `BertModelSimplifiedWrapper` and `TextToTextBertModel`

## BertModelSimplifiedWrapper Class

### Functionality
`BertModelSimplifiedWrapper` wraps a Hugging Face BERT model to produce text embeddings. It extracts the pooler output from the underlying model.

### Inheritance
Inherits from `torch.nn.Module`, the base for neural network modules in PyTorch.

### Motivation
Simplifies usage of BERT by exposing only the essential embedding output, hiding underlying complexity for text embedding tasks.

### Documentation for `BertModelSimplifiedWrapper.forward`

#### Functionality
This method computes the pooled output from a BERT model by using its pooler output. It accepts token IDs and attention masks, passes them to the underlying BERT model, and returns the transformed pooler output.

#### Parameters
- `input_ids` (Tensor): Tensor containing token IDs of the input text.
- `attention_mask` (Tensor): Tensor indicating which tokens to attend.

#### Returns
- (Tensor): The pooler output with a linear transformation and tanh activation applied on the first token.

#### Usage
- **Purpose**: Generate text embeddings using the BERT model's pooler output.

#### Example
```python
from transformers import AutoModel
import torch

model = AutoModel.from_pretrained("bert-base-uncased")
wrapper = BertModelSimplifiedWrapper(model)
inputs = {
    "input_ids": torch.tensor([[101, 102]]),
    "attention_mask": torch.tensor([[1, 1]])
}
emb_output = wrapper.forward(**inputs)
```

---

## TextToTextBertModel Class

### Functionality
`TextToTextBertModel` is a wrapper that encapsulates a BERT model and its tokenizer into a unified interface for generating text embeddings. It leverages the HuggingFace transformers AutoModel and AutoTokenizer to compute embeddings from input texts using BERT's pooler output.

### Parameters
- `bert_model`: A string identifier for a pretrained BERT model or an existing AutoModel instance.
- `bert_tokenizer`: The tokenizer for processing texts. If omitted, it is auto-loaded based on the model's configuration.
- `max_length`: Maximum number of tokens to consider during tokenization.

### Usage
- **Purpose**: Provides a simple interface for generating text embeddings for both queries and items, ensuring consistency in usage.
- **Inheritance**: Inherits from `EmbeddingsModelInterface` to conform with the repository's embedding model structure.

#### Example
```python
from transformers import AutoModel

# Initialize the model using a pretrained BERT model
model = TextToTextBertModel(
    AutoModel.from_pretrained('bert-base-uncased')
)

# Generate embeddings using the query model
embeddings = model.get_query_model()(input_ids, attention_mask)
```

### Documentation for `TextToTextBertModel` Methods

#### `get_query_model`

##### Functionality
Returns the model used for query processing. It is the wrapped BERT model shared with item processing. Use this to obtain the query encoder.

##### Parameters
This method takes no parameters.

##### Usage
Call this method to get the model for encoding queries.

##### Example
```python
model_inst = TextToTextBertModel("bert-base-uncased")
query_model = model_inst.get_query_model()
output = query_model(input_ids, attention_mask)
```

---

#### `get_items_model`

##### Functionality
Returns the wrapped BERT model used for processing items. Since the same model is used for both query and items, it simply returns the model component.

##### Parameters
None.

##### Usage
- **Purpose**: Retrieve the model used to process item data in the text-to-text BERT implementation.

##### Example
```python
items_model = text_to_text_bert.get_items_model()
```

---

#### `get_query_model_params`

##### Functionality
Returns an iterator over the parameters of the query model. This iterator can be used to apply optimizers or other processing steps on the model parameters during training or evaluation.

##### Parameters
None.

##### Usage
- **Purpose**: Retrieve all query model parameters for further processing, such as in training loops or custom optimization routines.

##### Example
```python
model = TextToTextBertModel('bert-base-uncased')
params = model.get_query_model_params()
for param in params:
    print(param.shape)
```

---

#### `get_items_model_params`

##### Functionality
Returns an iterator over the parameters of the items model. Since the query and items models are the same, this method returns the same parameters as `get_query_model_params`.

##### Parameters
None.

##### Usage
Use this method to collect model parameters for training or fine-tuning operations where the items model parameters are needed.

##### Example
```python
model = TextToTextBertModel(bert_model, bert_tokenizer)
for param in model.get_items_model_params():
    print(param.shape)
```

---

#### `is_named_inputs`

##### Functionality
Indicates if the model uses named inputs. BERT models require inputs named "input_ids" and "attention_mask" for proper functioning.

##### Parameters
This property does not accept any parameters.

##### Usage
- **Purpose**: Signals that the model expects named inputs for generating text embeddings.

##### Example
```python
model = TextToTextBertModel("bert-base-uncased")
print(model.is_named_inputs)  # Output: True
```

---

#### `get_query_model_inputs`

##### Functionality
This method returns sample inputs for model tracing by tokenizing predefined text using `TEST_INPUT_TEXTS`. It filters the output to keep only 'input_ids' and 'attention_mask', and moves the tensors to the specified device.

##### Parameters
- `device`: Device to place the tensors. If None, the model's device is used.

##### Usage
Call this method to generate example inputs for model tracing. It is useful for debugging or exporting the model.

##### Example
```python
inputs = model.get_query_model_inputs(device=torch.device('cpu'))
print(inputs)
```

---

#### `get_items_model_inputs`

##### Functionality
Retrieves example inputs for the items model. This method returns the same inputs as `get_query_model_inputs`, offering a consistent approach for model tracing and inference.

##### Parameters
- `device`: Optional parameter specifying the target device for tensor placement. If not provided, the model's default device is used.

##### Usage
- **Purpose**: Generate standardized input tensors for the items model to facilitate efficient model tracing and inference.

##### Example
```python
device = torch.device('cuda')
inputs = model.get_items_model_inputs(device)
```

---

#### `get_query_model_inference_manager_class`

##### Functionality
This method returns the class used to manage query model inference in Triton deployments. It supports tracing and serving the model efficiently during inference.

##### Parameters
None.

##### Return
Returns the `JitTraceTritonModelStorageManager` class which handles inference management for the query model.

##### Usage
- **Purpose:** Acquire the proper inference manager class for preparing the query model for Triton inference.

##### Example
```python
model = TextToTextBertModel(bert_model, bert_tokenizer)
manager_class = model.get_query_model_inference_manager_class()
# manager_class is JitTraceTritonModelStorageManager
```

---

#### `get_items_model_inference_manager_class`

##### Functionality
This method returns the class used to manage items model inference for Triton. It provides the same inference manager as used for queries, ensuring consistency in model deployment.

##### Parameters
None.

##### Usage
- **Purpose**: Acquire the inference manager class for the items model in Triton.

##### Example
```python
# Obtain the inference manager class
manager_class = text_to_text_bert_model.get_items_model_inference_manager_class()

# Create an inference manager instance
inference_manager = manager_class(model_instance)
```

---

#### `fix_query_model`

##### Functionality
Freezes the embeddings and a given number of lower encoder layers during fine-tuning. This prevents these layers from updating during training by setting `requires_grad` to `False`.

##### Parameters
- `num_fixed_layers`: Number of layers from the bottom to freeze.

##### Usage
Call `fix_query_model` during training to lock lower model layers.

##### Example
```python
model = TextToTextBertModel("bert-base-uncased")
model.fix_query_model(3)
```

---

#### `unfix_query_model`

##### Functionality
This method enables all layers of the query model for fine-tuning. It resets the frozen state by setting the `requires_grad` attribute to `True` on the model embeddings and every encoder layer. This allows the model to update its weights during training.

##### Parameters
This method does not take any parameters.

##### Usage
- **Purpose**: Use this method when you want to unfreeze all layers of the query model to allow full training.

##### Example
```python
model.unfix_query_model()
```

---

#### `fix_item_model`

##### Functionality
This method freezes a specified number of layers in the item model during fine-tuning. Since query and item models use the same BERT instance, it simply calls `fix_query_model` to freeze the corresponding layers.

##### Parameters
- `num_fixed_layers`: An integer representing the number of layers from the bottom of the model to freeze. This must be less than the total number of layers, otherwise a ValueError is raised.

##### Usage
- **Purpose** - Ensures that a subset of the model layers remain unchanged during fine-tuning, preserving the performance of pretrained layers.

##### Example
```python
model.fix_item_model(3)
```

---

#### `unfix_item_model`

##### Functionality
This method enables gradient updates for all layers of the item model. Since query and items use the same model, it reactivates training by calling the underlying `unfix_query_model` method.

##### Parameters
None.

##### Usage
- **Purpose**: Re-enable training for the item model by unfixing all of its layers.

##### Example
```python
model = TextToTextBertModel('bert-base-uncased')
model.unfix_item_model()
```

---

#### `tokenize`

##### Functionality
Tokenizes a text query or a list of queries into a dictionary of tensors. The method uses the underlying BERT tokenizer with specified maximum length, padding, and truncation settings.

##### Parameters
- `query`: A string or a list of strings representing the input text.

##### Usage
- **Purpose**: Prepares text input for the BERT model by generating token ids and attention masks.

##### Example
Tokenizing a single query:
```python
tokenized = model.tokenize("Example query")
```

Tokenizing multiple queries:
```python
queries = ["First query", "Second query"]
tokenized = model.tokenize(queries)
```

---

#### `forward_query`

##### Functionality
This method tokenizes a provided text query and returns its embedding as a tensor. If the input query is empty, a warning is logged.

##### Parameters
- `query`: A non-empty string representing the text query.

##### Returns
A tensor (FloatTensor or Tensor) containing the query embedding.

##### Usage
- **Purpose**: Convert a text query into an embedding for search or similarity computations.

##### Example
```python
embedding = model.forward_query("Your query text")
```

---

#### `forward_items`

##### Functionality
Processes a list of text items through the BERT model and returns their embedding representations. It tokenizes the input texts and computes embeddings using the forward method of the wrapped model.

##### Parameters
- `items`: List[str] containing text items to encode.

##### Usage
- **Purpose**: Generate embedding vectors for a list of text items.

##### Example
```python
items = ["hello", "world"]
embeddings = model.forward_items(items)
```