## Documentation for TextToTextBERTTritonClient

### Functionality

This class is a specialized TritonClient designed to handle text-to-text tasks using a BERT model. It leverages tokenizers from the transformers library to convert input text into the format required by the Triton Inference Server.

### Motivation

The main motivation for TextToTextBERTTritonClient is to abstract and simplify the preprocessing of text data for BERT inference. By handling tokenization, padding, and truncation uniformly, it ensures that input data is consistently formatted, reducing errors during inference.

### Inheritance

TextToTextBERTTritonClient is derived from the base TritonClient class. It inherits connection management and server communication capabilities, while adding functionality for text-specific preprocessing.

### Parameters

- **url**: URL of the Triton Inference Server.
- **plugin_name**: Name of the plugin or model used for the inference tasks.
- **embedding_model_id**: Identifier for the deployed model.
- **tokenizer**: A tokenizer (either PreTrainedTokenizer or PreTrainedTokenizerFast) used to process input text.
- **preprocessor**: Optional text preprocessing function to prepare inputs.
- **model_name**: Identifier or friendly name for the model, default value is set for common use cases.
- **retry_config**: Optional retry configuration that defines the retry policy.
- **max_length**: Maximum length used during text tokenization.

### Usage

**Purpose** - Facilitate the conversion of raw text into tokenized input that can be sent to a Triton Inference Server via a BERT model.

#### Example

```python
from transformers import AutoTokenizer
from embedding_studio.embeddings.inference.triton.text_to_text.bert import TextToTextBERTTritonClient

# Initialize tokenizer and client
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

client = TextToTextBERTTritonClient(
    url='localhost:8000',
    plugin_name='bert_inference',
    embedding_model_id='model_123',
    tokenizer=tokenizer
)

# Prepare a query for inference
infer_inputs = client._prepare_query('Hello, world!')
```

---

## Documentation for `TextToTextBERTTritonClient._prepare_query`

### Functionality

Prepares a single text input by tokenizing the provided query and constructing a list of Triton InferInput objects for inference.

### Parameters

- `query`: A string containing the text to be processed.

### Usage

- **Purpose**: Convert a query into a tokenized format and package it as InferInput objects for the Triton server.

#### Example

Suppose you have a tokenizer that returns a dictionary with keys `input_ids` and `attention_mask`. This method creates InferInput instances for these keys to send to a Triton service.

---

## Documentation for `TextToTextBERTTritonClient._prepare_items`

### Functionality

This method prepares a list of text inputs for inference. It applies optional preprocessing, tokenization, and converts the tokenized results into InferInput objects for the Triton server. It supports both raw strings and dictionaries as input.

### Parameters

- `data`: A list containing text entries. Each entry is either a plain string or a dictionary with text data.

### Usage

- **Purpose**: To transform a list of text inputs into a format that the Triton Inference Server can process.

#### Example

Given a list of texts:
```python
texts = ["hello world", "sample input"]
```
Usage:
```python
infer_inputs = client._prepare_items(texts)
```

---

## Documentation for `TextToTextBERTTritonClientFactory`

### Functionality

Creates a Triton client for text-to-text tasks using a BERT-based tokenizer and model configurations. It helps in setting up clients that send text data to the Triton Inference Server.

### Inheritance

Inherits from `TritonClientFactory` to share common client configuration and retry policies.

### Parameters

- **url**: URL of the Triton Inference Server.
- **plugin_name**: Name of the plugin used for inference tasks.
- **preprocessor**: Optional function to preprocess text input.
- **model_name**: Model name for the tokenizer tailored to the task.
- **retry_config**: Optional configuration for retry policies.

### Usage

This factory streamlines the creation of `TextToTextBERTTritonClient` instances by centralizing the common configuration. It downloads and reuses the tokenizer, making it easier to manage different model versions.

#### Example

```python
client = factory.get_client('model_id123')
client.infer(query)
```

---

## Documentation for `TextToTextBERTTritonClientFactory.get_client`

### Functionality

This method creates a new instance of TextToTextBERTTritonClient. It sets up the client with the required parameters to connect to a Triton Inference Server for text-to-text processing using BERT.

### Parameters

- `embedding_model_id`: The deployed model identifier.
- `**kwargs`: Additional keyword arguments for client customization.

### Usage

- **Purpose** - Retrieve a client configured for text tokenization and inference tasks.

#### Example

```python
client = factory.get_client("model_id_123", timeout=30)
```