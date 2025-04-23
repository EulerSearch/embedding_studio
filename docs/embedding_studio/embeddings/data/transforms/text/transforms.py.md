# Documentation for `text_transforms`

## Functionality

The `text_transforms` function applies a transformation to each text entry in a Dataset column. It updates the Dataset with a new column containing the transformed text, aiding in preprocessing for embedding models.

## Parameters

- `examples`: A Dataset containing the input text.
- `transform`: A callable to process the text; defaults to no change.
- `raw_text_field_name`: The original column name (default "item").
- `text_values_name`: The name for the new transformed text column (default "text").

## Usage

Preprocess text data by applying a transformation before model training.

### Example

```python
from datasets import load_dataset
from embedding_studio.embeddings.data.transforms.text.transforms import text_transforms

data = load_dataset("sample_dataset")
transformed_data = text_transforms(data, transform=str.lower)
```