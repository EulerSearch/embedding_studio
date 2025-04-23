# Information
This description file relates to method `dict_transforms`.

## Documentation for `dict_transforms`

### Functionality

The function adds a text column to a dataset. It uses a transform to create a text string from each row. By default, it uses the `get_text_line_from_dict` function.

### Parameters

- `examples`: A Dataset object containing row data.
- `transform`: A callable to derive a text string from a row. The default is `get_text_line_from_dict`.
- `text_values_name`: The name of the text column. Defaults to "text".

### Usage

- **Purpose**: Automatically add a text column to a dataset.

#### Example

Import the function and apply it to a dataset:

```python
from embedding_studio.embeddings.data.transforms.dict.transforms import dict_transforms

# Assume 'dataset' is a Dataset object
updated_dataset = dict_transforms(dataset)
print(updated_dataset["text"])
```