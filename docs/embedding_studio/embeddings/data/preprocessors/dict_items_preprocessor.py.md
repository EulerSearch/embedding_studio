## Documentation for `DictItemsDatasetDictPreprocessor`

### Functionality

The `DictItemsDatasetDictPreprocessor` class processes datasets containing dictionary items. It normalizes field names, applies a transformation to extract text lines from dict items, and builds an `ItemsSet` for each dataset split. It inherits from `ItemsDatasetDictPreprocessor`, ensuring consistency in data handling.

### Parameters

- `field_normalizer`: An instance of `DatasetFieldsNormalizer` that standardizes field names and specifies the ID field.
- `transform`: Optional callable that extracts a text line from a dictionary. If not provided, defaults to `get_text_line_from_dict`.

### Usage

- **Purpose**: To convert dictionary-based datasets into a text-based format for embedding applications.

### Example

Below is an example of how to use the preprocessor:

```python
from embedding_studio.embeddings.data.preprocessors.dict_items_preprocessor \
    import DictItemsDatasetDictPreprocessor
from embedding_studio.embeddings.data.utils.fields_normalizer \
    import DatasetFieldsNormalizer

normalizer = DatasetFieldsNormalizer(...)
preprocessor = DictItemsDatasetDictPreprocessor(
    field_normalizer=normalizer,
    transform=None  # uses the default transform
)

processed_dataset = preprocessor.convert(original_dataset)
```

The example shows how to normalize and transform a dataset for embedding operations.

---

## Documentation for `DictItemsDatasetDictPreprocessor.get_id_field_name`

### Functionality

Returns the name of the unique identifier field from the normalized dataset. This method accesses the `id_field_name` attribute of the field normalizer to obtain the ID field.

### Parameters

None.

### Usage

Use this method to retrieve the key that uniquely identifies a dataset item after normalization.

#### Example

If the field normalizer is configured with `id_field_name` set to "id", then calling `get_id_field_name()` returns "id".

---

## Documentation for `DictItemsDatasetDictPreprocessor.convert`

### Functionality

Normalizes dataset fields and applies dict transforms. Returns a `DatasetDict` with each key holding an `ItemsSet` constructed from the normalized dataset.

### Parameters

- `dataset`: The `DatasetDict` to be preprocessed; typically contains train/test splits.

### Returns

- `DatasetDict`: A preprocessed dataset where values are `ItemsSet` objects built from the input data.

### Usage

- **Purpose**: Prepare the dataset for embedding computations by normalizing fields and transforming dictionary items.

#### Example

Example usage:

```python
preprocessor = DictItemsDatasetDictPreprocessor(normalizer, transform)
updated_dataset = preprocessor.convert(dataset)
```