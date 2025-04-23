# Documentation for `TextItemsDatasetDictPreprocessor`

## Overview
The `TextItemsDatasetDictPreprocessor` class is designed to preprocess dataset dictionaries by normalizing field names, applying text transformations, and converting dataset splits into text item sets. It maps the raw text data into a unified format that can later be utilized for embeddings or further text processing.

### Constructor Parameters
- `field_normalizer`: An instance of `DatasetFieldsNormalizer` used to normalize field names and dataset structure.
- `transform`: A callable that transforms the text items (defaults to a function that performs no change).

### Inheritance
The class inherits from `ItemsDatasetDictPreprocessor`, extending its base functionality specifically for text-related data processing.

### Functionality
- **Purpose**: To prepare a dataset for text-based embedding processes by standardizing fields and applying optional text transformations.

### Method: `get_id_field_name`
#### Functionality
Returns the identifier field name from the field normalizer by accessing the `id_field_name` attribute and returning its value.

#### Parameters
- `self`: Instance of `TextItemsDatasetDictPreprocessor`.

#### Usage
- **Purpose**: To retrieve the identifier field name used in dataset items.

##### Example
```python
preprocessor = TextItemsDatasetDictPreprocessor(field_normalizer)
field_name = preprocessor.get_id_field_name()
```

### Method: `convert`
#### Functionality
Receives a `DatasetDict` and applies field normalization and text transformations. It iterates over the keys of the dataset, creates `ItemsSet` objects for each key, and applies text transforms using a provided function. The result is a new `DatasetDict` with preprocessed items for further use.

#### Parameters
- `dataset`: A `DatasetDict` to be preprocessed, which holds data for different splits, such as 'train' and 'test'.

#### Return
- A `DatasetDict` where each value is an `ItemsSet` with text transformations applied.

#### Usage
- **Purpose**: To convert raw dataset dictionaries into a normalized form with text transformations, ready for further processing.

##### Example
Suppose you have a dataset called `data_ds`:
```python
preprocessor = TextItemsDatasetDictPreprocessor(
    field_normalizer, transform
)
processed_ds = preprocessor.convert(data_ds)
```

### Example of Instantiation and Usage
```python
from embedding_studio.embeddings.data.preprocessors.text_items_preprocessor import TextItemsDatasetDictPreprocessor
from embedding_studio.embeddings.data.utils.fields_normalizer import DatasetFieldsNormalizer

# Create a normalizer instance
normalizer = DatasetFieldsNormalizer(
    id_field_name='id',
    field_mapping={'name': 'text'}
)

# Instantiate the preprocessor with the normalizer and an optional transform
preprocessor = TextItemsDatasetDictPreprocessor(
    field_normalizer=normalizer,
    transform=lambda x: x.lower()
)

# Process a dataset dict (DatasetDict) using the preprocessor
processed_dataset = preprocessor.convert(dataset)
```