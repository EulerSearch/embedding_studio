## Documentation for `ImageItemsDatasetDictPreprocessor`

### Functionality
The `ImageItemsDatasetDictPreprocessor` class preprocesses image data by normalizing dataset fields, applying image transforms for resizing and padding, and wrapping the transformed images in an ItemsSet. This class inherits from `ItemsDatasetDictPreprocessor` and adapts processing specifically for image items.

### Parameters
- `field_normalizer`: A `DatasetFieldsNormalizer` instance to normalize field names.
- `n_pixels`: An integer representing the side length of images (default: 224).
- `transform`: A callable for image transformation, defaulting to a resize and pad function.

### Usage
- **Purpose**: To preprocess image datasets by standardizing fields and applying necessary transforms. This enables the creation of train/test splits with properly processed image data.

#### Example
```python
preprocessor = ImageItemsDatasetDictPreprocessor(
    field_normalizer=my_normalizer,
    n_pixels=224
)
processed_dataset = preprocessor(dataset)
```

---

## Documentation for `ImageItemsDatasetDictPreprocessor.get_id_field_name`

### Functionality
This method returns the ID field name as defined in the associated `DatasetFieldsNormalizer` instance. It provides a standardized way to access the identifier used for each item in the dataset.

### Parameters
- None.

### Usage
- **Purpose**: Retrieve the standardized identifier field name for dataset items.

#### Example
Assuming you have an instance of the preprocessor:
```python
preprocessor = ImageItemsDatasetDictPreprocessor(field_normalizer)
id_field = preprocessor.get_id_field_name()
```
The variable `id_field` will contain the name of the ID field.

---

## Documentation for `ImageItemsDatasetDictPreprocessor.convert`

### Functionality
This method normalizes dataset fields, applies image transforms, and creates `ItemsSet` objects for each split of the dataset. It prepares the data for embedding generation by converting raw image data into processed pixel arrays.

### Parameters
- `dataset`: `DatasetDict`. The dataset containing train/test splits to be preprocessed.

### Returns
- `DatasetDict` where each value is an `ItemsSet` containing normalized and transformed images.

### Usage
Apply this method to prepare your image dataset. It ensures that fields are normalized and images are resized and padded for further processing.

#### Example
```python
processed_dataset = preprocessor.convert(raw_dataset)
```