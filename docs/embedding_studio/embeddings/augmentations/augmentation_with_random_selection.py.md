# Documentation for `AugmentationWithRandomSelection`

## Functionality
This class provides a base for applying a raw augmentation and selecting a random subset of the generated outputs. It inherits from `AugmentationInterface` to ensure a standard augmentation API.

## Parameters
- `selection_size`: Determines the number or proportion of results to select. If a float, it represents the fraction of augmented objects. If an int, it specifies the exact number of objects.

## Motivation
The class enables flexible augmentation by allowing random selection from a set of generated transformations, useful for randomized testing and data diversification.

## Inheritance
- Inherits from `AugmentationInterface`.

## Usage Example
Subclass `AugmentationWithRandomSelection` and implement the `_raw_transform` method. Use `transform()` to apply the augmentation and retrieve a random subset of results.

---

## Documentation for `AugmentationWithRandomSelection._raw_transform`

### Functionality
This method applies a raw transformation to the input object. It is abstract and must be implemented by subclasses to define custom augmentation logic. The method should return a list of augmented objects.

### Parameters
- `object`: The input data to be augmented. It can be of any type.

### Usage
- **Purpose**: To establish a blueprint for creating augmentations. Implement this method to generate initial augmented objects before any selection mechanism.

#### Example
Below is an example implementation:
```python
def _raw_transform(self, object: Any) -> List[Any]:
    return [f"Example 1: {object}",
            f"Example 2: {object}",
            f"Example 3: {object}"]
```

---

## Documentation for `AugmentationWithRandomSelection._select_augmentations`

### Functionality
Selects a subset of augmented objects based on the `selection_size` parameter provided at initialization. If `selection_size` is 1 or greater than or equal to the total number of augmentations, the full augmented list is returned. For values less than 1, it randomly selects a proportion of augmented objects. For integer values other than 1, it selects the specified number of random samples.

### Parameters
- `transformed` (List[Any]): A list of augmented objects obtained from `_raw_transform`.

### Usage
- **Purpose**: To filter and randomly choose a subset of augmentations after generating all possible augmentations.

#### Example
If `_raw_transform` returns ["a", "b", "c", "d"] and `selection_size` is 0.5, this method will randomly select about 2 augmentations.

---

## Documentation for `AugmentationWithRandomSelection.transform`

### Functionality
Transforms the input object by first applying the raw augmentation via `_raw_transform` and then randomly selecting a subset with `_select_augmentations`.

### Parameters
- `object`: The input object to be augmented. It is passed to `_raw_transform` and can be of any type as defined by the augmentation process.

### Usage
- **Purpose**: Generate augmented variations and select a subset based on a given selection criteria.

#### Example
Assuming a subclass implements `_raw_transform`:

```python
aug = YourAugmentationSubclass(selection_size=0.5)
results = aug.transform('sample text')
```
Note:
- If `selection_size` is less than 1, it represents a fraction of results to select.
- If it is an integer, it specifies the exact number to select.