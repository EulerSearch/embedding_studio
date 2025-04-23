# Documentation for `DistBasedSelector`

## Overview

DistBasedSelector is an abstract class that makes selection decisions based on pre-calculated distance metrics. It leverages existing distance information and applies custom decision logic defined in its subclasses. This helps separate raw distance handling from selection policy.

### Motivation

The motivation behind DistBasedSelector is to provide a reusable framework for distance-based selection. By using pre-computed distances, it enables flexibility in applying different metrics and normalization techniques, thus promoting a clean and modular design.

### Inheritance

DistBasedSelector extends AbstractSelector and is intended to be subclassed. Subclasses must implement the `_calculate_binary_labels` method to define how binary selection is determined based on processed distance values.

### Parameters

- `search_index_info`: Contains configuration for the search index.
- `is_similarity`: Boolean flag indicating if values represent similarity.
- `margin`: Threshold margin value for positive selection.
- `softmin_temperature`: Temperature parameter for softmin calculations.
- `scale_to_one`: Whether to normalize distance values to a [0, 1] range.

### Usage

Subclasses should implement the `_calculate_binary_labels` method to convert normalized distances into binary selection labels.

#### Example Implementation

```python
class CustomSelector(DistBasedSelector):
    def _calculate_binary_labels(self, corrected_values):
        # Example threshold-based selection
        return corrected_values > 0.5
```

## Method Documentation

### `DistBasedSelector.vectors_are_needed`

#### Functionality

This property indicates whether the selector needs to access the actual embedding vectors. It always returns False because distance-based selectors use pre-calculated distance values only.

#### Return Value

- bool: Always returns False.

#### Usage

Use this property to check if embedding vectors should be loaded. When it returns False, you can bypass vector retrieval and rely solely on distance metrics.

##### Example

```python
selector = DistBasedSelector(search_index_info, 
                              is_similarity=True, 
                              margin=0.2, 
                              softmin_temperature=1.0,
                              scale_to_one=False)
if not selector.vectors_are_needed:
    # Proceed with using precomputed distances
    pass
```

### `DistBasedSelector._calculate_binary_labels`

#### Functionality

This method converts corrected distance values into binary labels. Each element in the input tensor is classified as selected (1) if it meets a specific threshold, or not selected (0) otherwise. It is meant to be customized in subclass implementations.

#### Parameters

- `corrected_values`: A torch.Tensor of distance values that have been adjusted by the selection margin. These values determine the binary outcome based on a decision boundary.

#### Returns

- A torch.Tensor containing binary labels, where 1 indicates a selected item and 0 indicates a non-selected item.

#### Usage

Subclasses of DistBasedSelector should implement this method to define their own selection criteria. Below is an example implementation:

##### Example

```python
# Example implementation in a subclass

def _calculate_binary_labels(self, corrected_values: torch.Tensor) -> torch.Tensor:
    # Apply simple threshold-based selection
    return corrected_values > 0
```

### `DistBasedSelector._convert_values`

#### Functionality

This method extracts raw distance values from a list of objects and normalizes them into a tensor based on the metric type and similarity mode. It adjusts the values by inverting dot products, transforming cosine similarities, or taking the reciprocal for Euclidean distances.

#### Parameters

- `categories`: A list of objects with a `distance` attribute. Each object encapsulates a distance metric.

#### Returns

- A `torch.Tensor` of normalized distance values.

#### Usage

- **Purpose** - Transform raw distance metrics to a normalized tensor for further selection logic.

##### Example

Assuming a list of objects with a `distance` field:

```python
objects = [obj1, obj2, obj3]
tensor = selector._convert_values(objects)
```

Ensure the selector is configured with the correct metric type for accurate normalization.

### `DistBasedSelector.select`

#### Functionality

Selects indices of objects from a list based on their normalized distance values, adjust via a margin, and generate binary labels. Indices corresponding to a "1" label are returned.

#### Parameters

- `categories`: List of objects having a `distance` attribute. The distance is normalized and adjusted with a margin.
- `query_vector`: Optional tensor. Not used in the selection process, provided for interface compatibility.

#### Usage

- **Purpose**: Select indices of objects based on pre-calculated distances.

##### Example

```python
# Example usage:
selector = YourDistBasedSelector(search_index_info, is_similarity=True)
indices = selector.select(categories)
```