# Documentation for VectorsBasedSelector

### Functionality
VectorsBasedSelector selects items based on vector comparisons by calculating distances or similarities between embedding vectors.

### Motivation
When precise embedding values are needed, this selector computes similarities or distances using cosine or dot product metrics. It also supports normalization and softmin operations.

### Inheritance
Inherits from AbstractSelector to leverage the basic selection framework provided by the project.

### Usage
- **Purpose**: Facilitate selection using actual embedding vectors. 
- Supports configurable normalization and differentiable softmin.

#### Example
```python
selector = VectorsBasedSelector(search_index_info,
                                is_similarity=True,
                                margin=0.3,
                                softmin_temperature=0.5,
                                scale_to_one=True)
selected_items = selector.select(query_vector, item_vectors)
```

---

## Documentation for `VectorsBasedSelector.vectors_are_needed`

### Functionality
This method indicates whether the selector requires actual embedding vectors. It returns True since VectorsBasedSelector directly operates on vectors.

### Parameters
This method does not require any parameters.

### Usage
- **Purpose**: Determine if vector data is essential for the selection process.

#### Example
```python
selector = VectorsBasedSelector(search_index_info, ...)
selector.vectors_are_needed
# Returns: True
```

---

## Documentation for `VectorsBasedSelector._calculate_distance`

### Functionality
Calculates similarity or distance between query and item embeddings. Depending on the configured metric (COSINE or DOT), it normalizes vectors, computes inner products, and adjusts outputs using scaling or softmin temperature.

### Parameters
- `query_vector`: Tensor of shape [N1, D] representing query embeddings.
- `item_vectors`: Tensor of shape [N2, M, D] representing item embeddings.
- `softmin_temperature`: (Optional) Temperature for softmin approximation; default value is 1.0.
- `is_similarity`: (Optional) Boolean flag; when True, treats computed values as similarities instead of distances.

### Usage
This method returns a tensor of shape [N1, N2] that contains either similarity scores or distance values depending on the input flag and metric configuration. It supports cosine distance conversion and normalization adjustments.

#### Example
For a cosine metric, query and item vectors are normalized and their dot products computed. If `is_similarity` is False, it returns 1 - cosine similarity as the distance measure.

---

## Documentation for `VectorsBasedSelector._calculate_binary_labels`

### Functionality
Calculates binary selection labels from corrected distance values. This method provides a blueprint for threshold-based decision logic in subclasses. It determines which items to select by comparing adjusted distances with a zero threshold, where values above zero indicate selection.

### Parameters
- `corrected_values`: A tensor containing adjusted distances or similarities after applying a margin. Its shape is typically [N1, N2].

### Usage
- **Purpose**: Override this method in subclasses to define the decision boundary for selection. A common implementation returns a binary tensor by comparing the corrected values against zero.

#### Example
```python
def _calculate_binary_labels(self, corrected_values: torch.Tensor) -> torch.Tensor:
    # Items with a corrected value > 0 are selected
    return corrected_values > 0
``` 

---

## Documentation for `VectorsBasedSelector.select`

### Functionality
Selects and returns indices of objects based on vector comparisons. The method converts objects to tensor form, computes distances or similarities between a query vector and object vectors, applies a margin threshold, and then generates binary selection labels to determine which objects are selected.

### Parameters
- `categories`: A list of objects with distance metrics and vectors.
- `query_vector`: An optional torch.Tensor representing the query embedding. It can be in 1D or 2D format.

### Usage
- **Purpose**: To filter and select object indices based on computed vector distances or similarities.

#### Example
```python
# Assuming 'selector' is an instance of VectorsBasedSelector
selected_indices = selector.select(categories, query_vector)
```