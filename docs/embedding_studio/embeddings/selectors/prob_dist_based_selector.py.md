## Documentation for `ProbsDistBasedSelector`

### Functionality

This class implements a selection mechanism where corrected distance values are converted into probabilities using a sigmoid function. It inherits from `DistBasedSelector`, and applies a threshold to decide which items to select.

### Parameters

- `search_index_info`: Configuration and index info.
- `is_similarity`: Flag for similarity based measures.
- `margin`: Margin added to adjust distances.
- `softmin_temperature`: Temperature for softmin operations.
- `scale`: Scaling factor for sigmoid conversion.
- `prob_threshold`: Threshold probability for selection.
- `scale_to_one`: Whether to normalize distances to [0,1].

### Usage

**Purpose**: The class is designed to provide a probabilistic based selection in embedding search scenarios by transforming distance metrics using a sigmoid function for binary decision.

#### Example

```python
selector = ProbsDistBasedSelector(search_index_info, is_similarity=True)
labels = selector._calculate_binary_labels(tensor_values)
```

---

## Documentation for `ProbsDistBasedSelector._calculate_binary_labels`

### Functionality

This method converts corrected distance values into probabilities using the sigmoid function and then applies a threshold to decide binary selection. A value greater than the threshold is marked as 1, while a lower value is marked as 0.

### Parameters

- `corrected_values`: A torch.Tensor of adjusted distance values. The values must be already adjusted by a margin.

### Returns

- A torch.Tensor of binary labels where 1 indicates selected and 0 indicates not selected.

### Usage

This method is used when a probabilistic decision is required to select items. It multiplies the corrected values by a scaling factor, applies the sigmoid function, and compares the result to a given threshold.

#### Example

Assuming `selector` is an instance of ProbsDistBasedSelector and `corrected_tensor` is a torch.Tensor of adjusted distances:

```python
binary_labels = selector._calculate_binary_labels(corrected_tensor)
```