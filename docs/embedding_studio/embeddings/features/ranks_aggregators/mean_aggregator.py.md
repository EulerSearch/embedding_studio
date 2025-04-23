# Documentation for `MeanAggregator`

## Functionality
The MeanAggregator class computes the average of a set of rank values. It returns a fallback value when the input list is empty or when the sum of ranks is zero. This behavior ensures robustness in aggregating rankings, making it suitable for various applications.

## Parameters
- `if_empty_value`: Value returned when the input list is empty.
- `if_zeroes_value`: Value returned when the sum of all ranks is zero.

## Method Descriptions

### `MeanAggregator._aggregate`

#### Functionality
Calculates the arithmetic mean of the provided ranks. If the list is empty, it returns the `if_empty_value`. If the sum of ranks is zero, it returns the `if_zeroes_value`.

#### Parameters
- `ranks`: A list of floats or a torch.Tensor containing rank values.

#### Return
- A float representing the computed mean value.

#### Usage
- **Purpose**: Aggregates a set of rank values by computing their mean. It handles special cases to avoid division by zero and provides defaults for empty inputs.

#### Example
```python
from embedding_studio.embeddings.features.ranks_aggregators.mean_aggregator import MeanAggregator

aggregator = MeanAggregator(if_empty_value=0.0, if_zeroes_value=0.0)
result = aggregator._aggregate([1.0, 2.0, 3.0])
print(result)
```

### `MeanAggregator._aggregate_differentiable`

#### Functionality
This method computes the differentiable mean value from a tensor of ranks. It checks if the input tensor is empty and returns the default value specified by `if_empty_value`. If the sum of ranks is zero, it returns `if_zeroes_value`. Otherwise, it computes the mean along the last dimension while preserving gradient information.

#### Parameters
- `ranks`: A torch.Tensor containing the rank values to aggregate.

#### Return
- A torch.Tensor with the computed mean value.

#### Usage
- **Purpose**: Provides a differentiable aggregation that supports gradient backpropagation in neural network computations.

#### Example
```python
import torch
aggregator = MeanAggregator(if_empty_value=0.0, if_zeroes_value=0.0)
ranks = torch.tensor([1.0, 2.0, 3.0])
result = aggregator._aggregate_differentiable(ranks)
print(result)
```

## Inheritance
Inherits from `RanksAggregator`.