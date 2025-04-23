# Documentation for `MinAggregator`

### Functionality

MinAggregator is designed to return the minimum value among a list of rank scores. It calculates the minimum value from a list or tensor of rank scores, and when the list is empty, it returns a default value provided at initialization. This helps in scenarios where a minimal measure is needed, including support for gradient computation using differentiable operations.

### Parameters

- `if_empty_value` (float): The value returned when the rank list is empty.
- `ranks`: A list of floats or a tensor containing rank scores for the aggregation.

### Usage

- **Purpose**: Aggregates rank values by selecting the lowest score. This functionality is implemented in two methods:
    - `_aggregate`: For standard minimum calculation from lists or tensors.
    - `_aggregate_differentiable`: For minimum calculation that allows gradient propagation during training.

- **Motivation**: To capture the minimal performance among several candidates in ranking tasks.
- **Inheritance**: Inherits from RanksAggregator, aligning with a set of aggregation strategies.

### Example

Using `_aggregate` method:
```python
aggregator = MinAggregator(if_empty_value=0.0)
result = aggregator._aggregate([3.5, 2.1, 4.7])
print(result)  # Outputs: 2.1
```

Using `_aggregate_differentiable` method:
```python
import torch

ranks = torch.tensor([2.0, 5.0, 1.0])
min_value = aggregator._aggregate_differentiable(ranks)
```

### Key Points

- The `_aggregate` method calculates the minimum from a list or tensor of rank scores.
- The `_aggregate_differentiable` method uses the "differentiable_extreme" function to compute the differentiable minimum, allowing for gradient flow during training.