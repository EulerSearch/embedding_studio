# Documentation for `RanksAggregator`

### Functionality
This abstract base class defines an interface for aggregating subitem ranks into a single representative value. Implementations provide both differentiable and non-differentiable methods for aggregation. When items are divided into subitems, combining their rank values into one score is needed. This class abstracts the aggregation behavior to support various use cases, such as training and evaluation.

### Inheritance
RanksAggregator inherits from Python's ABC, ensuring that any subclass implements the following abstract methods:
- `_aggregate`
- `_aggregate_differentiable`

This design enforces a consistent interface across different aggregation strategies.

## Documentation for `RanksAggregator._aggregate`

### Functionality
Aggregates multiple subitem ranks into a single value. Concrete implementations should define the exact aggregation. This method accepts a list of floats or a torch.Tensor and returns a scalar float representing the aggregated rank.

### Parameters
- `ranks`: list of floats or torch.Tensor
  A sequence of subitem rank values to be aggregated.

### Usage
- **Purpose**: To combine individual subitem ranks into one overall rank.

#### Example
```python
def _aggregate(self, ranks: Union[List[float], torch.Tensor]) -> float:
    if len(ranks) == 0:
        return 0.0
    return sum(ranks) / len(ranks)
```

## Documentation for `RanksAggregator._aggregate_differentiable`

### Functionality
This method aggregates a tensor of subitem ranks into a single differentiable tensor value. It supports gradient computation, which is essential during training in deep learning models.

### Parameters
- `ranks`: A torch.Tensor containing subitem ranks. The tensor must have a compatible shape for aggregation.

### Returns
- A torch.Tensor representing the aggregated rank that supports gradient computation.

### Usage
- **Purpose**: Aggregate subitem ranks in a differentiable manner for model training.

#### Example
Simple usage in a training process:
```python
aggregated_rank = ranks_aggregator._aggregate_differentiable(
    ranks_tensor
)
```