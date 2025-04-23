# Documentation for `differentiable_extreme`

## Functionality
Differentiable approximation of max or min using softmax.

## Parameters
- `x`: Input tensor.
- `beta`: Scaling parameter. A larger value sharpens the softmax, 
  approximating the true max/min more closely.
- `mode`: Either "max" or "min" to specify the approximation target.

## Usage
- **Purpose**: Provides a differentiable method to compute an extreme value
  over a tensor.

### Example

Simple usage:

```python
import torch
from embedding_studio.embeddings.models.utils.differentiable_extreme \
     import differentiable_extreme

x = torch.tensor([1.0, 2.5, 0.5])
result = differentiable_extreme(x, beta=1e5, mode="max")
print(result)
```