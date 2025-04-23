# Documentation for `soft_indicator`

## Functionality

The `soft_indicator` function computes a differentiable indicator that approximates the behavior of a hard indicator (x < threshold). It utilizes a sigmoid function to smoothly transition between outputs as the input values approach the specified threshold.

## Parameters

- **x**: A `torch.FloatTensor` containing input values.
- **threshold**: A float that defines the threshold, near which the indicator is approximately 0.5.
- **steepness**: An int controlling the sharpness of the transition.

## Usage

- **Purpose**: To generate weights ranging from 0.0 to 1.0 based on a smooth comparison with a threshold.

### Example

For example:
```python
x = torch.tensor([0.005, 0.02])
soft_indicator(x, threshold=0.01, steepness=100)
```
might return a tensor similar to `[1.0, 0.0]`.