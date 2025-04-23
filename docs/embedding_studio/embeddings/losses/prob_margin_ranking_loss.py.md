# Documentation for `ProbMarginRankingLoss`

## Functionality

This class computes a probabilistic margin ranking loss for fine-tuning embeddings. It modifies the standard margin ranking loss by using a sigmoid function instead of ReLU, incorporating confidence scores to lessen the impact of noisy training examples.

### Motivation

The design aims to reduce the influence of noisy or misleading pairings by introducing an additional loss term for small ranking differences. This helps maintain the quality of embeddings despite minor discrepancies in training data.

### Inheritance

Inherits from the `RankingLossInterface`, ensuring it meets the standard interface for ranking loss functions within the project.

### Parameters

- `base_margin`: The margin used in the loss calculation.
- `do_fine_small_difference`: Flag to enable an extra loss term for small differences between positive and negative ranks.

### Usage

Instantiate the class and call it with a features object containing positive and negative ranks, their confidences, and target values.

## Method Documentation

### `ProbMarginRankingLoss.set_margin`

#### Functionality

Sets a new value for the base margin used in the ranking loss calculation. This update replaces the current margin with a new one.

#### Parameters

- `margin`: The new margin value to use. Must be a positive number.

#### Usage

- **Purpose** - Update the margin used in loss calculations.

##### Example

```python
loss_fn = ProbMarginRankingLoss()
loss_fn.set_margin(1.5)
```

### `ProbMarginRankingLoss.forward`

#### Functionality

This method computes the loss for embedding fine-tuning.

#### Parameters

- `features`: An instance of `FineTuningFeatures` that must include:
  - `positive_ranks`: Rank values for positive examples.
  - `negative_ranks`: Rank values for negative examples.
  - `positive_confidences`: Confidence scores for positive examples.
  - `negative_confidences`: Confidence scores for negative examples.
  - `target`: Direction for ranking adjustment.

#### Usage

- **Purpose** - To compute the loss for embedding fine-tuning. Use this method to obtain a differentiable loss value powering optimization.

##### Example

```python
loss_fn = ProbMarginRankingLoss(base_margin=1.0)
loss = loss_fn.forward(features)
```