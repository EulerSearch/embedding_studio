# Documentation for FineTuningParams

## Class Overview
The `FineTuningParams` class stores configuration parameters for the fine-tuning procedure. It validates learning rates, weight decays, and margin values to ensure they are within acceptable ranges.

### Parameters
- `num_fixed_layers`: Number of fixed embedding layers.
- `query_lr`: Learning rate of the query model optimizer.
- `items_lr`: Learning rate of the items model optimizer.
- `query_weight_decay`: Weight decay for the query model optimizer.
- `items_weight_decay`: Weight decay for the items model optimizer.
- `margin`: Margin value used in loss computation.
- `not_irrelevant_only`: Flag to use only relevant input samples.
- `negative_downsampling`: Ratio for negative sample downsampling.
- `min_abs_difference_threshold`: Minimum threshold for absolute difference filtering.
- `max_abs_difference_threshold`: Maximum threshold for absolute difference filtering.
- `examples_order`: List of ExamplesType values defining example order.

### Usage
- **Purpose**: Provides a validated configuration for fine-tuning.
- **Motivation**: Ensures that fine-tuning parameters are correctly specified and within expected ranges, supporting flexible input types.
- **Inheritance**: Inherits from `pydantic.BaseModel`.

### Example
```python
from embedding_studio.experiments.finetuning_params import FineTuningParams

params = FineTuningParams(
    num_fixed_layers=3,
    query_lr=0.001,
    items_lr=0.001,
    query_weight_decay=0.0001,
    items_weight_decay=0.0001,
    margin=1.0,
    not_irrelevant_only=True,
    negative_downsampling=0.5
)
```

## Method Documentation

### `validate_examples_order`
#### Functionality
This method validates and converts the `examples_order` parameter for the `FineTuningParams` model. It accepts input in various formats, such as a comma-separated string or a tuple, and converts the value into a list of `ExamplesType`.

#### Parameters
- `value`: The input to be validated. If provided as a string, it is split by commas and cast to integers. If a tuple, it is converted to a list.

#### Returns
- A list of `ExamplesType` items representing the provided input.

#### Usage
- **Purpose**: Ensure the `examples_order` is correctly formatted for subsequent processing in fine-tuning.

#### Example
```python
val = "1,2,3"
validated_order = FineTuningParams.validate_examples_order(val)
# validated_order becomes a list of ExamplesType enums.
```

### `validate_positive_float`
#### Functionality
This method checks that a given value is a positive float. It is used as a validator for learning rate fields such as `query_lr` and `items_lr` in the `FineTuningParams` class. If the value is not a float or is not greater than zero, a `ValueError` is raised.

#### Parameters
- `value`: A float expected to be positive. If it is not a float or is less than or equal to zero, the validation fails.

#### Usage
Use this validator indirectly by providing learning rate values to the `FineTuningParams` model. The model will automatically call this method to ensure parameters meet the criteria.

#### Example
```python
params = FineTuningParams(
    num_fixed_layers=2,
    query_lr=0.01,
    items_lr=0.01,
    query_weight_decay=0.0,
    items_weight_decay=0.0,
    margin=1.0,
    not_irrelevant_only=True,
    negative_downsampling=0.5
)
```
Attempting to pass a non-positive float will raise an error:
```python
params = FineTuningParams(
    num_fixed_layers=2,
    query_lr=-0.01,  # This will cause a ValueError
    items_lr=0.01,
    query_weight_decay=0.0,
    items_weight_decay=0.0,
    margin=1.0,
    not_irrelevant_only=True,
    negative_downsampling=0.5
)
```

### `validate_non_negative_float`
#### Functionality
Validates that the input value is a non-negative float. It checks that the value is an instance of float and that it is greater than or equal to zero. If the check fails, it raises a ValueError, ensuring only valid values are used in model configuration.

#### Parameters
- `value`: A float value that must be non-negative.

#### Usage
- **Purpose**: To verify that parameters such as weight decay are set to a valid, non-negative float prior to use in training.

#### Example
```python
weight_decay = 0.0
# The validator returns 0.0 if valid, otherwise raises an error.
```

### `validate_non_negative_float_margin`
#### Functionality
Validates that the margin is a non-negative float. It checks whether the input is a float and if it is greater than or equal to zero. If not, it raises a ValueError.

#### Parameters
- `value`: A float representing the margin. Must be non-negative.

#### Usage
- **Purpose**: Ensure the margin value is valid before processing it.

#### Example
```python
try:
    margin = FineTuningParams.validate_non_negative_float_margin(0.75)
except ValueError as err:
    print(err)
```

### `validate_non_negative_int`
#### Functionality
This method validates that a given value is a non-negative integer. It ensures the value is at least 0 and raises a ValueError if the condition is not met.

#### Parameters
- `value`: The integer to validate; must be 0 or positive.

#### Usage
- **Purpose**: Validates integer parameters, such as `num_fixed_layers` in fine-tuning configurations.

#### Example
```python
>>> FineTuningParams.validate_non_negative_int(3)
3

>>> FineTuningParams.validate_non_negative_int(-1)
ValueError: -1 must be a non-negative integer
```

### `id`
#### Functionality
Generates a unique ID for the parameter set by computing a SHA-256 hash of the string representation of the instance.

#### Parameters
This method does not take any parameters.

#### Usage
- **Purpose**: Create a unique hash ID for a `FineTuningParams` object.

#### Example
```python
>>> params = FineTuningParams(
    num_fixed_layers=3,
    query_lr=0.01,
    items_lr=0.01,
    query_weight_decay=0.0,
    items_weight_decay=0.0,
    margin=1.0,
    not_irrelevant_only=True,
    negative_downsampling=0.5
)
>>> print(params.id)
```