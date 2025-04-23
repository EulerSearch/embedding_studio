# Documentation for FineTuningIteration and its Method parse

## Class Overview
FineTuningIteration represents a specific iteration of the fine-tuning process. It stores the plugin name, run identifier, and batch identifier to distinguish different iterations. Built on pydantic's BaseModel, it offers data validation and easy serialization.

### Attributes
- `batch_id`: Session batch identifier for the iteration.
- `run_id`: Run identifier of the starting model.
- `plugin_name`: Name of the tuned embedding.

### Purpose
To encapsulate and validate iteration details in the fine-tuning workflow.

### Example
Create an iteration instance:
```python
iteration = FineTuningIteration(
    batch_id="001",
    run_id="run123",
    plugin_name="my_embedding"
)
```

## Method: parse

### Functionality
The `parse` method takes an experiment name string and returns a FineTuningIteration object. It supports parsing of both initial and regular experiment formats. In the initial case, only the `plugin_name` is set. Otherwise, `run_id` and `batch_id` are populated.

### Parameters
- `experiment_name`: Experiment name string. Expected format:
  - For initial experiments: `"plugin_name / initial / ..."`
  - For regular experiments: `"plugin_name / iteration / run_id / batch_id"`

### Purpose
Parse an experiment name and create a FineTuningIteration instance.

### Example
For an initial experiment:
```python
iteration = FineTuningIteration.parse(
    "my_plugin / initial / sample"
)
```

For a regular experiment:
```python
iteration = FineTuningIteration.parse(
    "my_plugin / iteration / 12345 / 67890"
)
```