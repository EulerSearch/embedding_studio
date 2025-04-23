# Documentation for `MetricValue`

### Functionality
This class holds a metric's name and value. It ensures that the name is a non-empty string and that its value is a float. It provides a way to adjust the name by adding a prefix via the `add_prefix` method.

### Main Purposes
- Encapsulate a metric with its name and numeric value.
- Validate the metric's attributes upon creation.
- Support name modifications for clarity in reporting metrics.

### Motivation
The MetricValue class ensures consistency when handling metrics in the system. By validating inputs, it prevents common errors and maintains data integrity when metrics are aggregated and processed.

### Inheritance
MetricValue does not inherit from any other class; it is a base class used to represent individual metric values.

### Usage
**Purpose** - Create instances of metric values to be used with accumulators, like `MetricsAccumulator`.

#### Example
```python
mv = MetricValue("accuracy", 0.95)
mv.add_prefix("test")
print(mv.name, mv.value)
```

## Documentation for `MetricValue.name`

### Functionality
Returns the metric's name as a string. This property gives access to the name provided when the MetricValue instance was created.

### Parameters
This property does not accept any parameters.

### Usage
- **Purpose**: Retrieve the name of the metric.

#### Example
```python
metric = MetricValue("accuracy", 0.95)
print(metric.name)  # Output: "accuracy"
```

## Documentation for `MetricValue.value`

### Functionality
Provides access to the metric value stored in the instance as a float. It is a read-only property that returns the internal value.

### Parameters
- None.

### Usage
- **Purpose** - Retrieve the metric's value from a MetricValue instance.

#### Example
```python
mv = MetricValue("accuracy", 0.95)
print(mv.value)  # Output: 0.95
```

## Documentation for `MetricValue.add_prefix`

### Functionality
Prepends a given prefix to the metric name. This method updates the internal name by adding the prefix followed by an underscore, which helps in grouping or identifying metrics.

### Parameters
- `prefix`: A string to be prepended to the current metric name, separated by an underscore.

### Usage
- **Purpose**: Modify the metric's name by adding a prefix. This is useful for organizing metrics from different contexts.

#### Example
```python
mv = MetricValue("accuracy", 0.95)
mv.add_prefix("test")
# Now mv.name is "test_accuracy"
```

---

# Documentation for `MetricsAccumulator`

### Functionality
MetricsAccumulator accumulates metric values and computes various aggregations such as mean, min, max, and a sliding mean over a fixed window. It filters metric values by matching names and performs basic error checking on input parameters.

### Parameters
- `name`: Identifies the metric to accumulate.
- `calc_mean`: If True, compute the mean of the metrics.
- `calc_sliding`: If True, compute a sliding mean over a window.
- `calc_min`: If True, compute the minimum value.
- `calc_max`: If True, compute the maximum value.
- `window_size`: Defines the size of the sliding window (integer > 1).

### Inheritance
This class inherits from Python's default object class and serves as a self-contained aggregator for metric values.

### Usage
Initialize with desired parameters and use the accumulate method by passing MetricValue instances. The method returns a list of aggregated results after each accumulation.

#### Example
```python
accumulator = MetricsAccumulator('loss', calc_mean=True, calc_sliding=False, calc_min=False, calc_max=False, window_size=10)
metric = MetricValue('loss', 0.5)
aggregates = accumulator.accumulate(metric)
```

## Documentation for `MetricsAccumulator.name`

### Functionality
This getter returns the internal name of a MetricsAccumulator. It is used to verify if a metric value matches the expected name when accumulating metrics.

### Parameters
None.

### Usage
- Purpose: Retrieve the assigned name of the accumulator.

#### Example
```python
acc = MetricsAccumulator('accuracy')
print(acc.name)  # Output: 'accuracy'
```

## Documentation for `MetricsAccumulator.clear`

### Functionality
This method clears all accumulated metric values in the MetricsAccumulator. It resets the internal list of values.

### Parameters
None.

### Usage
Use this method when you need to remove all stored metric values before starting a new accumulation process.

#### Example
```python
acc = MetricsAccumulator("accuracy", calc_mean=True)
# accumulate metric values...
acc.clear()
```

## Documentation for `MetricsAccumulator.accumulate`

### Functionality
This method adds a metric value to the accumulator if the metric's name matches the accumulator's name. It then computes aggregated results based on stored values. The aggregations can include the last recorded value, mean, sliding mean, minimum, and maximum, depending on configuration.

### Parameters
- `value`: An instance of `MetricValue` that holds a metric name and a float value. The method will accumulate the metric only if the given name equals the accumulator's name.

### Usage
- **Purpose** - To collect a metric value and compute aggregated results from previously accumulated values.

#### Example
```python
accumulator = MetricsAccumulator("accuracy", calc_mean=True, calc_sliding=True, window_size=5)
result = accumulator.accumulate(MetricValue("accuracy", 0.95))
# This will add the metric value of 0.95 and return aggregated values such as last value, mean, and sliding mean.
```

## Documentation for `MetricsAccumulator.aggregate`

### Functionality
Aggregates accumulated metric values and computes the latest value, mean, sliding mean, minimum, and maximum. The computed aggregations depend on the accumulator's configuration flags (calc_mean, calc_sliding, calc_min, calc_max).

### Parameters
None

### Usage
- **Purpose**: Compute aggregated metrics from the accumulated values in the MetricsAccumulator instance.

#### Example
For an accumulator configured with calc_mean and calc_min enabled, calling `aggregate()` returns a list of tuples with the last value, mean, and minimum metrics.