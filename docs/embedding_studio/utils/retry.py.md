## Documentation for `retry_method`

### Functionality

Decorator to run a class method with a retry mechanism. It reads the retry configuration from the instance's `retry_config` attribute and determines the maximum attempts and wait time based on the configuration.

### Parameters

- `name` (str, optional): Overrides the method name. If not provided, the original function's name is used.

### Behavior

- If `max_attempts` is None or less than or equal to 1, the method is executed once.
- Retries on `RequestException` (including server errors), `Timeout`, `ConnectionError`, or other retryable exceptions defined by the class.
- Logs errors on each failed attempt and waits `wait_time_seconds` between retries.
- Raises `MaxAttemptsReachedException` if all attempts fail.

### Usage

- **Purpose**: Provide robust method execution through automatic retries.

#### Example

```python
class MyClass:
    retry_config = {
        'my_method': RetryConfig(max_attempts=3, wait_time_seconds=2)
    }

    @retry_method()
    def my_method(self):
        # method implementation
        pass
```

## Documentation for `retry_function`

### Functionality

The decorator retries a function execution on failure. It catches exceptions specified by `attempt_exception_types`, logs the error, and retries the function up to `max_attempts` times, waiting `wait_time_seconds` between attempts. An optional `is_retryable_error` function can customize which exceptions are considered retryable.

### Parameters

- `max_attempts`: Maximum number of attempts before failing.
- `wait_time_seconds`: Seconds to wait between attempts.
- `attempt_exception_types`: Tuple of exception types that trigger a retry.
- `is_retryable_error`: Optional function to decide if an error is retryable.

### Usage

- **Purpose**: To automatically retry a function when transient errors occur.

#### Example

```python
@retry_function(
    max_attempts=5,
    wait_time_seconds=1,
    attempt_exception_types=(ConnectionError,),
    is_retryable_error=lambda e: isinstance(e, ConnectionError)
)
def my_function():
    # function implementation
    pass
```

## Documentation for `retry_method.wrapper`

### Functionality

Decorator wrapper that provides retry logic for a class method. When a decorated method fails with retryable exceptions, the wrapper automatically retries the method execution. It uses retry parameters from the instance's `retry_config` and waits a specified duration between attempts until the maximum number is reached.

### Parameters

- `self`: The instance of the class containing `retry_config`.
- `*args`: Positional arguments passed to the original method.
- `**kwargs`: Keyword arguments passed to the original method.

### Behavior

- Extracts retry parameters from `self.retry_config` using the method name.
- Attempts to execute the decorated method and returns its result on success.
- Catches `RequestException`, `Timeout`, `ConnectionError`, and other specified exceptions to determine if a retry is warranted.
- Logs errors and sleeps for `wait_time_seconds` between attempts.
- Raises `MaxAttemptsReachedException` if all retry attempts fail.

### Usage

- **Purpose**: Automatically handles transient failures in class methods without cluttering the business logic with retry loops.

#### Example

```python
@retry_method()
def fetch_data(self, url):
    response = requests.get(url)
    return response.json()
```

This decorator ensures that temporary network issues will trigger retries according to the specified configuration.