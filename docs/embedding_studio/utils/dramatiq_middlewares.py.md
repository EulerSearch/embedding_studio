# Documentation for `ActionsOnStartMiddleware`

## Functionality
`ActionsOnStartMiddleware` executes a list of callable actions after a Dramatiq worker has booted. This middleware is used for initialization tasks such as setting up connections, loading resources, or performing other startup operations.

## Inheritance
Inherits from the base `Middleware` class provided by Dramatiq.

## Motivation
The middleware allows customization of the worker boot process by running a set of predefined actions, simplifying the initialization routine.

## Parameters
- `actions`: A list of callables that take no parameters. Each action is executed after the worker boots.

## Method: `after_worker_boot`

### Functionality
The `after_worker_boot` method executes the registered actions after a Dramatiq worker has booted. The method first calls the parent class's method.

### Parameters
- `broker`: The broker instance to which the worker is connected.
- `worker`: The worker instance that has completed booting.

### Usage
- **Purpose**: It is used by Dramatiq to perform post-boot initialization in a worker environment.

#### Example
Assuming an actions list with a callable that logs a startup message, when a worker boots, the `after_worker_boot` method is invoked, calling the logging function as part of the initialization.

### Example Code
Instantiate the middleware with a list of initialization actions:

```python
actions = [init_database, init_cache]
middleware = ActionsOnStartMiddleware(actions)
```