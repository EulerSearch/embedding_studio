# Documentation for `pytest_load_initial_conftests`

## Functionality

The `pytest_load_initial_conftests` hook sets the environment variable "ES_UNIT_TESTS" to "1", signaling that unit tests are being run.

## Parameters

- **args**: A list of command line arguments (unused in this hook).
- **early_config**: The early configuration object provided by pytest.
- **parser**: The parser instance for processing command line options.

## Usage

The purpose of this hook is to automatically initialize the test environment prior to loading initial conftest files in pytest.

### Example

No direct call is required; pytest executes this hook during startup.