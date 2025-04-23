# Merged Documentation

## Documentation for `current_time`

### Functionality

Returns the current UTC time with timezone. Used for passing it to Pydantic models for proper handling with the freezegun module's `freeze_time` function.

### Parameters

None.

### Usage

**Purpose**: To obtain the current UTC time with timezone info.

#### Example

```python
from embedding_studio.utils.datetime_utils import current_time
print(current_time())
```

---

## Documentation for `unaware_utc_to_aware_utc`

### Functionality

Converts a timezone-unaware datetime object, assumed to be in UTC, into a timezone-aware UTC datetime.

### Parameters

- `stamp`: A datetime object without timezone info, assumed to represent UTC time.

### Usage

- **Purpose**: Convert tz unaware datetime to aware datetime.

#### Example

```python
from datetime import datetime
from embedding_studio.utils.datetime_utils import unaware_utc_to_aware_utc

dt = datetime(2023, 1, 1)
aware_dt = unaware_utc_to_aware_utc(dt)
print(aware_dt)
```

---

## Documentation for `utc_timestamp`

### Functionality

Returns the current UTC time as an integer timestamp. It calls the `current_time` function and uses its timestamp value.

### Parameters

None.

### Usage

- **Purpose**: To obtain the current UTC timestamp in seconds.

#### Example

```python
from embedding_studio.utils import datetime_utils
ts = datetime_utils.utc_timestamp()
print(ts)
```

---

## Documentation for `check_utc_timestamp`

### Functionality

Checks if a given UTC timestamp is within a specified delta range relative to the current UTC timestamp.

### Parameters

- `timestamp`: UTC timestamp in seconds.
- `delta_sec`: Maximum delta allowed (in seconds) relative to the current time.
- `delta_minus_sec`: Maximum allowed delta for past timestamps.
- `delta_plus_sec`: Maximum allowed delta for future timestamps.

### Usage

- **Purpose**: Ensure the provided UTC timestamp falls within a timely range relative to the current time.

#### Example

Simple usage:

```python
check_utc_timestamp(utc_ts, delta_sec=10)
```