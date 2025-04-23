## Documentation for `acquire_lock`

### Functionality

This function obtains an exclusive file lock. It opens the file in write mode and applies a lock using `fcntl`, ensuring exclusive access.

### Parameters

- `lock_file_path`: A string representing the file path used for locking.

### Usage

- **Purpose**: To secure exclusive access to a file resource.

#### Example

```python
import fcntl
lock = acquire_lock("/tmp/mylock.file")
# Perform critical operations
release_lock(lock)
```

---

## Documentation for `release_lock`

### Functionality

This function releases a previously acquired file lock. It unlocks the file using `fcntl` and then closes the corresponding file object, ensuring that resources are freed.

### Parameters

- `lock_file`: The file object that holds the acquired lock. This object is returned by a call to `acquire_lock`.

### Usage

- **Purpose**: To safely remove an exclusive lock from a file and close the file resource after critical operations are completed.

#### Example

```python
lock = acquire_lock("/tmp/mylock.file")
# Perform critical operations
release_lock(lock)
```