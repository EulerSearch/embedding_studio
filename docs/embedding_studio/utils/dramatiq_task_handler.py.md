# Documentation for `update_task_with_retry` and `create_and_send_task`

## `update_task_with_retry`

### Functionality

Attempts to update a task in the database by associating it with a broker ID from the message queue. If a MongoDB error occurs, the error is logged and the exception is re-raised to be handled by the caller.

### Parameters

- **task**: The task object to update.
- **broker_id**: The broker ID to associate with the task.
- **task_manager**: CRUD manager for performing the update operation on the task.

### Usage

- **Purpose**: Update a task with a broker ID and handle errors.

#### Example

```python
update_task_with_retry(task, "broker-123", task_manager)
```

---

## `create_and_send_task`

### Functionality

Creates a Dramatiq message for a task, updates the task with the corresponding broker ID, and sends the message to the message queue. If any step fails, the function returns None.

### Parameters

- **worker**: Callable that is a Dramatiq actor. It is used to create and send the message.
- **task**: The task object to be sent to the message queue and updated.
- **task_crud**: Instance of CRUDBase for handling task database operations.

### Usage

- **Purpose**: To initialize and send a task message reliably with error logging.

#### Example

```python
updated_task = create_and_send_task(actor, task, task_crud)
if updated_task is None:
    # Handle error case
```