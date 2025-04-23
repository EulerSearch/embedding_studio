## Documentation for `ClickstreamDao`

### Functionality
`ClickstreamDao` defines an abstract interface for managing clickstream sessions and events. It specifies methods to register, update, push events, mark sessions as irrelevant, and retrieve session details and batches.

### Motivation
This abstract class standardizes the way clickstream data is accessed and manipulated. By enforcing a consistent API, it allows different implementations to handle session data while maintaining uniformity across the project.

### Inheritance
`ClickstreamDao` inherits from the Python ABC class, making it an abstract base class. All methods are abstract, meaning that concrete subclasses must provide their own implementations.

### Usage
Implement the abstract methods in a subclass to create a working clickstream data access object. Example usage involves registering sessions, updating session details, and handling batch processes.

### Methods

#### `register_session`
- **Functionality**: Registers a new click stream session. If a session with the specified id already exists, the session remains unchanged.
- **Parameters**: 
  - `session`: A new session object that includes all required details.
- **Returns**: A registered session object, enhanced with a batch id and session number.
- **Purpose**: Registers a session and ensures unique recording in the system.
- **Example**:
  ```python
  registered = clickstream_dao.register_session(new_session)
  ```

#### `update_session`
- **Functionality**: Updates an existing click stream session. This method takes a session object with updated details and refreshes its data, including batch id and session number.
- **Parameters**: 
  - `session`: A session object containing the new data for update.
- **Purpose**: To update session data within the click stream system.
- **Example**:
  ```python
  updated_session = dao.update_session(my_session)
  ```

#### `push_events`
- **Functionality**: The `push_events` method is used to add new session events safely to the system. It ensures that events are not duplicated if they already exist.
- **Parameters**: 
  - `events`: A list of session events to be pushed.
- **Purpose**: To reliably add session events without modifying existing events.
- **Example**:
  ```python
  dao.push_events([event1, event2, # additional session events])
  ```

#### `mark_session_irrelevant`
- **Functionality**: Marks a click stream session as irrelevant in the system. This method receives a session id and returns an updated session record that is flagged as irrelevant.
- **Parameters**: 
  - `session_id`: Identifier of the session to mark as irrelevant.
- **Purpose**: Invoke this method when a session should be excluded from regular data processing because it is not considered relevant.
- **Example**:
  ```python
  updated_session = dao.mark_session_irrelevant("12345")
  ```

#### `get_session`
- **Functionality**: Retrieves a registered click stream session along with its associated events. Returns the session with events if found, or None when no matching session exists.
- **Parameters**: 
  - `session_id`: Unique identifier for the session to retrieve.
- **Purpose**: Fetch click stream session data along with its related events for further processing or analysis.
- **Example**:
  ```python
  session = dao.get_session("session123")
  ```

#### `get_batch_sessions`
- **Functionality**: Retrieves a list of sessions that belong to a specified batch. It allows filtering by session number, and supports limits on the number of sessions as well as the number of events per session.
- **Parameters**: 
  - `batch_id`: Identifier of the session batch.
  - `after_number`: Skip sessions with session numbers less than or equal to this value.
  - `limit`: Maximum number of sessions to return.
  - `events_limit`: Maximum number of events to include for each session.
- **Purpose**: Retrieve sessions with events for a given batch. Optionally filter sessions by number and limit the results.
- **Example**:
  ```python
  dao.get_batch_sessions(batch_id="batch123", after_number=50, limit=10, events_limit=5)
  ```

#### `get_batch`
- **Functionality**: This method retrieves a session batch identified by the provided `batch_id` from the clickstream data. It returns the corresponding `SessionBatch` object if found, or `None` if no match exists.
- **Parameters**: 
  - `batch_id`: Unique string identifier for the session batch to retrieve.
- **Purpose**: To fetch the batch of registered sessions associated with a given batch identifier.
- **Example**:
  ```python
  batch = clickstream_dao.get_batch("batch123")
  ```

#### `release_batch`
- **Functionality**: Releases the current collecting batch using a unique operation idempotency key. Returns the released batch if it exists; otherwise, returns None.
- **Parameters**: 
  - `release_id`: Unique release id used as an idempotency key.
- **Purpose**: To safely release a collecting batch without risk of duplicate processing.
- **Example**:
  ```python
  batch = ClickstreamDao.release_batch("unique_release_id")
  ```

#### `update_batch_status`
- **Functionality**: This method updates the status of a session batch using its batch id. It accepts a new status and returns the updated session batch, or None if the specified batch is not found.
- **Parameters**: 
  - `batch_id`: A string representing the unique identifier of the batch.
  - `status`: An instance of `SessionBatchStatus` indicating the new status.
- **Purpose**: To change the status of an existing session batch in the data store.
- **Example**:
  ```python
  updated_batch = dao.update_batch_status("batch123", SessionBatchStatus.COMPLETED)
  ```