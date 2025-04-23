## Documentation for `MongoClickstreamDao`

### Functionality
MongoClickstreamDao provides a MongoDB-based implementation of the ClickstreamDao interface. It is used to store and retrieve user session data, including sessions, events, and batches.

### Inheritance
This class inherits from the ClickstreamDao interface, ensuring it meets the contract required for clickstream data access.

### Parameters
- `mongo_database`: A pymongo database instance used to access the MongoDB collections for sessions, session events, and session batches.

### Motivation and Purpose
The main purpose of this class is to bridge the application with a MongoDB storage solution. It abstracts the details of MongoDB operations while providing clear methods to manipulate session-related data. This allows the rest of the system to work with clickstream data without worrying about the underlying database specifics.

### Usage
This module is intended for backend data processing where MongoDB is the selected storage solution. It can be used to perform CRUD operations on clickstream data with validations enforced by the underlying MongoDB database.

#### Example
Below is a simple example of how the MongoClickstreamDao might be instantiated:

```python
from pymongo import MongoClient
from embedding_studio.data_access.mongo.clickstream import MongoClickstreamDao

client = MongoClient('mongodb://localhost:27017/')
db = client['mydatabase']
clickstream_dao = MongoClickstreamDao(mongo_database=db)
```

## Method Documentation

### `register_session`
#### Functionality
Registers a new session in the database. If a session with the given ID already exists, the method returns the existing session with batch metadata. It increments the session batch and handles duplicate session registration by catching duplicate key errors.

#### Parameters
- `session`: The session object to be registered. It must conform to the expected session model.

#### Returns
- `RegisteredSession`: An object containing session details and associated batch information.

#### Usage
- **Purpose**: Registers a new session or retrieves an already registered session if it exists.

##### Example
```python
session = Session(...)
reg_session = dao.register_session(session)
```

### `update_session`
#### Functionality
Updates an existing session in the MongoDB database, replacing the session data with updated values and associating it with a new batch.

#### Parameters
- `session`: Session object with updated session data. It must include a valid session_id and other necessary attributes.

#### Usage
- **Purpose**: Update a session record in the database with new data and refreshed batch information.

##### Example
```python
# Assume session is a Session object with updated data
updated_session = mongo_clickstream_dao.update_session(session)
```

### `push_events`
#### Functionality
Stores a list of session events into the MongoDB database. It attempts a bulk insertion of all provided events. If duplicate events are encountered, a warning is logged and the process continues.

#### Parameters
- `session_events`: A list of session events to store. Each event represents an action within a user session.

#### Usage
- **Purpose**: Insert multiple session events in one call. Designed for batch processing of event data in the database.

##### Example
```python
dao.push_events(events_list)
```

### `mark_session_irrelevant`
#### Functionality
Marks a session as irrelevant in the database by setting its "is_irrelevant" flag to True. It updates the session document and returns the updated session if found.

#### Parameters
- `session_id`: The identifier of the session to mark as irrelevant.

#### Usage
- **Purpose**: To exclude a session from further processing by marking it as irrelevant in the database.

##### Example
```python
updated_session = dao.mark_session_irrelevant("session123")
if updated_session:
    print("Session marked as irrelevant")
```

### `get_session`
#### Functionality
Retrieve a session along with its associated events from the MongoDB database. Returns a SessionWithEvents object if found; otherwise, returns None.

#### Parameters
- `session_id`: A string representing the unique session identifier.

#### Return Value
- A SessionWithEvents object that includes session details and events, or None if the session is not found.

#### Usage
- **Purpose**: To fetch user session data for processing or analysis.

##### Example
```python
session = dao.get_session("abc123")
if session:
    # process session data
    print(session.events)
else:
    print("Session not found")
```

### `get_batch_sessions`
#### Functionality
Retrieve sessions belonging to a specific batch.

#### Parameters
- `batch_id`: ID of the batch to retrieve sessions for.
- `after_number`: Retrieve sessions with numbers greater than the given value.
- `limit`: Maximum number of sessions to retrieve.
- `events_limit`: Maximum number of events to retrieve per session.

#### Usage
- **Purpose**: Obtain a list of SessionWithEvents objects, filtered as needed.

##### Example
```python
session_dao = MongoClickstreamDao(...)
sessions = session_dao.get_batch_sessions(
    batch_id="example-id",
    after_number=10,
    limit=5,
    events_limit=20
)
```

### `get_batch`
#### Functionality
Retrieves a batch record by its unique ID. This method queries the underlying MongoDB collection and returns the associated batch if found, or None otherwise.

#### Parameters
- `batch_id`: A string denoting the unique identifier for the batch to be retrieved.

#### Usage
- **Purpose**: To obtain a batch object from the database using its ID.

##### Example
```python
dao = MongoClickstreamDao(mongo_database)
batch = dao.get_batch("batch_12345")
```

### `release_batch`
#### Functionality
Marks the current collecting batch as released. This method updates the batch status from collecting to released, assigns the provided release ID, and records the release timestamp. If a duplicate release is attempted, it returns the already released batch.

#### Parameters
- `release_id`: ID to assign to the released batch.

#### Returns
- An updated SessionBatch object if the release is successful, or None if no collecting batch is found.

#### Usage
- **Purpose**: Finalize the current batch by updating its status to released.

##### Example
```python
released_batch = dao.release_batch("release123")
```

### `update_batch_status`
#### Functionality
This method updates the status of a batch in the database by its unique identifier. It performs a find_one_and_update operation to set the new status based on the provided value, and returns the updated SessionBatch or None if no batch is found.

#### Parameters
- `batch_id`: The unique identifier of the batch.
- `status`: A SessionBatchStatus enum value indicating the new status.

#### Usage
- **Purpose**: Modify the state of a batch during processing events.

##### Example
```python
updated_batch = dao.update_batch_status(
    batch_id, SessionBatchStatus.collecting
)
if updated_batch:
    print("Batch status updated successfully.")
else:
    print("Batch not found.")
```

### `_get_session_events`
#### Functionality
Retrieves events for a specific session from MongoDB. It queries based on the provided session identifier and limits the number of events returned.

#### Parameters
- `session_id`: The unique identifier for the session.
- `limit`: Maximum number of events to retrieve. Defaults to 100.

#### Usage
- **Purpose**: To fetch events related to a session from the database.

##### Example
```python
# Retrieve up to 50 events for a session
session_events = dao._get_session_events("session123", limit=50)
```

### `_increment_session_batch`
#### Functionality
This method increments the session counter in the current collecting batch. If no collecting batch exists, it creates a new batch and assigns a creation timestamp.

#### Parameters
This method does not accept any parameters.

#### Returns
- Updated or newly created SessionBatch object.

#### Usage
Invoke this method to ensure a collecting batch exists and to update the session counter accordingly.

##### Example
```python
# Increment the session counter in the current batch
batch = dao._increment_session_batch()
print(batch.session_counter)
```