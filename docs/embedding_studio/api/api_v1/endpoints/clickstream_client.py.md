## Documentation Summary

### create_session

#### Functionality
Registers a new user interaction session by validating and normalizing the provided timestamp. The session is then registered in the clickstream data store to track user search journeys from the beginning of interaction.

#### Parameters
- `body` (SessionCreateRequest): Contains session creation details including the `created_at` timestamp.

#### Usage
- **Purpose**: Initiate tracking of a new user session with search data.

#### Example
```python
from embedding_studio.api.api_v1.endpoints import clickstream_client
from embedding_studio.api.api_v1.schemas.clickstream_client import SessionCreateRequest

req = SessionCreateRequest(created_at="2022-01-01T00:00:00")
clickstream_client.create_session(req)
```

---

### get_session

#### Functionality
Retrieves a complete session by its ID including all related interaction events. If no session is found, it raises an HTTP 404 error.

#### Parameters
- `session_id`: A string representing the unique session ID.

#### Usage
- **Purpose:** Obtain session details and user interaction events.

#### Example
```python
from embedding_studio.api.api_v1.endpoints.clickstream_client import get_session

session = get_session("example_session_id")
```

---

### push_events

#### Functionality
Adds user interaction events to an existing session. The function processes and normalizes timestamps for each event before storing them in the clickstream data store. It captures user interactions like clicks to analyze user behavior.

#### Parameters
- `body`: Instance of SessionAddEventsRequest containing:
  - `session_id`: The unique identifier for the session.
  - `events`: A list of event objects, each having a timestamp and event data.

#### Usage
- **Purpose**: Appends interaction events to an existing session with normalized timestamps.

#### Example
```python
import requests

payload = {
  "session_id": "abc123",
  "events": [
    {"created_at": "2023-01-01T00:00:00Z", "name": "click"}
  ]
}

response = requests.post(
  "http://localhost:8000/session/events", json=payload
)
```

---

### mark_session_irrelevant

#### Functionality
Flags a session as irrelevant for analytics and model improvement. It marks sessions that should not influence search quality metrics or be used for model training. If the session does not exist, a 404 error is raised.

#### Parameters
- body: SessionMarkIrrelevantRequest. Contains:
  - `session_id` (str): Unique identifier for the session.

#### Usage
- **Purpose**: To flag a session as irrelevant for analytical or training use.

#### Example
```python
from embedding_studio.api.api_v1.endpoints.clickstream_client import mark_session_irrelevant
from embedding_studio.api.schema import SessionMarkIrrelevantRequest

request = SessionMarkIrrelevantRequest(session_id="your_session_id")
mark_session_irrelevant(request)
```

---

### _ensure_timestamp

#### Functionality
Ensures that a timestamp is valid or assigns a new UTC timestamp if missing. It validates the provided timestamp against allowed delta ranges. If the timestamp fails validation, an HTTPException is raised.

#### Parameters
- `request_timestamp` (Optional[int]): Input timestamp that may be None. When None, a new UTC timestamp is generated; otherwise, it is validated.

#### Usage
Used to normalize and verify timestamps during clickstream session handling.

#### Example
```python
timestamp = _ensure_timestamp(user_timestamp)
```