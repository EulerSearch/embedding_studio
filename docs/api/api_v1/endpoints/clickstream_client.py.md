# Documentation for Clickstream Processing API

---

## Method: `POST /session` — create_session

### Functionality
Creates a new user interaction session. Stores initial query and returned search results for tracking user search behavior.

### Request Parameters
- `session_id` *(str)*: Unique identifier for the session.
- `search_query` *(str)*: The original query issued by the user.
- `search_results` *(List[SearchResultItem])*: The ranked results returned to the user.
- `search_meta` *(Dict, optional)*: Additional search execution metadata (e.g. latency).
- `payload_filter` *(PayloadFilter, optional)*: Payload constraints applied during search.
- `sort_by` *(SortByOptions, optional)*: Sorting criteria used during the search.
- `user_id` *(str, optional)*: ID of the user initiating the session.
- `created_at` *(int, optional)*: Timestamp in milliseconds since epoch.

### Request JSON Example
```json
{
  "session_id": "sess-123",
  "search_query": "transformer optimization",
  "search_results": [
    {
      "object_id": "obj-1",
      "rank": 1,
      "meta": {
        "source": "hf_models",
        "score": 0.92
      }
    }
  ],
  "search_meta": {
    "latency_ms": 450
  },
  "payload_filter": {
    "task": "text-classification"
  },
  "sort_by": {
    "field": "downloads",
    "order": "desc"
  },
  "user_id": "user-456",
  "created_at": 1701012345678
}
```

---

## Method: `GET /session` — get_session

### Functionality
Returns a complete session along with stored metadata and user interaction events.

### Response Fields
- Same as `SessionCreateRequest`, plus:
- `created_at`: Mandatory timestamp.
- `is_irrelevant`: Flag indicating whether this session is ignored during analysis.
- `events`: List of `SessionEvent` objects.

### Example Response
```json
{
  "session_id": "sess-123",
  "search_query": "transformer optimization",
  "search_results": [
    {
      "object_id": "obj-1",
      "rank": 1.0
    }
  ],
  "search_meta": {
    "latency_ms": 450
  },
  "payload_filter": {
    "task": "text-classification"
  },
  "sort_by": {
    "field": "downloads",
    "order": "desc"
  },
  "user_id": "user-456",
  "created_at": 1701012345678,
  "is_irrelevant": false,
  "events": [
    {
      "event_id": "evt-1",
      "object_id": "obj-1",
      "event_type": "click",
      "created_at": 1701012350000,
      "meta": {
        "rank_position": 1
      }
    }
  ]
}
```

---

## Method: `POST /session/events` — push_events

### Functionality
Adds one or more interaction events to an existing session.

### Request Fields
- `session_id`: ID of the existing session.
- `events`: List of interaction events.

Each event includes:
- `event_id`: Unique identifier.
- `object_id`: ID of the result the user interacted with.
- `event_type`: Defaults to `"click"`.
- `created_at`: Millisecond timestamp of the event.
- `meta`: Optional metadata (e.g., rank clicked, time to click).

### Request JSON Example
```json
{
  "session_id": "sess-123",
  "events": [
    {
      "event_id": "evt-1",
      "object_id": "obj-1",
      "event_type": "click",
      "created_at": 1701012350000,
      "meta": {
        "rank_position": 1
      }
    }
  ]
}
```

---

## Method: `POST /session/irrelevant` — mark_session_irrelevant

### Functionality
Marks a session as irrelevant so it's excluded from analytics and model fine-tuning.

### Request JSON Example
```json
{
  "session_id": "sess-123"
}
```
- `session_id`: The session to exclude.

### Expected Response
- HTTP 200 OK if successful
- HTTP 404 if session not found

---
