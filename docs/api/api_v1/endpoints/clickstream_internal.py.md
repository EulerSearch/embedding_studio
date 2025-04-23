# Documentation for Internal Clickstream Management API

---

## Method: `POST /session/use-for-improvement`

### Functionality
Marks a session to be used in model improvement workflows. Skips sessions that are either payload-search-based or have no valid event history.

### Request JSON Example
```json
{
  "session_id": "sess-abc123"
}
```
- `session_id` *(str)*: Unique identifier of the session to be used for training or evaluation.

---

## Method: `GET /batch/sessions`

### Functionality
Retrieves a paginated set of sessions within a batch for analysis or processing. Returns session metadata and user interaction events.

### Query Parameters
- `batch_id` *(str)*: Identifier for the batch to retrieve.
- `after_number` *(int, optional, default=0)*: Start from this session number.
- `limit` *(int, optional, default=10)*: Max number of sessions to return.
- `events_limit` *(int, optional, default=100)*: Max number of events per session.

### Response JSON Example
```json
{
  "batch_id": "batch-001",
  "last_number": 123,
  "sessions": [
    {
      "session_number": 123,
      "session_id": "sess-abc123",
      "search_query": "ai transformer",
      "search_results": [
        {
          "object_id": "obj-1",
          "rank": 1.0
        }
      ],
      "search_meta": {
        "latency_ms": 210
      },
      "payload_filter": {
        "task": "classification"
      },
      "sort_by": {
        "field": "stars",
        "order": "desc"
      },
      "user_id": "user-123",
      "created_at": 1701234567890,
      "is_irrelevant": false,
      "events": [
        {
          "event_id": "evt-001",
          "object_id": "obj-1",
          "event_type": "click",
          "created_at": 1701234570000,
          "meta": {
            "clicked_rank": 1
          }
        }
      ]
    }
  ]
}
```
- `batch_id`: ID of the current session batch.
- `last_number`: Last session number retrieved, for pagination.
- `sessions`: List of full session objects with events and metadata.
- `session_number`: Sequential number used for pagination.
- `events`: List of user interaction events (`click`, etc.) attached to each session.

---

## Method: `POST /batch/release`

### Functionality
Marks the end of batch session collection and makes the sessions available for downstream processing (e.g. training, evaluation, metrics).

### Request JSON Example
```json
{
  "release_id": "release-v1"
}
```
- `release_id` *(str)*: The unique identifier of the release batch to finalize.

### Response JSON Example
```json
{
  "release_id": "release-v1",
  "batch_id": "batch-001",
  "released_at": 1701234999999
}
```
- `release_id`: The batch release identifier.
- `batch_id`: The batch from which sessions were released.
- `released_at`: Timestamp of release operation in milliseconds since epoch.

---