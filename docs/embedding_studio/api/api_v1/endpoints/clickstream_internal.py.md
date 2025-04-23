# Documentation for Session Improvement Methods

## Method: `use_session_for_improvement`

### Functionality
Submits a session for improvement by first verifying that the session exists and contains the expected events. It then schedules this session for inclusion in model improvement tasks. Sessions identified as payload-search are skipped because they are not suitable for training purposes.

### Parameters
- `body` (UseSessionForImprovementRequest): Contains the session ID (`session_id`) to specify which session to schedule. 

### Usage
- **Purpose**: Schedule a valid session for model improvement after verifying its contents.

#### Example
Use this endpoint with a POST request as shown below:

```
POST /session/use-for-improvement
Content-Type: application/json

{
  "session_id": "your_session_id_here"
}
```

---

## Method: `get_batch_sessions`

### Functionality
Retrieves a paginated batch of sessions for processing. It uses pagination and event limiting to efficiently handle large collections of sessions. The method returns a dictionary with the batch id, the last session number processed, and a list of session data ready for further improvement workflows.

### Parameters
- `batch_id`: A string identifier representing the session batch.
- `after_number`: An integer indicating the starting session number for pagination, defaults to 0.
- `limit`: An integer defining the maximum number of sessions to return, defaults to 10.
- `events_limit`: An integer specifying the maximum number of events per session, defaults to 100.

### Usage
- **Purpose**: To retrieve a subset of sessions for batch processing in model improvement tasks.

#### Example
A sample API call might look like:

```
GET /batch/sessions?batch_id=abc123&after_number=0&limit=10&events_limit=100
```

---

## Method: `release_batch`

### Functionality
Marks a batch of sessions as processed and ready for deployment. Finalizes the batch processing operation, making data available for downstream consumers. Returns a 404 error if no active batch exists.

### Parameters
- `body`: A BatchReleaseRequest object containing the release_id used to release the batch.

### Usage
- **Purpose**: To finalize batch processing and mark sessions as released.

#### Example
To release a batch, use the following call:

```
release_batch(body=BatchReleaseRequest(release_id="your_release_id"))
```