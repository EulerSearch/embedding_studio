# Clickstream Management in Embedding Studio

## Introduction

Clickstream data is a vital resource for improving search quality in Embedding Studio. It records how users interact with search results, providing insights that can be used to fine-tune embedding models and improve relevance. This guide explains how clickstream data flows through the system and how to leverage it effectively.

## Core Concepts

- **Session**: A continuous user interaction with the search system
- **Search Events**: Individual user actions within a session (clicks, views, etc.)
- **Relevance Feedback**: Converting user interactions into training signals
- **Model Improvement**: Using session data to fine-tune embedding models

## Architecture Overview

The clickstream system consists of several components:

1. **Client API endpoints**: For registering sessions and events
2. **Internal API endpoints**: For processing and utilizing session data
3. **Data Access Objects (DAOs)**: For storing and retrieving session data
4. **Converters**: For transforming session data into fine-tuning inputs
5. **Workers**: For processing sessions and improving models

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Client Events  │────▶│  Session Store  │────▶│  Model Training │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## API Endpoints

### Client-Facing Endpoints

The `clickstream_client.py` module provides these endpoints for client applications:

#### 1. Create Session

```python
@router.post("/session", status_code=status.HTTP_200_OK)
def create_session(body: SessionCreateRequest) -> None:
    """
    Creates a new user interaction session with search data.
    """
```

Use this endpoint when a user starts a new search session. It initializes tracking for that session.

#### 2. Get Session

```python
@router.get("/session", status_code=status.HTTP_200_OK, response_model=SessionGetResponse)
def get_session(session_id: str) -> SessionWithEvents:
    """
    Retrieves a complete session by ID with all related interaction events.
    """
```

Use this to retrieve a complete session with all its events, useful for debugging or analytics.

#### 3. Push Events

```python
@router.post("/session/events", status_code=status.HTTP_200_OK)
def push_events(body: SessionAddEventsRequest) -> None:
    """
    Adds user interaction events to an existing session.
    """
```

Use this endpoint to record user interactions such as clicks, views, or purchases.

#### 4. Mark Session Irrelevant

```python
@router.post("/session/irrelevant", status_code=status.HTTP_200_OK)
def mark_session_irrelevant(body: SessionMarkIrrelevantRequest) -> None:
    """
    Flags a session as irrelevant for analytics and model improvement.
    """
```

Use this to exclude certain sessions from training data, such as bot sessions or test searches.

### Internal Endpoints

The `clickstream_internal.py` module provides these endpoints for system processes:

#### 1. Use Session for Improvement

```python
@router.post("/session/use-for-improvement", status_code=status.HTTP_200_OK)
def use_session_for_improvement(body: UseSessionForImprovementRequest) -> None:
    """
    Submits a session for use in search quality improvement processes.
    """
```

Internal endpoint to add sessions to the model improvement queue.

#### 2. Get Batch Sessions

```python
@router.get("/batch/sessions", status_code=status.HTTP_200_OK, response_model=BatchSessionsGetResponse)
def get_batch_sessions(batch_id: str, after_number: int = 0, limit: int = 10, events_limit: int = 100):
    """
    Retrieves a paginated batch of sessions for processing.
    """
```

Internal endpoint to retrieve batches of sessions for processing.

#### 3. Release Batch

```python
@router.post("/batch/release", status_code=status.HTTP_200_OK, response_model=BatchReleaseResponse)
def release_batch(body: BatchReleaseRequest):
    """
    Marks a batch of sessions as processed and ready for deployment.
    """
```

Internal endpoint to finalize batch processing.

## Data Flow

### 1. Session Creation

When a search is performed:

1. Client creates a session with search query and metadata
2. System assigns a unique session ID and batch ID
3. Session is stored in the database

```python
# Example session creation
response = requests.post(
    "https://api.embeddingstudio.com/api/v1/clickstream/session",
    json={
        "session_id": "user_123_search_456",
        "search_query": "red summer dress",
        "search_meta": {"filters": {"category": "clothing"}},
        "search_results": [
            {"object_id": "prod_789", "meta": {"title": "Red sundress"}}
        ],
        "created_at": int(time.time())
    }
)
```

### 2. Event Recording

As users interact with search results:

1. Client records events (clicks, purchases, etc.)
2. Events include object IDs and timestamps
3. Events are associated with the session ID

```python
# Example event recording
response = requests.post(
    "https://api.embeddingstudio.com/api/v1/clickstream/session/events",
    json={
        "session_id": "user_123_search_456",
        "events": [
            {
                "event_id": "click_001",
                "object_id": "prod_789",
                "event_type": "click",
                "meta": {"position": 3},
                "created_at": int(time.time())
            }
        ]
    }
)
```

### 3. Session Processing

Later, system processes use the data:

1. Sessions are retrieved in batches
2. Events are analyzed to infer relevance
3. Sessions are converted to training inputs
4. Model improvement is scheduled

## Converting to Training Data

The `ClickstreamSessionConverter` transforms sessions into fine-tuning inputs:

```python
# Create a converter instance
converter = ClickstreamSessionConverter(
    item_type=ProductItemMeta,
    query_item_type=TextQueryItem,
    fine_tuning_type=FineTuningInput,
    event_type=ClickStreamSessionEvent
)

# Convert a session to training input
training_input = converter.convert(session_data)
```

The conversion process:

1. Extracts the search query and metadata
2. Maps event types to importance scores
3. Identifies which results were interacted with
4. Creates structured training inputs

## MongoDB Storage

Sessions are stored in MongoDB with several collections:

- `sessions`: Stores session data, search queries, and metadata
- `session_events`: Stores individual user interaction events
- `session_batches`: Groups sessions for processing
- `sessions_for_improvement`: Tracks sessions used for model training

## Best Practices

### Tracking Events

1. **Track diverse events**: clicks, views, add-to-cart, purchases
2. **Include positions**: Record where in results the item appeared
3. **Add timestamps**: Time-based analysis can reveal patterns
4. **Record search contexts**: Include filters, sorting, and user segments

### Session Analysis

1. **Analyze session volume**: Low session count may indicate poor coverage
2. **Check event distribution**: Ensure balanced representation of event types
3. **Monitor conversions**: Track click-through and purchase rates
4. **Compare segments**: Look for variations across user groups

### Data Quality

1. **Filter bot traffic**: Use mark_session_irrelevant for non-human sessions
2. **Validate event sequencing**: Ensure logical order of events
3. **Check payload content**: Ensure search metadata is properly structured
4. **Monitor batch processing**: Track batch completion and error rates

## Using ClickstreamDao

The `ClickstreamDao` interface provides methods for working with session data:

```python
# Get a session with its events
session = context.clickstream_dao.get_session(session_id)

# Mark a session for model improvement
task = context.sessions_for_improvement.create(
    schema=SessionForImprovementCreateSchema(
        session_id=session_id,
    ),
    return_obj=True,
)
context.sessions_for_improvement.update(obj=task)

# Get batches of sessions for processing
sessions = context.clickstream_dao.get_batch_sessions(
    batch_id=batch_id,
    after_number=last_processed_number,
    limit=100
)
```
