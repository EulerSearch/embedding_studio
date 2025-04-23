## Documentation for `SessionEventWithImportance`

### Functionality

This abstract class defines a session event that carries an importance score. It combines Pydantic's BaseModel for data validation and ABC for abstract methods enforcement. Its design enforces a blueprint for events needing a scoring mechanism to indicate priority or relevance.

### Parameters

- `object_id`: Unique identifier for the associated event object.

### Inheritance

- Inherits from ABC to dictate abstract methods.
- Inherits from BaseModel to leverage Pydantic's data validation and instantiation.

### Usage

**Purpose:** Establish a contract for event classes that incorporate importance scoring.

#### Example Implementation

```python
class CustomEvent(SessionEventWithImportance):
    importance: float

    @property
    def event_importance(self) -> float:
        return self.importance * 1.5

    @classmethod
    def from_model(cls, event: SessionEvent) -> "CustomEvent":
        return cls(
            object_id=event.object_id,
            importance=event.meta.get("score", 1)
        )
```

---

## Documentation for `SessionEventWithImportance.event_importance`

### Functionality

This property returns a numeric value that represents the importance of a session event. It provides a score used to assess and rank event relevance.

### Parameters

- **None**: This is a property method and accepts no parameters.

### Usage

- **Purpose**: Retrieve the importance score from a session event.

#### Example

```python
# Assuming an instance of a class extending 
# SessionEventWithImportance
importance = event.event_importance
print(f"Importance: {importance}")
```

---

## Documentation for `SessionEventWithImportance.from_model`

### Functionality

Converts a `SessionEvent` into an instance of a session event with an importance score. This method extracts event data and maps it to create a new event object with importance info.

### Parameters

- `event`: An instance of `SessionEvent` containing event details and metadata.

### Usage

- **Purpose**: Transform a raw event model into a session event object with an associated importance metric.

#### Example

```python
event = SessionEvent(object_id="abc123", 
                     meta={"importance": 0.75})
session_event = DummySessionEventWithImportance.from_model(event)
print(session_event.event_importance)
```

---

## Documentation for `DummySessionEventWithImportance`

### Functionality

The DummySessionEventWithImportance class is a concrete implementation of the SessionEventWithImportance abstract class. It is designed primarily for testing or default usage, providing a mechanism to assign a preset importance score to session events.

### Parameters

- `object_id`: Unique identifier for the event's associated object.
- `importance`: A float representing the event's importance. Defaults to 1.0.

### Usage

- Purpose: Default implementation of the SessionEventWithImportance interface, useful for testing scenarios.

#### Example

A typical usage example:

```python
event = SessionEvent(...initial values...)
dummy_evt = DummySessionEventWithImportance.from_model(event)
print(dummy_evt.event_importance)
```

---

## Documentation for `DummySessionEventWithImportance.event_importance`

### Functionality

Returns the importance score of a session event. This value is defined by the instance attribute and represents how important the event is.

### Parameters

- None

### Usage

- Purpose: To fetch the importance score for a dummy session event.

#### Example

```python
# Create a dummy session event with a specific importance
event = DummySessionEventWithImportance(
    object_id='123',
    importance=0.85
)
print(event.event_importance)  # Output: 0.85
```

---

## Documentation for `DummySessionEventWithImportance.from_model`

### Functionality

This method creates a new instance of DummySessionEventWithImportance from a given SessionEvent model. It extracts the object_id and the associated importance value, defaulting to 1.0 if not provided in the event metadata.

### Parameters

- `event`: A SessionEvent object that contains attributes like object_id and meta. The meta attribute may include the importance value used for initializing the instance.

### Usage

- **Purpose** - Convert a SessionEvent model into a DummySessionEventWithImportance instance for testing or basic usage.

#### Example

Assuming you have a SessionEvent instance named event, you can convert it as follows:

```python
dummy_event = DummySessionEventWithImportance.from_model(event)
print(dummy_event.event_importance)
```