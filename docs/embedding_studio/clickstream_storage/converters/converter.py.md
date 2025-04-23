# ClickstreamSessionConverter Documentation

## Overview

The `ClickstreamSessionConverter` class is designed to convert clickstream session data into structured fine-tuning inputs, which incorporate query items, events, and associated metadata. This functionality streamlines the transformation of raw session data into a standardized representation suitable for model fine-tuning.

## Functionality

The main method, `convert`, processes session data, including search queries, results, and events, generating a structured format that can be used for model fine-tuning. The converter handles various input types, ensuring flexibility in adapting to different clickstream data formats.

## Parameters

### ClickstreamSessionConverter Class
- `item_type`: Class type for item metadata.
- `query_item_type`: Class type for query items (default: `TextQueryItem`).
- `fine_tuning_type`: Class type for fine-tuning inputs (default: `FineTuningInput`).
- `event_type`: Class type for session events (default: `DummySessionEventWithImportance`).

### `convert` Method
- `session_data`: A `SessionWithEvents` object or a dictionary containing session attributes such as search query, search metadata, results, and events.

## Returns
The `convert` method returns a `FineTuningInputWithItems` object that wraps a fine-tuning input with associated metadata items. This includes a query item and event importance scores.

## Usage

The primary purpose of the `ClickstreamSessionConverter` is to transform raw clickstream data into a structured input for model fine-tuning. It creates query items, filters events, and maps importance scores. Users can extend this base class to override conversion methods and customize the transformation logic according to specific project needs.

### Example

An example session data dictionary might look like this:

```python
session_data = {
  "search_query": "example query",
  "search_meta": {"key": "value"},
  "search_results": [
      {"object_id": 1, "meta": {"info": "sample"}},
      {"object_id": 2, "meta": {"info": "sample2"}}
  ],
  "events": [
      {"object_id": 1, "event_importance": 0.8}
  ],
  "created_at": "2021-01-01T00:00:00"
}
```

To use the converter:

```python
converter = ClickstreamSessionConverter(ItemMeta)
fine_tuning_input = converter.convert(session_data)
``` 

This example demonstrates how to instantiate the `ClickstreamSessionConverter` with the desired item metadata class and convert session data into a format ready for model fine-tuning.