# Collect clickstream

A clickstream is a sequence of user search sessions. A search session is a series of user clicks on search results.
Clickstreams are used to gather user feedback and improve the model based on it.

## Search session

A search session is a series of user clicks on search results. A search session is identified by a unique identifier
`session_id`.

Typically, the workflow involves the creation of a search session, followed by populating it
with user clicks. Subsequently, the session is closed and submitted for fine-tuning. It's worth noting that the same
search session can be submitted for fine-tuning multiple times, including for
[different algorithms](fine_tuning_method.md). This flexibility allows for iterative refinement and enhancement of the
search model.

To register a new search session, send a POST request to the `/api/v1/clickstream/session` endpoint:

```shell
curl -X POST 'http://localhost:5000/api/v1/clickstream/session' \
-H 'Content-Type: application/json' \
-d '{
    "session_id": "c327c30bd3db459093e5f5d254bdd144,
    "search_query": "user search query",    
    "search_results":[
        {
            "object_id": "2f827cbb9fe544cdaca44b2dafd37785",
            "rank": 0.3
            "item_info": {}
        },
        {
            "object_id": "e000f005dc1343ea8ddf234c557832a5"
        },
        {
            "object_id": "96365929fa5b4286b3a04639cbc7c1d5"
        }
    ],
    "search_meta": {},
    "user_id" : "b604bbb1a57a418d925ebc9d615de99a",
    "created_at": 1703376873, 
}'
```

where:

* `session_id` - unique search session identifier
* `search_query` - user's search query
* `search_results` - list of search result object ids with optional rank and meta
* `search_meta` -  (optional) meta-information about the search query (any data)
* `user_id` - (optional) your user identifier
* `created_at` - (optional) utc timestamp of session start. Value by default: current time on server

## Events

Events refer to any occurrences that you can register within a search session. For instance, these can include user
clicks or other user interactions. In the context of a search session, events serve as recorded actions that contribute
to improvement of the search experience.

To add events (clicks) to registered search session, send a POST request to the `/api/v1/clickstream/session/events`
endpoint:

```shell
curl -X POST 'http://localhost:5000/api/v1/clickstream/session/events' \
-H 'Content-Type: application/json' \
-d '{
    "session_id": "search_session_id",    
    "events": [
        {
            "event_id": "472cc4232af2449a97317d0057c9a2cf",
            "object_id": "2f827cbb9fe544cdaca44b2dafd37785",
            "item_info": {},
            "event_type": "click",
            "created_at": 1703376873
        },
        {
            "event_id": "124b6bc39e9f43e7b5be6b992d94f505",
            "object_id": "2f827cbb9fe544cdaca44b2dafd37785",
            "item_info": {},
            "created_at": 1703376874
        },
        {
            "event_id": "b0f4adcac63b4a35a02a45f06b0b1cb9",
            "object_id": "e000f005dc1343ea8ddf234c557832a5",           
        }  
    ]
}'
```

where

* `session_id` - id of registered search session
* `events` - list of new session events with:
    * `event_id` - event identifier (must be unique for this session)
    * `object_id` - search result object id
    * `meta` - (optional) any meta data
    * `event_type` - (optional) event type. Value by default: `click`
    * `created_at` - (optional) utc timestamp of event. Value by default: current time on server

After you have gathered the required amount of user feedback, you can release the session batch and use it for
fine-tuning.
To do this, send a POST request to the `/api/v1/clickstream/internal/batch/release` endpoint:

```shell
curl -X POST '127.0.0.1:5000/api/v1/clickstream/internal/batch/release' \
-H 'Content-Type: application/json' \
-d '{
    "release_id": "090bda77a5e048d59a1e310dc20fbd6d"
}'
```

where `release_id` - unique identifier of release (idempotency key).

In response, you will receive:

```json
{
  "batch_id": "65844a671089823652b83d43"
}
```

where `batch_id` - unique identifier of clickstream session batch - it can be used to run fine-tuning