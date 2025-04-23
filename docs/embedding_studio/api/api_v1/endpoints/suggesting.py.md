## Documentation Overview

### Method: `get_suggestions`

#### Parameters
- **phrase**: The input text for which suggestions are generated.
- **domain**: The context or domain to refine the suggestion results.
- **top_k**: The number of suggestions to retrieve.

#### Usage
- **Purpose**: Retrieve and format suggestion recommendations based on an input phrase and domain.

##### Example
Send a POST request to the `/get-top-k` endpoint with a JSON body:
```json
{
  "phrase": "example text",
  "domain": "sample_domain",
  "top_k": 5
}
```

### Method: `delete_phrase`

#### Functionality
Deletes a phrase from the system by its unique ID. It acts as an HTTP DELETE endpoint that calls the phrase manager to remove a phrase from the underlying storage.

#### Parameters
- **phrase_id** (str): The unique identifier of the phrase to delete.

#### Usage
- **Purpose**: Permanently remove a phrase from the system.
- Trigger this function by making an HTTP DELETE request to the endpoint with the phrase ID in the URL.

##### Example
To delete a phrase with ID "abc123", send an HTTP DELETE request:
```
DELETE /phrases/delete/abc123
```

### Method: `add_labels`

#### Functionality
Adds labels to an existing suggestion phrase. The function takes a request containing the phrase ID and a list of labels to add. It verifies the phrase exists and raises an HTTP 404 error if not.

#### Parameters
- **body**: An instance of `SuggestingPhrasesAddLabelsRequest`.
  - **phrase_id** (str): Unique identifier of the phrase.
  - **labels** (List[str]): List of labels to attach.

#### Usage
- **Purpose**: Associate new labels with a suggestion phrase.

##### Example
Payload:
```json
{
  "phrase_id": "abc123",
  "labels": ["tag1", "tag2"]
}
```

### Method: `remove_labels`

#### Functionality
This endpoint removes specified labels from a phrase. It uses the underlying phrases manager, and if the phrase ID is not found, a 404 HTTP error is raised.

#### Parameters
- **phrase_id**: The unique identifier of the phrase.
- **labels**: A list of labels to remove from the phrase.

#### Usage
- **Purpose**: To update a phrase by removing one or more labels.

##### Example
Request payload example:
```json
{
  "phrase_id": "example_id",
  "labels": ["urgent", "review"]
}
```

### Method: `add_domains`

#### Functionality
This endpoint adds new domains to an existing suggestion phrase. It delegates the operation to the phrases manager to associate domains with the phrase. If the phrase is not found, a 404 error is returned.

#### Parameters
- **phrase_id**: Unique identifier of the suggestion phrase.
- **domains**: List of domain names to add to the phrase.

#### Usage
- **Purpose**: Update a suggestion phrase by linking it with relevant domains.

##### Example
Request example:
```json
POST /phrases/add-domains
{
  "phrase_id": "unique_phrase_id",
  "domains": ["domain1", "domain2"]
}
```

### Method: `remove_domains`

#### Functionality
Removes specified domains from a suggestion phrase by updating the phrase and removing the domain labels provided in the request.

#### Parameters
- **phrase_id**: Identifier for the suggestion phrase.
- **domains**: List of domains to remove from the phrase.

#### Usage
- **Purpose**: Refine suggestion visibility by removing irrelevant domains.

##### Example
Request body:
```json
{
  "phrase_id": "example_id",
  "domains": ["domain1", "domain2"]
}
```

### Method: `update_probability`

#### Functionality
Updates the probability value of a suggestion phrase. It receives a request body that includes the phrase's unique identifier and the new probability value. It then delegates this update to the phrases manager.

#### Parameters
- **body**: An instance of `SuggestingPhrasesAdjustProbabilityRequest`.
  - **phrase_id**: The unique identifier for the suggesting phrase.
  - **new_prob**: The new probability value to assign.

#### Usage
- **Purpose**: Adjust the ranking probability for a phrase. This endpoint is used for fine-tuning suggestion order.

##### Example
```bash
curl -X POST http://<host>/phrases/update-probability \
     -H "Content-Type: application/json" \
     -d '{"phrase_id": "some-id", "new_prob": 0.7}'
```

### Method: `get_phrase_info`

#### Functionality
Retrieves information about a phrase using the provided phrase ID. Internally, it calls the phrase manager to fetch details such as the phrase text, its probability, and associated labels. The result is wrapped in the SuggestingPhrase model.

#### Parameters
- **phrase_id** (str): The unique identifier of the phrase. Provided as a query parameter.

#### Usage
- **Purpose**: To obtain detailed information about a phrase by its ID. In case of an invalid phrase ID is supplied, a 404 HTTP exception is raised.

##### Example
A typical request:
```
GET /phrases/get-info?phrase_id=123
```
Expected response (for a valid phrase ID):
```json
{
  "phrase": "example phrase",
  "prob": 0.75,
  "labels": ["label1", "label2"]
}
```

### Method: `list_phrases`

#### Functionality
Return a paginated list of phrase IDs.

#### Parameters
- **offset**: Starting index for phrase pagination.
- **limit**: Maximum number of phrases to retrieve.

#### Usage
- **Purpose**: Retrieve a paginated list of phrases for the API.

##### Example
Request body:
```json
{
  "offset": 0,
  "limit": 10
}
```