## Documentation for `AbstractSuggestionPhraseManager`

### Functionality
This abstract class defines the interface for managing suggestion phrases stored in a MongoDB collection. It ensures that all derived phrase managers implement methods for converting phrases, adding new phrases, deleting phrases, and updating probability scores.

### Motivation
The class promotes consistent implementation of core operations for phrase suggestion management. It provides a contract that concrete classes must follow, reducing errors and increasing maintainability.

### Inheritance
AbstractSuggestionPhraseManager inherits from Python's built-in ABC class. This inheritance enforces that every subclass implements all abstract methods defined in this interface.

### Example Implementation
Below is an example of a concrete class extending this abstract class:

```python
class MyPhraseManager(AbstractSuggestionPhraseManager):
    def convert_phrase_to_request(self, phrase, domain=None):
        request = SuggestingRequest(text=phrase)
        if domain:
            request.domain = domain
        return request

    def add(self, phrases):
        # Convert phrases to documents and add them to MongoDB
        pass

    def delete(self, phrase_ids):
        # Delete phrases from MongoDB using their IDs
        pass

    def update_probability(self, phrase_id, new_probability):
        # Update the probability score for a phrase
        pass
```

### Method Descriptions

#### Method: `convert_phrase_to_request`

- **Functionality**: Converts an input phrase and an optional domain into a SuggestingRequest object for suggestion operations.
- **Parameters**:
  - `phrase`: The input string to be converted into a request.
  - `domain`: (Optional) Domain to associate with the request.
- **Usage**:
  - **Purpose**: Standardizes input phrases into a structured suggestion request for further processing.
  
```python
def convert_phrase_to_request(self, phrase: str, domain: Optional[str] = None) -> SuggestingRequest:
    request = SuggestingRequest(text=phrase)
    if domain:
        request.domain = domain
    return request
```

#### Method: `add`

- **Functionality**: Insert multiple suggesting phrase documents into a MongoDB collection.
- **Parameters**:
  - `phrases`: List of SuggestingPhrase objects to be added.
- **Usage**:
  - **Purpose**: Insert multiple phrase documents into the database and obtain their unique insertion IDs.

```python
def add(self, phrases: List[SuggestingPhrase]) -> List[str]:
    documents = [phrase.dict() for phrase in phrases]
    result = self._collection.insert_many(documents)
    return [str(id) for id in result.inserted_ids]
```

#### Method: `delete`

- **Functionality**: Deletes documents from the MongoDB collection by their MongoDB _id values.
- **Parameters**:
  - `phrase_ids`: List[str] -- List of string IDs that identify the phrases to be deleted.
- **Usage**:
  - **Purpose**: Remove multiple suggestion phrases from the storage.

```python
# Delete example phrases
manager.delete([
    "5f43abcd1234ef567890abcd",
    "5f43abcd1234ef567890abce"
])
```

#### Method: `update_probability`

- **Functionality**: Updates the probability score for a specific phrase document in the MongoDB collection.
- **Parameters**:
  - `phrase_id`: String ID of the phrase to update.
  - `new_probability`: New probability value to set.
- **Usage**:
  - **Purpose**: Update a phrase document's probability score in the database.

```python
def update_probability(self, phrase_id: str, new_probability: float) -> None:
    object_id = ObjectId(phrase_id)
    self._collection.update_one(
        {"_id": object_id},
        {"$set": {"probability": new_probability}}
    )
```

#### Method: `add_labels`

- **Functionality**: Adds labels to a document without duplicating existing ones.
- **Parameters**:
  - `phrase_id`: String ID of the phrase to update.
  - `labels`: List of label strings to add.
- **Usage**:
  - **Purpose**: Update a document by ensuring new labels are added only if they are not already present.

```python
def add_labels(self, phrase_id: str, labels: List[str]) -> None:
    object_id = ObjectId(phrase_id)
    self._collection.update_one(
        {"_id": object_id},
        {"$addToSet": {"labels": {"$each": labels}}}
    )
```

#### Method: `remove_labels`

- **Functionality**: Removes specified labels from a phrase document.
- **Parameters**:
  - `phrase_id`: String identifier for the phrase to update.
  - `labels`: List of label strings to remove from the document.
- **Usage**:
  - **Purpose**: Modify a document by removing selected labels.

```python
def remove_labels(self, phrase_id: str, labels: List[str]) -> None:
    object_id = ObjectId(phrase_id)
    self._collection.update_one(
        {"_id": object_id},
        {"$pull": {"labels": {"$in": labels}}}
    )
```

#### Method: `remove_all_label_values`

- **Functionality**: Removes specified labels from all documents that contain any of the given labels.
- **Parameters**:
  - `labels`: List of strings representing labels to remove.
- **Usage**:
  - **Purpose**: Remove labels from all documents in the collection that have any of the specified labels.

```python
# Assuming an implementation exists
manager.remove_all_label_values(["label1", "label2"])
```

#### Method: `add_domains`

- **Functionality**: Adds domains to a phrase document without duplicating existing domains.
- **Parameters**:
  - `phrase_id`: String ID of the phrase to be updated.
  - `domains`: List of domain strings to add.
- **Usage**: 
  - **Purpose**: Updates a document in the MongoDB collection using '$addToSet' with '$each' to ensure only new domains are added.

```python
def add_domains(self, phrase_id: str, domains: List[str]) -> None:
    object_id = ObjectId(phrase_id)
    self._collection.update_one(
        {"_id": object_id},
        {"$addToSet": {"domains": {"$each": domains}}}
    )
```

#### Method: `remove_domains`

- **Functionality**: Removes specified domain strings from a phrase document in the database.
- **Parameters**:
  - `phrase_id`: String ID of the phrase document to be updated.
  - `domains`: List of domain strings to remove from the document.
- **Usage**:
  - **Purpose**: Remove unwanted domains from a phrase document while retaining other domain values.

```python
def remove_domains(self, phrase_id: str, domains: List[str]) -> None:
    object_id = ObjectId(phrase_id)
    self._collection.update_one(
        {"_id": object_id},
        {"$pull": {"domains": {"$in": domains}}}
    )
```

#### Method: `remove_all_domain_values`

- **Functionality**: Removes specified domains from all documents that contain them.
- **Parameters**:
  - `domains`: List of domain strings to remove from all documents.
- **Usage**:
  - **Purpose**: Clean up domain information across all stored suggestion phrases.

```python
def remove_all_domain_values(self, domains: List[str]) -> None:
    self._collection.update_many(
        {"domains": {"$in": domains}},
        {"$pull": {"domains": {"$in": domains}}}
    )
```

#### Method: `get_info_by_id`

- **Functionality**: Fetches a phrase document from the MongoDB collection using the provided MongoDB _id.
- **Parameters**:
  - `phrase_id`: A string representing the MongoDB _id of the phrase to retrieve.
- **Usage**:
  - **Purpose**: To retrieve detailed information about a suggestion phrase stored in a MongoDB collection and convert it into a SearchDocument for further processing.

```python
# Assume 'manager' is an instance of a class implementing the AbstractSuggestionPhraseManager interface.
search_doc = manager.get_info_by_id("609f8f3d2f75b0b8f1a2c986")
print(search_doc.text)
```

#### Method: `list_phrases`

- **Functionality**: Returns a paginated list of full phrase documents from the MongoDB collection.
- **Parameters**:
  - `offset`: Number of documents to skip for pagination.
  - `limit`: Maximum number of documents to return.
- **Usage**:
  - **Purpose**: Retrieve a subset of phrase documents for display or further processing.

```python
# Retrieve the first 100 phrase documents
phrases = manager.list_phrases(offset=0, limit=100)

for phrase in phrases:
    print(phrase.text)
```