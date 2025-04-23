# Documentation for CRUDBase Class and Its Methods

## CRUDBase Class

### Functionality
This module supplies a generic base for CRUD operations on a MongoDB collection. It leverages Pydantic models to validate input and output, ensuring consistent conversions and error handling.

### Purpose and Motivation
- Simplify CRUD method implementation by reusing common operations.
- Ensure uniform data validation and conversion with Pydantic.
- Manage MongoDB ObjectIDs consistently across operations.

### Inheritance
CRUDBase is built as a generic class inheriting Python's Generic. It uses three type variables:
- `SchemaInDbType`: Represents the database schema model.
- `CreateSchemaType`: Model format for creating new records.
- `UpdateSchemaType`: Model format for updating existing records.

### Usage Example
Below is a simple example of subclassing CRUDBase:

```python
from pymongo.collection import Collection
from pydantic import BaseModel
from embedding_studio.data_access.mongo.crud_base import CRUDBase

class MySchema(BaseModel):
    name: str
    value: int

class MyCRUD(CRUDBase[MySchema, MySchema, MySchema]):
    pass

collection: Collection = my_mongo_client['my_database']['my_collection']
crud = MyCRUD(collection, MySchema)
```

---

## Documentation for CRUDBase Methods

### `to_object_id`

#### Functionality
Converts a string or ObjectId into a valid ObjectId. If the input is a string, it checks if it is a valid ObjectId. Returns the corresponding ObjectId or None if invalid. If the input is already an ObjectId, it returns it directly.

#### Parameters
- `id`: A string representing a MongoDB ObjectId or an actual ObjectId. If a string is provided, it must be valid according to MongoDB standards.

#### Usage
- **Purpose:** Ensures that an identifier is in ObjectId format for MongoDB operations.

#### Example
If you receive an ID as a string:
```python
object_id = CRUDBase.to_object_id('607f1f77bcf86cd799439011')
```
If the string is invalid, the method returns None.

---

### `CRUDBase.exists`

#### Functionality
Checks if an object with a specific ID exists in the MongoDB collection. The method converts the given string or ObjectId into a valid ObjectId and then queries the collection with a limit of one. It returns a boolean indicating whether the document exists.

#### Parameters
- `id`: A string or ObjectId representing the object's identifier. It is converted and validated before querying the database.

#### Return Value
- `bool`: True if the object exists; False otherwise.

#### Usage
- **Purpose**: To verify the existence of a document in the database before performing operations like updates or deletions.

#### Example
Assuming you have an instance of CRUDBase:
```python
exists_result = crud_instance.exists("60f...123")
if exists_result:
    print("Document exists!")
else:
    print("Document not found!")
```

---

### `CRUDBase.get`

#### Functionality
Retrieve an object from the MongoDB collection by its ID. The method converts the given ID into an ObjectId and queries the collection. The result is then validated against a Pydantic model. If the ID is invalid or if no matching document is found, the method returns None.

#### Parameters
- `id`: A string or ObjectId representing the unique identifier of the object to retrieve.

#### Return Value
- An instance of the Pydantic model if the document is found, else None.

#### Usage
Use this method to fetch a document by its ID. For example:
```python
result = crud.get("60c72b2f9b1e8b431ea7f9d8")
if result:
    print(result)
```

---

### `CRUDBase.get_by_filter`

#### Functionality
Retrieves a list of objects from the database that match the provided filter. It supports skipping a number of results and limiting the number of returned documents.

#### Parameters
- `filter`: A dictionary representing MongoDB filter criteria.
- `skip`: Number of documents to skip. Defaults to 0.
- `limit`: Maximum number of documents to retrieve. Defaults to 100.

#### Usage
- **Purpose** - Query and return matching objects from a MongoDB collection.

#### Example
```python
filter = {"status": "active"}
active_docs = crud.get_by_filter(filter, skip=0, limit=50)
```

---

### `CRUDBase.get_all`

#### Functionality
This method retrieves a set of objects from the MongoDB collection. It supports skipping a number of documents and limiting the number returned. Each document is validated using the associated Pydantic model.

#### Parameters
- `skip`: The number of documents to skip. Default is 0.
- `limit`: The maximum number of documents to retrieve. Default is 100.

#### Usage
- Purpose: Retrieve a slice of the entire collection.

#### Example
Obtain the first 50 records:
```python
objects = crud.get_all(skip=0, limit=50)
```

---

### `CRUDBase.get_by_idempotency_key`

#### Functionality
Retrieves an object from the MongoDB collection using the provided idempotency key. If no matching object is found, returns None.

#### Parameters
- `idempotency_key` (uuid.UUID): Unique idempotency key for lookup.

#### Usage
This method fetches an entry with a specified idempotency key to avoid duplicate operations.

#### Example
Assuming you have a CRUDBase instance (crud_base) and a valid UUID (key), invoke as:
```python
result = crud_base.get_by_idempotency_key(key)
```
If result is None, no matching record exists.

---

### `CRUDBase.create`

#### Functionality
The method converts a Pydantic model instance into a dictionary and inserts it into the MongoDB collection. It accepts an optional custom identifier; if none is provided, a new ObjectId is generated. Optionally, the full object is returned by re-querying the database instead of the inserted id.

#### Parameters
- `schema`: A Pydantic model instance used for object creation.
- `id`: An optional custom identifier (string or ObjectId).
- `return_obj`: Boolean flag indicating whether to return the full created object (True) or just its id (False).

#### Usage
- Purpose: Create a new document in the MongoDB collection.

#### Example
To insert a new document using a creation schema instance:
```python
new_id = crud.create(schema_instance)
```
To retrieve the full object after insertion:
```python
new_obj = crud.create(schema_instance, return_obj=True)
```

---

### `CRUDBase.update`

#### Functionality
Updates an existing object in the database. The method takes an existing object and a set of new values, converts the object's id to ObjectId, applies the new values, and updates the corresponding record. If the updated_at field is present, it is replaced with the current time.

#### Parameters
- `obj`: The object to be updated. Must have an `id` attribute.
- `values`: Optional update data. Can be a dictionary, a Pydantic model, or any object convertible to a dictionary. Fields in `values` override the object's current fields (except `id`).

#### Usage
- **Purpose**: Modify an object's record with updated data.

#### Example
```python
updated_item = crud.update(item, {"name": "New Name"})
```

---

### `CRUDBase.remove`

#### Functionality
Removes an object from the MongoDB collection using its unique ID. The method converts the provided ID to an ObjectId and attempts to delete the corresponding record. It returns True if deletion is successful, otherwise False.

#### Parameters
- `id`: The unique identifier of the object to remove. It can be a string or an ObjectId.

#### Usage
- **Purpose**: Use this method to delete objects from the database. Ensure the object exists before calling remove.

#### Example
```python
result = crud_instance.remove("605c45f8e1f9ee0021c6caf3")
if result:
    print("Deletion successful")
else:
    print("Deletion failed")
```