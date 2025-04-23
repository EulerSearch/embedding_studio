## Documentation for `MongoDao`

### Functionality
The MongoDao class acts as a bridge between MongoDB collections and Pydantic models. It converts raw BSON documents into model instances and vice versa, easing CRUD operations and data validation.

### Motivation
This class was designed to reduce repetitive boilerplate in data access layers. By abstracting BSON-to-model translation, it enforces type safety and consistency across the application.

### Inheritance
MongoDao is a generic class that inherits from Generic[ModelT]. This allows it to work with any Pydantic model, providing a flexible and type-safe interface for MongoDB operations.

### Usage
- **Purpose**: To provide an abstraction layer that maps MongoDB collections to Pydantic models.

#### Example
```python
from pymongo import MongoClient
from my_models import MyModel

client = MongoClient()
db = client.my_database
collection = db.my_collection

mongo_dao = MongoDao(collection, MyModel, "id", "mongo_id")
document = mongo_dao.bson_to_model(collection.find_one())
print(document)
```

---

## Documentation for `MongoDao._init_indexes`

### Functionality
Initializes indexes for the MongoDB collection. It creates a unique index on the model_id if it differs from model_mongo_id. It also sets up additional indexes if provided.

### Parameters
- `additional_indexes`: Optional list of dictionaries with index options.

### Usage
- **Purpose**: Configures collection indexes based on model parameters and test conditions.

#### Example
```python
dao = MongoDao(collection, model, "id", "mongo_id", additional_indexes)
```

---

## Documentation for `MongoDao.bson_to_model`

### Functionality
This method converts a BSON document from MongoDB into a model instance by passing the document to the model's validate method. If a specific Mongo ID field is configured (via model_mongo_id), the method pops the "_id" field, converts it to a string, and assigns it to that field.

### Parameters
- `bson`: A BSON document from MongoDB.

### Usage
- **Purpose**: Convert BSON documents from MongoDB into Pydantic model instances for further application use.

#### Example
```python
bson = {"_id": ObjectId("507f191e810c19729de860ea"), "name": "Alice"}
model_instance = mongo_dao.bson_to_model(bson)
```

---

## Documentation for `MongoDao.bson_to_model_opt`

### Functionality
Converts a BSON document to a model instance, handling None values. If the input BSON document is not None, it returns an instance of the model class; otherwise, it returns None.

### Parameters
- `bson`: BSON document from MongoDB or None.

### Usage
- **Purpose**: Map BSON documents to model instances with null checks.

#### Example
Suppose you have a BSON document retrieved from MongoDB:
```python
# Example BSON document
doc = {"_id": ObjectId("..."), "field": "value"}

# Convert BSON to model instance
model_instance = dao.bson_to_model_opt(doc)
```
If `doc` is None, the function returns None.

---

## Documentation for `MongoDao.bsons_to_models`

### Functionality
This method converts an iterable of BSON documents into a list of model instances. It leverages the `bson_to_model` method to convert each document individually.

### Parameters
- **bsons**: An iterable of BSON documents retrieved from MongoDB.

### Usage
- **Purpose**: To transform multiple BSON documents into Pydantic model instances.

#### Example
```python
from embedding_studio.data_access.mongo.mongo_dao import MongoDao

# Assuming MyModel is a valid Pydantic model
dao = MongoDao(collection, MyModel, 'id', model_mongo_id='mongo_id')
models = dao.bsons_to_models(bsons)
```

---

## Documentation for `MongoDao.model_to_bson`

### Functionality
Converts a model instance into a BSON document for MongoDB storage. It utilizes the model_dump method with JSON mode and exclusion of None values. If the model_mongo_id field is present and set_id is True, it sets the "_id" field to an ObjectId based on the model_mongo_id value.

### Parameters
- `obj`: Model instance to convert.
- `set_id`: Boolean flag indicating whether to assign the MongoDB '_id'.
- `model_dump_kwargs`: Additional keyword arguments for model_dump.

### Usage
- **Purpose**: Prepares a model instance for insertion or update in MongoDB.

#### Example
```python
doc = mongo_dao.model_to_bson(my_model, set_id=True)
```

---

## Documentation for `MongoDao.models_to_bsons`

### Functionality
Converts an iterable of model instances into a list of BSON documents using each model's conversion rules. This method is useful for preparing data to be inserted into a MongoDB collection.

### Parameters
- `objs`: Iterable of model instances to convert into BSON format.

### Usage
- **Purpose**: Batch conversion of model instances to BSON documents for database operations.

#### Example
Suppose you have a list of model instances called `models`:
```python
bsons = mongo_dao.models_to_bsons(models)
```
This will return a list of BSON documents ready to be used in MongoDB operations.

---

## Documentation for `MongoDao.get_schema_properties`

### Functionality
Retrieves the property names defined in the model's schema. This method calls the model's schema() function and extracts the keys from the "properties" field of the schema.

### Parameters
- None

### Usage
- **Purpose**: Determine the fields defined in a model based on its schema.

#### Example
If a model's schema is defined as:
```json
{
  "properties": {
    "name": { ... },
    "age": { ... }
  }
}
```
Then this method returns a set like: `{'name', 'age'}`.

---

## Documentation for `MongoDao.get_model_projection`

### Functionality
Creates a MongoDB projection document that filters returned fields according to the model schema. It builds a dictionary where each key is a property from the model and the value is a boolean indicating its inclusion. If a special model identifier (model_mongo_id) is defined, it ensures that the _id field is handled appropriately.

### Parameters
(None)

### Usage
- **Purpose**: Used to generate a projection for MongoDB queries so that the query returns only the fields defined in the model schema.

#### Example
Given a model with properties such as "name" and "age", and if model_mongo_id is set, the produced projection might look like:
```python
{ "name": True, "age": True, "_id": True }
```

---

## Documentation for `MongoDao.model_id_to_db_id`

### Functionality
This method converts a model ID into a database ID for MongoDB. If model_id and model_mongo_id are equal, it converts the value into an ObjectId. Otherwise, the original value is returned.

### Parameters
- `obj_id`: The model ID to be converted. It represents the unique identifier in the model.

### Usage
- **Purpose**: To map a model identifier into the corresponding MongoDB ID field, ensuring correct conversion when needed.

#### Example
For a model with matching model_id and model_mongo_id, calling:
```python
id_field, db_id = model_id_to_db_id("507f1f77bcf86cd799439011")
```
Returns a tuple where id_field is "_id" and db_id is the ObjectId of the provided ID.

---

## Documentation for `MongoDao.get_db_id`

### Functionality
This method retrieves the database identifier from a model instance by extracting the field specified in the model_id and converting it using model_id_to_db_id. It returns a tuple with the identifier field name and the formatted id value for MongoDB queries.

### Parameters
- `obj`: The model instance from which to obtain the database id.

### Usage
- **Purpose**: To generate the appropriate identifier tuple for use in MongoDB operations.

#### Example
Assuming a model instance `user` with an attribute that matches `model_id`, the call:
```python
id_field, id_value = dao.get_db_id(user)
```
Returns the identifier tuple for database queries.

---

## Documentation for `MongoDao.find_one`

### Functionality
Finds a single document by its ID or a custom filter. If an ID is provided, the method updates the filter to match the ID and queries the MongoDB collection. The result is converted to its corresponding model instance or returns None if not found.

### Parameters
- `obj_id`: Optional; the identifier to search for in the collection.
- `**kwargs`: Additional parameters for MongoDB's find_one operation (e.g., custom filters, projection).

### Usage
- **Purpose**: Retrieve a document and convert it to a model instance using MongoDB's find_one functionality.

#### Example
Assuming a model for user data and corresponding DAO:
```python
# Retrieve by ID
user = dao.find_one("user_id_value")

# Retrieve using a custom filter
user = dao.find_one(filter={"email": "test@example.com"})
```

---

## Documentation for `MongoDao.find`

### Functionality
Search for documents that match a given filter in a MongoDB collection. It returns a list of model instances. Sorting and limiting results is supported through additional parameters.

### Parameters
- `sort_args`: Optional sorting arguments for the query.
- `find_kwargs`: Additional keyword arguments for the find method. Common keys include:
  - `filter`: A dictionary specifying query conditions.
  - `limit`: Maximum number of documents to return.

### Usage
- **Purpose**: Retrieve model instances that satisfy a given query on a MongoDB collection.

#### Example
```python
results = dao.find(sort_args=(-1,), filter={'age': {'$gt': 25}}, limit=10)
```

---

## Documentation for `MongoDao.insert_one`

### Functionality
Inserts a model instance into the MongoDB collection as a BSON document. It converts the model using `model_to_bson` and calls MongoDB's `insert_one` to store the document. The result is a `pymongo.results.InsertOneResult` indicating the outcome.

### Parameters
- `obj`: Model instance to insert into the collection.
- `**kwargs`: Extra arguments for MongoDB's `insert_one` method.

### Usage
- **Purpose**: Add a new document to the collection.

#### Example
```python
# Insert a model instance into the database
result = mongo_dao.insert_one(model_instance)
```

---

## Documentation for `MongoDao.insert_many`

### Functionality
Inserts multiple model instances as documents into the MongoDB collection. This method converts model instances to BSON documents and calls PyMongo's insert_many method to perform a bulk insert.

### Parameters
- `objs`: Iterable of model instances to insert.
- `**kwargs`: Additional keyword arguments for PyMongo's insert_many method.

### Usage
Use this method when you need to insert several documents at once, which can reduce the number of database calls and improve efficiency.

#### Example
```python
result = mongo_dao.insert_many(models)
print(result.inserted_ids)
```

---

## Documentation for `MongoDao.update_one`

### Functionality
Updates a single document in the MongoDB collection. If an object is provided, it updates the document matching the object's ID with the object's data. Otherwise, it uses the provided filter and update parameters.

### Parameters
- `obj`: Optional model instance with updated data.
- `**kwargs`: Additional parameters for MongoDB's update_one.

### Usage
- **Purpose**: Update a document in the collection.

#### Example
Using an object:
```python
result = dao.update_one(obj=model_instance)
```
Using filter and update parameters:
```python
result = dao.update_one(
    filter={"name": "John"},
    update={"$set": {"age": 30}}
)
```

---

## Documentation for `MongoDao.upsert_one`

### Functionality
This method updates a single document in the MongoDB collection. If a document matching the filter does not exist, it inserts a new one. It effectively wraps the update_one method with the upsert flag set to True.

### Parameters
- `obj`: Optional model instance containing the data to be upserted.
- `**kwargs`: Additional keyword arguments passed to MongoDB's update_one method (e.g., filter conditions).

### Usage
Use this method when you want to update a document if it exists, or insert it if it does not. It simplifies common upsert operations.

#### Example
For example, to upsert a user document, you might use:
```python
result = mongo_dao.upsert_one(user, filter={'email': user.email})
```
This updates an existing user with the given email or inserts a new user if one does not already exist.

---

## Documentation for `MongoDao.find_one_and_update`

### Functionality
Find a single document in the collection and update it. If a valid id is provided, the method constructs a filter using the model's id field, performs an update operation, and returns the updated model instance if found. Otherwise, returns None.

### Parameters
- `obj_id`: Optional. The ID value used for searching the document.
- `**kwargs`: Additional keyword arguments passed to MongoDB's find_one_and_update method. This may include update operations, filters, and other options.

### Usage
- **Purpose**: To update a document and retrieve the result in a single atomic operation using MongoDB's find_one_and_update.

#### Example
Suppose you want to update a user's email by their id:
```python
updated_user = mongo_dao.find_one_and_update(
    obj_id=user_id,
    update={'$set': {'email': 'new.email@example.com'}}
)
```

---

## Documentation for `MongoDao.delete_one`

### Functionality
Deletes a single document from a MongoDB collection. If an object ID is provided, it is converted using the model's ID conversion to create a MongoDB filter. The method then calls MongoDB's underlying delete_one function with any additional parameters.

### Parameters
- `obj_id`: The identifier of the document to be deleted. This ID is converted to the proper database format.
- `kwargs`: Additional keyword arguments passed to MongoDB's delete_one method. These may include options such as write concern.

### Usage
- **Purpose**: Remove a document from the collection based on its ID.

#### Example
```python
result = mongo_dao.delete_one("document_id")
print(result.deleted_count)
```

---

## Documentation for `MongoDao.find_one_and_delete`

### Functionality
Deletes a single document from a MongoDB collection by its identifier. The method searches for a document matching the provided ID and removes it, returning the deleted model instance if found.

### Parameters
- `obj_id`: The unique identifier used to find the document.
- `kwargs`: Additional arguments passed to MongoDB's find_one_and_delete operation (e.g., output projection).

### Usage
Call this method to delete a document and obtain its model representation. It is useful when you want to both remove a record and inspect the removed data.

#### Example
Suppose you have an instance with an ID of "123"; you can delete it as follows:
```python
result = dao.find_one_and_delete("123")
```
The returned result is the model instance of the deleted document if it exists, or `None` otherwise.