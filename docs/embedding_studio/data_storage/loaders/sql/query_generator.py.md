# Documentation for `AbstractQueryGenerator`

## Functionality

This abstract base class defines the interface for SQL query generators. It provides a blueprint for generating SQL queries to interact with a database table. Abstract methods include fetching a single row, multiple rows, paginated rows, and counting table rows.

## Motivation

The design of this class enforces a consistent interface for query generation. This common contract promotes maintainability and reusability of the code, ensuring that all concrete implementations adhere to a standard structure.

## Inheritance

AbstractQueryGenerator inherits from Python's ABC (Abstract Base Class). This establishes a contract requiring all subclasses to implement the abstract methods defined herein.

## Usage

- **Purpose:** Generate SQLAlchemy Select queries for various database operations.

### Example Implementation

A concrete class must implement all abstract methods:

```python
class QueryGenerator(AbstractQueryGenerator):
    def fetch_all(self, row_ids: List[str]) -> Select:
        # implementation

    def one(self, row_id: str) -> Select:
        # implementation

    def all(self, offset: int, batch_size: int) -> Select:
        # implementation

    def count(self) -> Select:
        # implementation
```

---

# Documentation for `QueryGenerator`

## Functionality

This class generates SQLAlchemy Select queries for common database operations on a specified table. It simplifies query construction by abstracting SQL query formation.

## Parameters

- `table_name`: Name of the database table to query.
- `engine`: SQLAlchemy Engine instance providing the database connection.

## Usage

- **Purpose**: Automate and standardize SQL query generation.
- **Motivation**: Reduce repetitive code and centralize query logic, enhancing consistency and maintainability.

## Inheritance

Inherits from `AbstractQueryGenerator`, an abstract base class that defines the interface for SQL query generators.

### Example

A sample usage:

```python
from sqlalchemy import create_engine
engine = create_engine("database_url")
qg = QueryGenerator("my_table", engine)
query = qg.fetch_all(['id1', 'id2'])
```

---

## Documentation for `QueryGenerator.fetch_all`

### Functionality

Generates a SELECT query that fetches multiple rows from a table based on a list of row IDs.

### Parameters

- `row_ids`: List of row identifiers used to filter rows.

### Usage

- **Purpose**: Build a SQLAlchemy query that filters rows by ID.

#### Example

```python
table = Table('my_table', metadata, autoload_with=engine)
query = select(table).where(table.c.id.in_(['id1', 'id2']))
```

---

## Documentation for `QueryGenerator.one`

### Functionality

Generates a SELECT query to fetch a single row by its ID. It creates a query that selects a row whose ID column equals the provided row ID.

### Parameters

- `row_id`: Identifier of the row to fetch.

### Usage

- **Purpose**: To retrieve a single row from a table based on its unique identifier.

#### Example

```python
table = Table('my_table', metadata, autoload_with=engine)
query = select(table).where(table.c.id == row_id)
```

---

## Documentation for `QueryGenerator.all`

### Functionality

Generate a SQLAlchemy Select query that fetches a batch of rows with pagination. The query orders rows by the id column and applies a limit and offset based on the provided arguments.

### Parameters

- `offset`: Number of rows to skip before starting the batch.
- `batch_size`: Maximum number of rows to return in the batch.

### Usage

- **Purpose**: Retrieve a subset of rows from a table using pagination.

#### Example

```python
generator = QueryGenerator("users", engine)
query = generator.all(offset=10, batch_size=5)
result = engine.execute(query)
```

---

## Documentation for `QueryGenerator.count`

### Functionality

Generates a SQLAlchemy SELECT query to count all rows in the specified table. This method creates a query that returns the total number of rows in the table.

### Parameters

None.

### Usage

- **Purpose**: To obtain the row count of a database table.

#### Example

Example usage:

```python
generator = QueryGenerator("users", engine)
count_query = generator.count()
```

The `count_query` can then be executed using the engine.
  
---

## Documentation for `AbstractQueryGenerator.fetch_all`

### Functionality

Generates a SELECT query to fetch multiple rows by their IDs. This method returns a SQLAlchemy Select object that represents the query.

### Parameters

- `row_ids`: List of row identifier strings to fetch.

### Usage

- **Purpose**: Retrieve multiple rows from a database table based on their IDs.

#### Example

```python
table = Table('my_table', self.metadata, autoload_with=self.engine)
return select(table).where(table.c.id.in_(row_ids))
```

---

## Documentation for `AbstractQueryGenerator.one`

### Functionality

This method generates a SQLAlchemy Select query that fetches a single row from a database table based on a given identifier. It provides an abstract interface, requiring a concrete subclass implementation.

### Parameters

- `row_id`: Identifier of the row to fetch, typically a primary key.

### Returns

- A SQLAlchemy Select object representing the query to retrieve a single row from the table.

### Usage

- **Purpose**: Define the interface for fetching a single row and serve as a blueprint for implementations in subclasses.

#### Example Implementation

```python
def one(self, row_id: str) -> Select:
    table = Table('my_table', self.metadata, autoload_with=self.engine)
    return select(table).where(table.c.id == row_id)
```

---

## Documentation for `AbstractQueryGenerator.all`

### Functionality

Generates an SQL query to fetch a batch of rows with pagination. The query applies an offset and a limit to retrieve a subset of rows from a database table.

### Parameters

- `offset`: Number of rows to skip.
- `batch_size`: Maximum number of rows to return.

### Usage

- **Purpose**: Retrieve a specific page of rows from a table using pagination.

#### Example

```python
table = Table('my_table', self.metadata, autoload_with=self.engine)
query = select(table).order_by(table.c.id)
query = query.limit(batch_size).offset(offset)
```

---

## Documentation for `AbstractQueryGenerator.count`

### Functionality

Generates a SQLAlchemy Select query that counts all rows in a table. This abstract method is to be implemented by subclasses to create a query returning the number of rows.

### Parameters

This method does not accept any parameters.

### Usage

- **Purpose** - To create a count query for the specified table.

#### Example

An example implementation:

```python
table = Table('my_table', self.metadata, autoload_with=self.engine)
return select(func.count()).select_from(table)
```