## Documentation for `SQLFileMeta`

### Functionality

The SQLFileMeta class provides metadata for SQL database rows by extending the ItemMeta class. It assigns a unique identifier for each row, ensuring that database records can be uniquely managed and queried.

### Inheritance

SQLFileMeta extends the ItemMeta class to include SQL-specific functionality for handling database row meta information.

### Purpose and Motivation

The main purpose of this class is to facilitate the tracking and identification of SQL table rows by deriving a unique database row identifier. This design supports efficient metadata management in SQL databases.

### Attributes

- **object_id**: Optional identifier for the SQL table row.
- **derived_id**: A computed property that returns a unique identifier prefixed with 'row:' based on the object_id.

### Method: `derived_id`

#### Functionality

Generates a unique row identifier by prefixing the `object_id` with "row:". If `object_id` is not set, a ValueError is raised to signal the absence of a valid identifier.

#### Parameters

This property takes no parameters.

#### Usage

- **Purpose**: Uniquely identify rows in SQL database tables.

#### Example

Assume an instance of SQLFileMeta with `object_id` set to "123". Accessing `derived_id` will return "row:123".