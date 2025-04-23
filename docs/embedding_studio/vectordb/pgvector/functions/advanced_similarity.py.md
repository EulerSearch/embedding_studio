# Documentation for Advanced Vector Search Functions in PostgreSQL

This documentation covers three methods related to generating PostgreSQL SQL functions for advanced vector search: `generate_advanced_vector_search_function`, `generate_advanced_vector_search_no_vectors_function`, and `generate_advanced_vector_search_similarity_ordered_function`. Each method enables different functionalities tailored to specific requirements of vector search operations.

## 1. Function: `generate_advanced_vector_search_function`

### Functionality
This function generates a PostgreSQL SQL function for advanced vector search. It filters objects based on payload and user ID, finds and scores vectors, and groups results by objects along with vector information.

### Parameters
- `model_id`: ID of the embedding model used in table name generation.
- `metric_type`: Distance metric for vector search. Options are COSINE, EUCLID, and DOT. Defaults to COSINE.

### Usage
- **Purpose:** Generate SQL that creates a PostgreSQL function for performing vector search with advanced filtering, scoring, and sorting features.

#### Example
```python
sql = generate_advanced_vector_search_function("my_model", metric_type=MetricType.COSINE)
# Execute the SQL in PostgreSQL to create the search function.
```

---

## 2. Function: `generate_advanced_vector_search_no_vectors_function`

### Functionality
This function generates a PostgreSQL SQL string for advanced vector search without including vector values in the results. It is useful in scenarios where only the distance and other related information are required, thus reducing data transfer and memory usage.

### Parameters
- `model_id`: A unique identifier for the embedding model, used to customize the SQL function name and target tables.
- `metric_type`: Optional distance metric type. Supported values are COSINE, EUCLID, and DOT. If an unsupported metric is provided, the function raises a ValueError.

### Usage
The returned SQL string can be executed on a PostgreSQL database to create a function that performs advanced vector search with filtering by payload and user. It handles sorting and grouping of results, and does not return actual vector values, only their computed distances.

#### Example
```python
sql = generate_advanced_vector_search_no_vectors_function("myModel", MetricType.COSINE)
execute_sql(sql)  # Execute the generated SQL in PostgreSQL
```

---

## 3. Function: `generate_advanced_vector_search_similarity_ordered_function`

### Functionality
This function generates SQL code to create a PostgreSQL function for advanced vector searches. The SQL function filters data based on payloads, user IDs, and sorts by specific columns while returning vectors and relevant data.

### Parameters
- `model_id`: A string denoting the embedding model ID, used for generating table names dynamically.
- `metric_type`: An optional parameter for the vector distance metric. Accepted values are COSINE, EUCLID, and DOT. Defaults to COSINE.

### Returns
A SQL string that creates a PostgreSQL function to perform advanced vector searches.

### Usage
- **Purpose:** To dynamically build SQL for complex vector searches with filters and sorting capabilities.

#### Example
Given a model ID "sample_model" with the default cosine metric:
```python
sql = generate_advanced_vector_search_similarity_ordered_function("sample_model")
-- Execute the SQL in PostgreSQL to create the search function.
``` 

This integration of functionality across the three methods allows users to perform advanced vector searches with varying degrees of detail and filtering capabilities in PostgreSQL.