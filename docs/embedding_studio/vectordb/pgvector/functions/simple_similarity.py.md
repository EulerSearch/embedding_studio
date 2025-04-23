### Documentation for Vector Search Functions

This document provides an overview of several methods for generating PostgreSQL functions that facilitate vector search operations in a database. Each function is tailored for specific needs within the context of vector searches.

---

#### 1. `generate_simple_vector_search_function`

**Functionality**  
Generates a PostgreSQL function that performs a simple vector search. The generated SQL function retrieves vectors from a database using a specified distance metric. It is designed for efficient retrieval when only basic sorting and vector inclusion are needed.

**Parameters**  
- `model_id`: A string representing the embedding model ID. It is used to generate table names and a unique function identifier.
- `metric_type`: Optional metric type from MetricType (COSINE, EUCLID, or DOT). Defaults to MetricType.COSINE.

**Returns**  
A SQL string that creates a PostgreSQL function for vector search.

**Usage**  
Use this function when a fast, simple vector search is required without advanced filtering or user-specific customizations.

**Example**  
```python
sql = generate_simple_vector_search_function("abc", MetricType.COSINE)
execute_sql(sql)
```

---

#### 2. `generate_simple_vector_search_no_vectors_function`

**Functionality**  
Generates a SQL function for simple vector search optimized for cases where vector values are excluded from the result set. This reduces data transfer and memory usage while preserving core search features.

**Parameters**  
- `model_id`: ID of the embedding model, used to generate table names.
- `metric_type`: Optional. Metric for vector distance (COSINE, EUCLID, DOT). Defaults to `MetricType.COSINE`.

**Usage**  
Create a PostgreSQL function to perform a vector search that omits the actual vectors in the output.

**Example**  
```python
sql = generate_simple_vector_search_no_vectors_function('example_model', MetricType.COSINE)
```

---

#### 3. `generate_simple_vector_search_similarity_ordered_function`

**Functionality**  
Generates SQL code for creating a PostgreSQL function that performs a simple vector search with vectors in the results. This function is ideal when no advanced filtering is required.

**Parameters**  
- `model_id`: A string identifier for the embedding model, used to name tables and functions.
- `metric_type`: An optional metric type (COSINE, EUCLID, DOT) that determines the distance operator. Defaults to COSINE.

**Returns**  
A SQL string that creates a PostgreSQL function for performing the search.

**Raises**  
- `ValueError`: If an unsupported metric type is provided.

**Usage**  
Provide a model ID and optionally a metric type to generate the SQL for a simple vector search function. The output SQL can be used to create a PostgreSQL function.

**Example**  
```python
sql_str = generate_simple_vector_search_similarity_ordered_function("example_model")
print(sql_str)
```

---

#### 4. `generate_simple_vector_search_similarity_ordered_no_vectors_function`

**Functionality**  
Creates a PostgreSQL function that performs a similarity-ordered search without including vector values. It returns similarity scores and object metadata, making it ideal for pure ranking and high-performance queries where vector data is not needed.

**Parameters**  
- `model_id`: Embedding model identifier used for table name generation.
- `metric_type`: Vector distance metric type. Options are COSINE, EUCLID, and DOT.

**Usage**  
Generate the SQL function for similarity-based search when only ranking and metadata are needed. This function creates a PostgreSQL function that can be executed for efficient similarity queries.

**Example**  
```python
sql = generate_simple_vector_search_similarity_ordered_no_vectors_function(
    model_id="my_model", metric_type=MetricType.COSINE)
print(sql)
```