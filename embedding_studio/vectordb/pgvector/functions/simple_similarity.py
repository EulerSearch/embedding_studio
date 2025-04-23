from typing import Optional

from embedding_studio.models.embeddings.models import MetricType


def generate_simple_vector_search_function(
    model_id: str, metric_type: Optional[MetricType] = MetricType.COSINE
) -> str:
    """
    Generate a PostgreSQL function for simple vector search with vectors in results.

    This function creates SQL for a simpler vector search without user filtering or
    advanced payload filtering. It can sort by payload fields or table columns and
    includes vectors in the results.

    The simple search is more efficient when:
    1. No user-specific customization is needed
    2. No complex payload filtering is required
    3. Basic sorting and vector retrieval is sufficient

    :param model_id: ID of the embedding model, used for table name generation
    :param metric_type: Vector distance metric type (COSINE, EUCLID, DOT)
    :return: SQL string that creates a PostgreSQL function for simple vector search
    :raises ValueError: If metric_type is not supported
    """
    operator_map = {
        MetricType.COSINE: "<=>",
        MetricType.EUCLID: "<->",
        MetricType.DOT: "<#>",
    }

    if metric_type not in operator_map:
        raise ValueError(f"Unsupported metric type: {metric_type}")

    distance_operator = operator_map[metric_type]
    metric_name = metric_type.value
    function_name = f"simple_v_{model_id}_{metric_name}"

    dbo_table = f"dbo_{model_id}"
    dbop_table = f"dbop_{model_id}"

    sql_function = f"""
CREATE OR REPLACE FUNCTION {function_name}(
    input_vector             vector,
    limit_results            INT          DEFAULT 10,
    offset_value             INT          DEFAULT 0,
    max_distance             FLOAT        DEFAULT NULL,
    sort_field               TEXT         DEFAULT NULL,
    sort_order               TEXT         DEFAULT 'asc',
    is_payload               BOOLEAN      DEFAULT FALSE,
    enlarged_limit           INT          DEFAULT 50,
    enlarged_offset          INT          DEFAULT 0,
    average_only             BOOLEAN      DEFAULT FALSE
)
RETURNS TABLE (
    result_object_id     VARCHAR(128),
    result_payload       JSONB,
    result_storage_meta  JSONB,
    result_user_id       VARCHAR(128),
    result_original_id   VARCHAR(128),
    result_part_ids      VARCHAR(128)[],
    result_vectors       vector[],
    result_distance      FLOAT,
    subset_count         INT
) AS $$
DECLARE
    order_clause         TEXT := '';
    query                TEXT;
BEGIN
    IF is_payload THEN
        IF sort_field IS NOT NULL THEN
            IF lower(sort_order) != 'desc' THEN
                order_clause := format('ORDER BY o.payload->%L ASC', sort_field);
            ELSE
                order_clause := format('ORDER BY o.payload->%L DESC', sort_field);
            END IF;
        END IF;
    ELSE
        IF sort_field IS NOT NULL THEN
            IF lower(sort_order) != 'desc' THEN
                order_clause := format('ORDER BY %I ASC', sort_field);
            ELSE
                order_clause := format('ORDER BY %I DESC', sort_field);
            END IF;
        END IF;
    END IF;

    query := format('
WITH filtered_objects AS (
    SELECT 
        object_id, 
        payload, 
        storage_meta, 
        user_id,  
        o.original_id, 
        row_number() over() AS rn1
    FROM {dbo_table} o
    WHERE (user_id IS NULL) %s
    limit $8
    offset $9
), prefiltered_vectors AS (
    SELECT
        op.object_id,
        op.part_id,
        op.vector,
        o.payload, 
        o.storage_meta, 
        o.user_id, 
        o.original_id, 
        o.rn1,
        row_number() over() AS rn,
        (op.vector {distance_operator} $1) AS distance
    FROM {dbop_table} op
    INNER JOIN filtered_objects o ON op.object_id = o.object_id 
    WHERE (NOT $10 OR is_average = TRUE) AND ($4 IS NULL OR op.vector {distance_operator} $1 <= $4)
), total_count AS (
    SELECT cast(count(distinct(object_id)) as int) AS total_filtered_objects_count
    FROM prefiltered_vectors
) SELECT 
	object_id AS result_object_id, 	
	payload AS result_payload, 
	storage_meta AS result_storage_meta, 
	user_id AS result_user_id, 
	original_id AS result_original_id, 
	ARRAY_AGG(part_id ORDER BY distance) AS result_part_ids,
    ARRAY_AGG(vector ORDER BY distance) AS result_vectors,
    MIN(distance) AS result_distance,
    tc.total_filtered_objects_count AS subset_count 
  FROM prefiltered_vectors 
  CROSS JOIN total_count tc
  GROUP BY 
    object_id, 
    payload, 
    storage_meta, 
    user_id, 
    original_id, 
    tc.total_filtered_objects_count, 
    prefiltered_vectors.rn1
  ORDER BY rn1 ASC
  LIMIT $2 OFFSET $3;', order_clause);

    RETURN QUERY EXECUTE query
    USING input_vector, limit_results, offset_value, max_distance, sort_field, sort_order, is_payload, enlarged_limit, enlarged_offset, average_only;
END;
$$ LANGUAGE plpgsql;    
"""
    return sql_function


def generate_simple_vector_search_no_vectors_function(
    model_id: str, metric_type: Optional[MetricType] = MetricType.COSINE
) -> str:
    """
    Generate a PostgreSQL function for simple vector search without vectors in results.

    Similar to generate_simple_vector_search_function but optimized for cases where
    the actual vector values are not needed in the results. This reduces data transfer
    and memory usage while maintaining search functionality.

    :param model_id: ID of the embedding model, used for table name generation
    :param metric_type: Vector distance metric type (COSINE, EUCLID, DOT)
    :return: SQL string that creates a PostgreSQL function for simple vector search
    :raises ValueError: If metric_type is not supported
    """
    operator_map = {
        MetricType.COSINE: "<=>",
        MetricType.EUCLID: "<->",
        MetricType.DOT: "<#>",
    }

    if metric_type not in operator_map:
        raise ValueError(f"Unsupported metric type: {metric_type}")

    distance_operator = operator_map[metric_type]
    metric_name = metric_type.value
    function_name = f"simple_{model_id}_{metric_name}"

    dbo_table = f"dbo_{model_id}"
    dbop_table = f"dbop_{model_id}"

    sql_function = f"""
CREATE OR REPLACE FUNCTION {function_name}(
    input_vector             vector,
    limit_results            INT          DEFAULT 10,
    offset_value             INT          DEFAULT 0,
    max_distance             FLOAT        DEFAULT NULL,
    sort_field               TEXT         DEFAULT NULL,
    sort_order               TEXT         DEFAULT 'asc',
    is_payload               BOOLEAN      DEFAULT FALSE,
    enlarged_limit           INT          DEFAULT 50,
    enlarged_offset          INT          DEFAULT 0,
    average_only             BOOLEAN      DEFAULT FALSE
)
RETURNS TABLE (
    result_object_id     VARCHAR(128),
    result_payload       JSONB,
    result_storage_meta  JSONB,
    result_user_id       VARCHAR(128),
    result_original_id   VARCHAR(128),
    result_part_ids      VARCHAR(128)[],
    result_distance      FLOAT,
    subset_count         INT
) AS $$
DECLARE
    order_clause         TEXT := '';
    query                TEXT;
BEGIN
    IF is_payload THEN
        IF sort_field IS NOT NULL THEN
            IF lower(sort_order) != 'desc' THEN
                order_clause := format('ORDER BY o.payload->%L ASC', sort_field);
            ELSE
                order_clause := format('ORDER BY o.payload->%L DESC', sort_field);
            END IF;
        END IF;
    ELSE
        IF sort_field IS NOT NULL THEN
            IF lower(sort_order) != 'desc' THEN
                order_clause := format('ORDER BY %I ASC', sort_field);
            ELSE
                order_clause := format('ORDER BY %I DESC', sort_field);
            END IF;
        END IF;
    END IF;

    query := format('
WITH filtered_objects AS (
    SELECT 
        object_id, 
        payload, 
        storage_meta, 
        user_id,  
        o.original_id, 
        row_number() over() AS rn1
    FROM {dbo_table} o
    WHERE (user_id IS NULL) %s
    limit $8
    offset $9
), prefiltered_vectors AS (
    SELECT
        op.object_id,
        op.part_id,
        o.payload, 
        o.storage_meta, 
        o.user_id, 
        o.original_id, 
        o.rn1,
        row_number() over() AS rn,
        (op.vector {distance_operator} $1) AS distance
    FROM {dbop_table} op
    INNER JOIN filtered_objects o ON op.object_id = o.object_id 
    WHERE (NOT $10 OR is_average = TRUE) AND ($4 IS NULL OR op.vector {distance_operator} $1 <= $4)
), total_count AS (
    SELECT cast(count(distinct(object_id)) as int) AS total_filtered_objects_count
    FROM prefiltered_vectors
) SELECT 
	object_id AS result_object_id, 	
	payload AS result_payload, 
	storage_meta AS result_storage_meta, 
	user_id AS result_user_id, 
	original_id AS result_original_id, 
	ARRAY_AGG(part_id) AS result_part_ids,
    MIN(distance) AS result_distance,
    tc.total_filtered_objects_count AS subset_count 
  FROM prefiltered_vectors 
  CROSS JOIN total_count tc
  GROUP BY 
    object_id, 
    payload, 
    storage_meta, 
    user_id, 
    original_id, 
    tc.total_filtered_objects_count, 
    prefiltered_vectors.rn1
  ORDER BY rn1 ASC
  LIMIT $2 OFFSET $3;', order_clause);

    RETURN QUERY EXECUTE query
    USING input_vector, limit_results, offset_value, max_distance, sort_field, sort_order, is_payload, enlarged_limit, enlarged_offset, average_only;
END;
$$ LANGUAGE plpgsql;    
"""
    return sql_function


def generate_simple_vector_search_similarity_ordered_function(
    model_id: str, metric_type: Optional[MetricType] = MetricType.COSINE
) -> str:
    """
    Generate a PostgreSQL function for similarity-ordered simple search with vectors.

    Creates SQL for a simple vector search that prioritizes similarity ordering.
    The function finds vectors most similar to the query vector first, without
    complex filtering. Includes vectors in the results.

    This function is optimized for:
    1. Pure similarity-based searches without complex filtering
    2. Cases where vectors need to be included in results
    3. Finding the most similar vectors in the most efficient way

    :param model_id: ID of the embedding model, used for table name generation
    :param metric_type: Vector distance metric type (COSINE, EUCLID, DOT)
    :return: SQL string that creates a PostgreSQL function for similarity-ordered search
    :raises ValueError: If metric_type is not supported
    """
    operator_map = {
        MetricType.COSINE: "<=>",
        MetricType.EUCLID: "<->",
        MetricType.DOT: "<#>",
    }

    if metric_type not in operator_map:
        raise ValueError(f"Unsupported metric type: {metric_type}")

    distance_operator = operator_map[metric_type]
    metric_name = metric_type.value
    function_name = f"simple_v_so_{model_id}_{metric_name}"

    dbo_table = f"dbo_{model_id}"
    dbop_table = f"dbop_{model_id}"

    sql_function = f"""
CREATE OR REPLACE FUNCTION {function_name}(
    input_vector             vector,
    limit_results            INT          DEFAULT 10,
    offset_value             INT          DEFAULT 0,
    max_distance             FLOAT        DEFAULT NULL,
    enlarged_limit           INT          DEFAULT 50,
    enlarged_offset          INT          DEFAULT 0,
    average_only             BOOLEAN      DEFAULT FALSE
)
RETURNS TABLE (
    result_object_id     VARCHAR(128),
    result_payload       JSONB,
    result_storage_meta  JSONB,
    result_user_id       VARCHAR(128),
    result_original_id   VARCHAR(128),
    result_part_ids      VARCHAR(128)[],
    result_vectors       vector[],
    result_distance      FLOAT,
    subset_count         INT
) AS $$
DECLARE
    query TEXT;
BEGIN
    query := format('
WITH filtered_vectors AS materialized (
    SELECT
        op.object_id,
        op.part_id,
        op.vector,
        (op.vector {distance_operator} $1) AS distance  
    FROM {dbop_table} op
    WHERE (NOT $7 OR op.is_average = TRUE) AND (op.user_id IS NULL) -- AND (op.vector {distance_operator} $1 <= $4)
	ORDER BY distance ASC
	LIMIT $5
	OFFSET $6
), total_count AS (
    SELECT cast(count(distinct(object_id)) as int) AS total_filtered_objects_count
    FROM filtered_vectors v WHERE v.distance <= $4
) SELECT
	v.object_id as result_object_id,
	o.payload as result_payload, 
	o.storage_meta as result_storage_meta, 
	o.user_id as result_user_id, 
	o.original_id as result_original_id, 
	ARRAY_AGG(part_id ORDER BY distance) AS result_part_ids,
    ARRAY_AGG(vector ORDER BY distance) AS result_vectors,
    MIN(v.distance) AS result_distance,
    tc.total_filtered_objects_count AS subset_count
FROM filtered_vectors v
INNER JOIN {dbo_table} o ON o.object_id = v.object_id
CROSS JOIN total_count tc
WHERE v.distance <= $4
GROUP BY 
    v.object_id,
    o.payload, 
    o.storage_meta, 
    o.user_id, 
    o.original_id, 
    tc.total_filtered_objects_count,
    v.distance
ORDER BY result_distance ASC
LIMIT $2 OFFSET $3;');
    RETURN QUERY EXECUTE query
    USING input_vector, limit_results, offset_value, max_distance, enlarged_limit, enlarged_offset, average_only;
END;
$$ LANGUAGE plpgsql;    
"""
    return sql_function


def generate_simple_vector_search_similarity_ordered_no_vectors_function(
    model_id: str, metric_type: Optional[MetricType] = MetricType.COSINE
) -> str:
    """
    Generate a PostgreSQL function for similarity-ordered simple search without vectors.

    Similar to generate_simple_vector_search_similarity_ordered_function but doesn't
    include vector values in the results. This is the most efficient search when only
    similarity scores and object metadata are needed.

    This function is ideal for:
    1. Pure similarity ranking scenarios
    2. Maximum performance where vector values aren't needed
    3. Simple "find most similar items" use cases

    :param model_id: ID of the embedding model, used for table name generation
    :param metric_type: Vector distance metric type (COSINE, EUCLID, DOT)
    :return: SQL string that creates a PostgreSQL function for similarity-ordered search
    :raises ValueError: If metric_type is not supported
    """
    operator_map = {
        MetricType.COSINE: "<=>",
        MetricType.EUCLID: "<->",
        MetricType.DOT: "<#>",
    }

    if metric_type not in operator_map:
        raise ValueError(f"Unsupported metric type: {metric_type}")

    distance_operator = operator_map[metric_type]
    metric_name = metric_type.value
    function_name = f"simple_so_{model_id}_{metric_name}"

    dbo_table = f"dbo_{model_id}"
    dbop_table = f"dbop_{model_id}"

    sql_function = f"""
CREATE OR REPLACE FUNCTION {function_name}(
    input_vector             vector,
    limit_results            INT          DEFAULT 10,
    offset_value             INT          DEFAULT 0,
    max_distance             FLOAT        DEFAULT NULL,
    enlarged_limit           INT          DEFAULT 50,
    enlarged_offset          INT          DEFAULT 0,
    average_only             BOOLEAN      DEFAULT FALSE
)
RETURNS TABLE (
    result_object_id     VARCHAR(128),
    result_payload       JSONB,
    result_storage_meta  JSONB,
    result_user_id       VARCHAR(128),
    result_original_id   VARCHAR(128),
    result_part_ids      VARCHAR(128)[],
    result_distance      FLOAT,
    subset_count         INT
) AS $$
DECLARE
    query TEXT;
BEGIN
    query := format('
WITH filtered_vectors AS materialized (
    SELECT
        op.object_id,
        op.part_id,
        (op.vector {distance_operator} $1) AS distance
    FROM {dbop_table} op
    WHERE (NOT $7 OR op.is_average = TRUE) AND (op.user_id IS NULL)
	ORDER BY distance ASC
	LIMIT $5
	OFFSET $6
), total_count AS (
    SELECT cast(count(distinct(object_id)) as int) AS total_filtered_objects_count
    FROM filtered_vectors v WHERE v.distance <= $4
) SELECT
	v.object_id as result_object_id,
	o.payload as result_payload, 
	o.storage_meta as result_storage_meta, 
	o.user_id as result_user_id, 
	o.original_id as result_original_id, 
	ARRAY_AGG(v.part_id) AS result_part_ids,
    MIN(v.distance) AS result_distance,
    tc.total_filtered_objects_count AS subset_count
FROM filtered_vectors v
INNER JOIN {dbo_table} o ON o.object_id = v.object_id
CROSS JOIN total_count tc
WHERE v.distance <= $4
GROUP BY 
    v.object_id,
    o.payload, 
    o.storage_meta, 
    o.user_id, 
    o.original_id, 
    tc.total_filtered_objects_count,
    v.distance
ORDER BY result_distance ASC
LIMIT $2 OFFSET $3;');
    RETURN QUERY EXECUTE query
    USING input_vector, limit_results, offset_value, max_distance, enlarged_limit, enlarged_offset, average_only;
END;
$$ LANGUAGE plpgsql;    
"""
    return sql_function
