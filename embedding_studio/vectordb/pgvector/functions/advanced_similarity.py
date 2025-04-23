from typing import Optional

from embedding_studio.models.embeddings.models import MetricType


def generate_advanced_vector_search_function(
    model_id: str, metric_type: Optional[MetricType] = MetricType.COSINE
) -> str:
    """
    Generate a PostgreSQL function for advanced vector search with vectors in the results.

    This function creates SQL for a complex vector search that includes filtering by payload,
    user ID, and allows sorting by payload fields or table columns. The generated function
    returns vectors along with search results.

    The search process involves:
    1. Applying payload and user filters to objects
    2. Finding and scoring vectors within filtered objects
    3. Grouping results by object with distance and vector information

    :param model_id: ID of the embedding model, used for table name generation
    :param metric_type: Vector distance metric type (COSINE, EUCLID, DOT)
    :return: SQL string that creates a PostgreSQL function for advanced vector search
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
    function_name = f"advanced_v_{model_id}_{metric_name}"

    dbo_table = f"dbo_{model_id}"
    dbop_table = f"dbop_{model_id}"

    sql_function = f"""
CREATE OR REPLACE FUNCTION {function_name}(
    input_vector             vector,
    user_id                  VARCHAR(128) DEFAULT NULL,
    payload_filter_sql       TEXT         DEFAULT NULL,
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
    payload_where_clause TEXT;
    order_clause         TEXT := '';
    additional_group_by  TEXT := '';
    additional_column    TEXT := '';
    query                TEXT;
BEGIN
    payload_where_clause := COALESCE(payload_filter_sql, 'TRUE');

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
            additional_group_by := format(', o.%I', sort_field);
            additional_column := format(', o.%I', sort_field);
            IF lower(sort_order) != 'desc' THEN
                order_clause := format('ORDER BY %I ASC', sort_field);
            ELSE
                order_clause := format('ORDER BY %I DESC', sort_field);
            END IF;
        END IF;
    END IF;

    IF user_id IS NOT NULL THEN
        query := format('
WITH customized_original_ids AS (
    SELECT original_id 
    FROM {dbo_table}
    WHERE user_id = $2
),
filtered_objects AS (
    SELECT 
        object_id, 
        payload, 
        storage_meta, 
        user_id,  
        o.original_id%s
    FROM {dbo_table} o
    LEFT JOIN customized_original_ids coi ON o.object_id = coi.original_id
    WHERE (user_id = $2 OR user_id IS NULL) AND (coi.original_id IS NULL) AND (%s) %s
    limit $10
    offset $11
), prefiltered_vectors AS (
    SELECT
        op.object_id,
        op.part_id,
        o.payload, 
        o.storage_meta, 
        o.user_id, 
        o.original_id, 
        (op.vector {distance_operator} $1) AS distance%s
    FROM {dbop_table} op
    INNER JOIN filtered_objects o ON op.object_id = o.object_id 
    WHERE (NOT $12 OR is_average = TRUE) AND ($6 IS NULL OR op.vector {distance_operator} $1 <= $6)
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
    ARRAY_AGG(vector ORDER BY distance)  AS result_vectors,
    MIN(distance) AS result_distance,
    tc.total_filtered_objects_count AS subset_count 
  FROM prefiltered_vectors o
  CROSS JOIN total_count tc
  GROUP BY 
    object_id, 
    payload, 
    storage_meta, 
    user_id, 
    original_id, 
    tc.total_filtered_objects_count%s %s
  LIMIT $4 OFFSET $5;
', additional_column, payload_where_clause, order_clause, additional_column, additional_group_by, order_clause);
    ELSE
        query := format('
WITH filtered_objects AS (
    SELECT 
        object_id, 
        payload, 
        storage_meta, 
        user_id,  
        o.original_id %s
    FROM {dbo_table} o
    WHERE (user_id IS NULL) AND (%s) %s
    limit $10
    offset $11
), prefiltered_vectors AS (
    SELECT
        op.object_id,
        op.part_id,
        o.payload, 
        o.storage_meta, 
        o.user_id, 
        o.original_id, 
        (op.vector {distance_operator} $1) AS distance %s
    FROM {dbop_table} op
    INNER JOIN filtered_objects o ON op.object_id = o.object_id 
    WHERE (NOT $12 OR is_average = TRUE) AND ($6 IS NULL OR op.vector {distance_operator} $1 <= $6)
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
    ARRAY_AGG(vector ORDER BY distance)  AS result_vectors,
    MIN(distance) AS result_distance,
    tc.total_filtered_objects_count AS subset_count 
  FROM prefiltered_vectors o
  CROSS JOIN total_count tc
  GROUP BY 
    object_id, 
    payload, 
    storage_meta, 
    user_id, 
    original_id, 
    tc.total_filtered_objects_count%s %s
  LIMIT $4 OFFSET $5;
', additional_column, payload_where_clause, order_clause, additional_column, additional_group_by, order_clause);
    END IF;

    RETURN QUERY EXECUTE query
    USING input_vector, user_id, payload_filter_sql, limit_results, offset_value, max_distance, sort_field, sort_order, is_payload, enlarged_limit, enlarged_offset, average_only;
END;
$$ LANGUAGE plpgsql;    
"""
    return sql_function


def generate_advanced_vector_search_no_vectors_function(
    model_id: str, metric_type: Optional[MetricType] = MetricType.COSINE
) -> str:
    """
    Generate a PostgreSQL function for advanced vector search without vectors in the results.

    Similar to generate_advanced_vector_search_function but optimized for cases where
    the actual vector values are not needed in the results. This reduces data transfer
    and memory usage while maintaining the same search capabilities.

    The search process involves:
    1. Applying payload and user filters to objects
    2. Finding and scoring vectors within filtered objects
    3. Grouping results by object with distance information (no vectors)

    :param model_id: ID of the embedding model, used for table name generation
    :param metric_type: Vector distance metric type (COSINE, EUCLID, DOT)
    :return: SQL string that creates a PostgreSQL function for advanced vector search
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
    function_name = f"advanced_{model_id}_{metric_name}"

    dbo_table = f"dbo_{model_id}"
    dbop_table = f"dbop_{model_id}"

    sql_function = f"""
CREATE OR REPLACE FUNCTION {function_name}(
    input_vector             vector,
    user_id                  VARCHAR(128) DEFAULT NULL,
    payload_filter_sql       TEXT         DEFAULT NULL,
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
    payload_where_clause TEXT;
    order_clause         TEXT := '';
    additional_group_by  TEXT := '';
    additional_column    TEXT := '';
    query                TEXT;
BEGIN
    payload_where_clause := COALESCE(payload_filter_sql, 'TRUE');

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
            additional_group_by := format(', o.%I', sort_field);
            additional_column := format(', o.%I', sort_field);
            IF lower(sort_order) != 'desc' THEN
                order_clause := format('ORDER BY %I ASC', sort_field);
            ELSE
                order_clause := format('ORDER BY %I DESC', sort_field);
            END IF;
        END IF;
    END IF;

    IF user_id IS NOT NULL THEN
        query := format('
WITH customized_original_ids AS (
    SELECT original_id 
    FROM {dbo_table}
    WHERE user_id = $2
),
filtered_objects AS (
    SELECT 
        object_id, 
        payload, 
        storage_meta, 
        user_id,  
        o.original_id%s
    FROM {dbo_table} o
    LEFT JOIN customized_original_ids coi ON o.object_id = coi.original_id
    WHERE (user_id = $2 OR user_id IS NULL) AND (coi.original_id IS NULL) AND (%s) %s
    limit $10
    offset $11
), prefiltered_vectors AS (
    SELECT
        op.object_id,
        op.part_id,
        o.payload, 
        o.storage_meta, 
        o.user_id, 
        o.original_id, 
        (op.vector {distance_operator} $1) AS distance%s
    FROM {dbop_table} op
    INNER JOIN filtered_objects o ON op.object_id = o.object_id 
    WHERE (NOT $12 OR is_average = TRUE) AND ($6 IS NULL OR op.vector {distance_operator} $1 <= $6)
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
  FROM prefiltered_vectors o
  CROSS JOIN total_count tc
  GROUP BY 
    object_id, 
    payload, 
    storage_meta, 
    user_id, 
    original_id, 
    tc.total_filtered_objects_count%s %s
  LIMIT $4 OFFSET $5;
', additional_column, payload_where_clause, order_clause, additional_column, additional_group_by, order_clause);
    ELSE
        query := format('
WITH filtered_objects AS (
    SELECT 
        object_id, 
        payload, 
        storage_meta, 
        user_id,  
        o.original_id %s
    FROM {dbo_table} o
    WHERE (user_id IS NULL) AND (%s) %s
    limit $10
    offset $11
), prefiltered_vectors AS (
    SELECT
        op.object_id,
        op.part_id,
        o.payload, 
        o.storage_meta, 
        o.user_id, 
        o.original_id, 
        (op.vector {distance_operator} $1) AS distance %s
    FROM {dbop_table} op
    INNER JOIN filtered_objects o ON op.object_id = o.object_id 
    WHERE (NOT $12 OR is_average = TRUE) AND ($6 IS NULL OR op.vector {distance_operator} $1 <= $6)
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
  FROM prefiltered_vectors o
  CROSS JOIN total_count tc
  GROUP BY 
    object_id, 
    payload, 
    storage_meta, 
    user_id, 
    original_id, 
    tc.total_filtered_objects_count%s %s
  LIMIT $4 OFFSET $5;
', additional_column, payload_where_clause, order_clause, additional_column, additional_group_by, order_clause);
    END IF;

    RETURN QUERY EXECUTE query
    USING input_vector, user_id, payload_filter_sql, limit_results, offset_value, max_distance, sort_field, sort_order, is_payload, enlarged_limit, enlarged_offset, average_only;
END;
$$ LANGUAGE plpgsql;    
"""
    return sql_function


def generate_advanced_vector_search_similarity_ordered_function(
    model_id: str, metric_type: Optional[MetricType] = MetricType.COSINE
) -> str:
    """
    Generate a PostgreSQL function for similarity-ordered advanced vector search with vectors.

    This function creates SQL for an advanced vector search that prioritizes similarity
    ordering over other sorting criteria. It's optimized for finding the most similar
    vectors first, then applying payload filters. Returns vectors in the results.

    The search process differs from non-similarity-ordered by:
    1. First finding the most similar vectors
    2. Then applying payload and user filters
    3. Finally grouping by object with full vector information

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
    function_name = f"advanced_v_so_{model_id}_{metric_name}"

    dbo_table = f"dbo_{model_id}"
    dbop_table = f"dbop_{model_id}"

    sql_function = f"""
CREATE OR REPLACE FUNCTION {function_name}(
    input_vector             vector,
    user_id                  VARCHAR(128) DEFAULT NULL,
    payload_filter_sql       TEXT         DEFAULT NULL,
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
    payload_where_clause TEXT;
    query                TEXT;
BEGIN
    payload_where_clause := COALESCE(payload_filter_sql, 'TRUE');

    IF user_id IS NOT NULL THEN
        query := format('
WITH customized_original_ids AS (
    SELECT original_id 
    FROM {dbo_table}
    WHERE user_id = $2
),
filtered_vectors AS (
    SELECT
        op.object_id,
        op.part_id AS part_id,
        op.vector AS vector,
        (op.vector {distance_operator} $1) AS distance
    FROM {dbop_table} op
    WHERE (op.user_id is NULL or op.user_id = $2) AND (NOT $9 OR op.is_average = TRUE)
	ORDER BY distance
	LIMIT $7
	OFFSET $8
),
filtered_objects AS (
    SELECT 
        o.object_id, 
        v.part_id, 
        v.vector,
        v.distance, 
        o.payload, 
        o.storage_meta, 
        o.user_id, 
        o.original_id
    FROM {dbo_table} o
    INNER JOIN filtered_vectors v ON o.object_id = v.object_id
    LEFT JOIN customized_original_ids coi ON coi.original_id = o.object_id
    WHERE (coi.original_id IS NULL) and (%s) 
),
total_count AS (
    SELECT cast(count(distinct(object_id)) as int) AS total_filtered_objects_count
    FROM filtered_objects o WHERE ($6 IS NULL OR o.distance <= $6)
) SELECT 
	object_id AS result_object_id, 	
	payload AS result_payload, 
	storage_meta AS result_storage_meta, 
	user_id AS result_user_id, 
	original_id AS result_original_id, 
	ARRAY_AGG(part_id ORDER BY distance) AS result_part_ids,
    ARRAY_AGG(vector ORDER BY distance) AS result_vectors,
    MAX(distance) AS result_distance,
    tc.total_filtered_objects_count AS subset_count
  FROM filtered_objects 
  CROSS JOIN total_count tc
  GROUP BY 
    object_id, 
    payload, 
    storage_meta,  
    user_id, 
    original_id, 
    tc.total_filtered_objects_count
  ORDER BY result_distance ASC
  LIMIT $4 OFFSET $5;', payload_where_clause);
    ELSE
        query := format('
WITH filtered_vectors AS (
    SELECT
        op.object_id,
        op.part_id AS part_id,
        op.vector as vector,
        (op.vector {distance_operator} $1) AS distance
    FROM {dbop_table} op
    WHERE (op.user_id IS NULL) AND (NOT $9 OR op.is_average = TRUE)
	ORDER BY distance
	LIMIT $7
	OFFSET $8
),
filtered_objects AS (
    SELECT 
        o.object_id, 
        v.part_id, 
        v.vector,
        v.distance, 
        o.payload, 
        o.storage_meta, 
        o.user_id, 
        o.original_id
    FROM {dbo_table} o
    INNER JOIN filtered_vectors v ON o.object_id = v.object_id
    WHERE (o.user_id IS NULL) AND ($6 IS NULL OR distance <= $6) AND (%s)
),
total_count AS (
    SELECT cast(count(distinct(object_id)) as int) AS total_filtered_objects_count
    FROM filtered_objects o WHERE ($6 IS NULL OR o.distance <= $6)
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
  FROM filtered_objects 
  CROSS JOIN total_count tc
  GROUP BY 
    object_id, 
    payload, 
    storage_meta,  
    user_id, 
    original_id, 
    tc.total_filtered_objects_count 
  ORDER BY result_distance ASC
  LIMIT $4 OFFSET $5;', payload_where_clause);
    END IF;

    RETURN QUERY EXECUTE query
    USING input_vector, user_id, payload_filter_sql, limit_results, offset_value, max_distance, enlarged_limit, enlarged_offset, average_only;
END;
$$ LANGUAGE plpgsql;    
"""
    return sql_function


def generate_advanced_vector_search_similarity_ordered_no_vectors_function(
    model_id: str, metric_type: Optional[MetricType] = MetricType.COSINE
) -> str:
    """
    Generate a PostgreSQL function for similarity-ordered advanced search without vectors.

    Similar to generate_advanced_vector_search_similarity_ordered_function but doesn't
    include vector values in the results. This is more efficient for cases where
    only object metadata and distance scores are needed.

    The search process follows the same approach as the with-vectors version:
    1. First finding the most similar vectors
    2. Then applying payload and user filters
    3. Finally grouping by object without including vector data

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
    function_name = f"advanced_so_{model_id}_{metric_name}"

    dbo_table = f"dbo_{model_id}"
    dbop_table = f"dbop_{model_id}"

    sql_function = f"""
CREATE OR REPLACE FUNCTION {function_name}(
    input_vector             vector,
    user_id                  VARCHAR(128) DEFAULT NULL,
    payload_filter_sql       TEXT         DEFAULT NULL,
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
    payload_where_clause TEXT;
    query                TEXT;
BEGIN
    payload_where_clause := COALESCE(payload_filter_sql, 'TRUE');

    IF user_id IS NOT NULL THEN
        query := format('
WITH customized_original_ids AS (
    SELECT original_id 
    FROM {dbo_table}
    WHERE user_id = $2
),
filtered_vectors AS (
    SELECT
        op.object_id,
        op.part_id AS part_id,
        (op.vector {distance_operator} $1) AS distance
    FROM {dbop_table} op
    WHERE (op.user_id is NULL or op.user_id = $2) AND (NOT $9 OR op.is_average = TRUE)
	ORDER BY distance
	LIMIT $7
	OFFSET $8
),
filtered_objects AS (
    SELECT 
        o.object_id, 
        v.part_id, 
        v.distance, 
        o.payload, 
        o.storage_meta, 
        o.user_id, 
        o.original_id
    FROM {dbo_table} o
    INNER JOIN filtered_vectors v ON o.object_id = v.object_id
    LEFT JOIN customized_original_ids coi ON coi.original_id = o.object_id
    WHERE (coi.original_id IS NULL) and (%s)
),
total_count AS (
    SELECT cast(count(distinct(object_id)) as int) AS total_filtered_objects_count
    FROM filtered_objects o WHERE ($6 IS NULL OR o.distance <= $6)
) SELECT 
	object_id AS result_object_id, 	
	payload AS result_payload, 
	storage_meta AS result_storage_meta, 
	user_id AS result_user_id, 
	original_id AS result_original_id, 
	ARRAY_AGG(part_id) AS result_part_ids,
    MAX(distance) AS result_distance,
    tc.total_filtered_objects_count AS subset_count
  FROM filtered_objects 
  CROSS JOIN total_count tc
  GROUP BY 
    object_id, 
    payload, 
    storage_meta,  
    user_id, 
    original_id, 
    tc.total_filtered_objects_count
  ORDER BY result_distance ASC
  LIMIT $4 OFFSET $5;', payload_where_clause);
    ELSE
        query := format('
WITH filtered_vectors AS (
    SELECT
        op.object_id,
        op.part_id AS part_id,
        (op.vector {distance_operator} $1) AS distance
    FROM {dbop_table} op
    WHERE (op.user_id IS NULL) AND (NOT $9 OR op.is_average = TRUE)
	ORDER BY distance
	LIMIT $7
	OFFSET $8
),
filtered_objects AS (
    SELECT 
        o.object_id, 
        v.part_id, 
        v.distance, 
        o.payload, 
        o.storage_meta, 
        o.user_id, 
        o.original_id
    FROM {dbo_table} o
    INNER JOIN filtered_vectors v ON o.object_id = v.object_id
    WHERE (o.user_id IS NULL) AND ($6 IS NULL OR distance <= $6) AND (%s)
),
total_count AS (
    SELECT cast(count(distinct(object_id)) as int) AS total_filtered_objects_count
    FROM filtered_objects o WHERE ($6 IS NULL OR o.distance <= $6)
) SELECT 
	object_id AS result_object_id, 	
	payload AS result_payload, 
	storage_meta AS result_storage_meta, 
	user_id AS result_user_id, 
	original_id AS result_original_id, 
	ARRAY_AGG(part_id) AS result_part_ids,
    MIN(distance) AS result_distance,
    tc.total_filtered_objects_count AS subset_count
  FROM filtered_objects 
  CROSS JOIN total_count tc
  GROUP BY 
    object_id, 
    payload, 
    storage_meta,  
    user_id, 
    original_id, 
    tc.total_filtered_objects_count 
  ORDER BY result_distance ASC
  LIMIT $4 OFFSET $5;', payload_where_clause);
    END IF;

    RETURN QUERY EXECUTE query
    USING input_vector, user_id, payload_filter_sql, limit_results, offset_value, max_distance, enlarged_limit, enlarged_offset, average_only;
END;
$$ LANGUAGE plpgsql;    
"""
    return sql_function


def generate_advanced_vector_search_similarity_ordered_no_vectors_function_(
    model_id: str, metric_type: Optional[MetricType] = MetricType.COSINE
) -> str:
    operator_map = {
        MetricType.COSINE: "<=>",
        MetricType.EUCLID: "<->",
        MetricType.DOT: "<#>",
    }

    if metric_type not in operator_map:
        raise ValueError(f"Unsupported metric type: {metric_type}")

    distance_operator = operator_map[metric_type]
    metric_name = metric_type.value
    function_name = f"advanced_so_{model_id}_{metric_name}"

    dbo_table = f"dbo_{model_id}"
    dbop_table = f"dbop_{model_id}"

    base_query = f"""
CREATE OR REPLACE FUNCTION {function_name}(
    input_vector             vector,
    user_id                  VARCHAR(128) DEFAULT NULL,
    payload_filter_sql       TEXT         DEFAULT NULL,
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
    payload_where_clause TEXT;
    query                TEXT;
BEGIN
    payload_where_clause := COALESCE(payload_filter_sql, 'TRUE');

    IF user_id IS NOT NULL THEN
        query := format($fmt$
WITH customized_original_ids AS (
    SELECT original_id FROM {dbo_table}
    WHERE user_id = $2
),
filtered_vectors AS (
    SELECT
        op.object_id,
        op.part_id,
        (op.vector {distance_operator} $1) AS distance
    FROM {dbop_table} op
    WHERE
        (op.user_id IS NULL OR op.user_id = $2)
        AND (NOT $9 OR op.is_average = TRUE)
        AND ($6 IS NULL OR (op.vector {distance_operator} $1) <= $6)
        AND EXISTS (
            SELECT 1 FROM {dbo_table} o
            WHERE o.object_id = op.object_id
              AND (%s)
              AND NOT EXISTS (
                  SELECT 1 FROM customized_original_ids coi
                  WHERE coi.original_id = o.object_id
              )
        )
    ORDER BY distance
    LIMIT $7 OFFSET $8
),
filtered_objects AS (
    SELECT
        v.object_id,
        o.payload,
        o.storage_meta,
        o.user_id,
        o.original_id,
        v.distance,
        v.part_id
    FROM filtered_vectors v
    JOIN {dbo_table} o ON o.object_id = v.object_id
),
total_count AS (
    SELECT COUNT(DISTINCT object_id)::INT AS total_filtered_objects_count
    FROM filtered_objects
)
SELECT
    o.object_id AS result_object_id,
    o.payload AS result_payload,
    o.storage_meta AS result_storage_meta,
    o.user_id AS result_user_id,
    o.original_id AS result_original_id,
    ARRAY_AGG(o.part_id) AS result_part_ids,
    MIN(o.distance) AS result_distance,
    tc.total_filtered_objects_count AS subset_count
FROM filtered_objects o
CROSS JOIN total_count tc
GROUP BY
    o.object_id, o.payload, o.storage_meta,
    o.user_id, o.original_id, tc.total_filtered_objects_count
ORDER BY result_distance ASC
LIMIT $4 OFFSET $5;
$fmt$, payload_where_clause);
    ELSE
        query := format($fmt$
WITH filtered_vectors AS (
    SELECT
        op.object_id,
        op.part_id,
        (op.vector {distance_operator} $1) AS distance
    FROM {dbop_table} op
    WHERE
        op.user_id IS NULL
        AND (NOT $9 OR op.is_average = TRUE)
        AND ($6 IS NULL OR (op.vector {distance_operator} $1) <= $6)
        AND EXISTS (
            SELECT 1 FROM {dbo_table} o
            WHERE o.object_id = op.object_id AND (%s)
        )
    ORDER BY distance
    LIMIT $7 OFFSET $8
),
filtered_objects AS (
    SELECT
        v.object_id,
        o.payload,
        o.storage_meta,
        o.user_id,
        o.original_id,
        v.distance,
        v.part_id
    FROM filtered_vectors v
    JOIN {dbo_table} o ON o.object_id = v.object_id
),
total_count AS (
    SELECT COUNT(DISTINCT object_id)::INT AS total_filtered_objects_count
    FROM filtered_objects
)
SELECT
    o.object_id AS result_object_id,
    o.payload AS result_payload,
    o.storage_meta AS result_storage_meta,
    o.user_id AS result_user_id,
    o.original_id AS result_original_id,
    ARRAY_AGG(o.part_id) AS result_part_ids,
    MIN(o.distance) AS result_distance,
    tc.total_filtered_objects_count AS subset_count
FROM filtered_objects o
CROSS JOIN total_count tc
GROUP BY
    o.object_id, o.payload, o.storage_meta,
    o.user_id, o.original_id, tc.total_filtered_objects_count
ORDER BY result_distance ASC
LIMIT $4 OFFSET $5;
$fmt$, payload_where_clause);
    END IF;

    RETURN QUERY EXECUTE query
    USING input_vector, user_id, payload_filter_sql,
          limit_results, offset_value, max_distance,
          enlarged_limit, enlarged_offset, average_only;
END;
$$ LANGUAGE plpgsql;
"""
    return base_query
