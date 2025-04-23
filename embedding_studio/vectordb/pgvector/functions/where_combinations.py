def get_where_combinations_function() -> str:
    """
    Generate SQL functions for creating and executing WHERE clause combinations.

    This function returns SQL that creates two PostgreSQL functions:

    1. generate_combinations: A recursive function that generates all possible
       combinations of elements from an array, of a specified size.

    2. generate_where_combinations: A function that executes a query with
       progressively relaxed WHERE clauses by trying different combinations of
       conditions.

    The primary use case is for search relaxation, where a query with strict
    conditions might return few or no results, but relaxing some conditions
    can provide partial matches.

    The search process:
    1. First tries all conditions together (full AND)
    2. Then tries all combinations of N-1 conditions
    3. Continues with smaller combinations until reaching size 2
    4. For each layer, respects limit and offset parameters

    This approach prioritizes results that match more conditions while still
    returning partial matches when necessary.

    :return: SQL string that creates PostgreSQL functions for WHERE clause combinations
    """
    return """CREATE OR REPLACE FUNCTION generate_combinations(arr text[], r int)
RETURNS SETOF text[] AS $$
DECLARE
    result text[];
BEGIN
    IF r = 0 THEN
        RETURN NEXT ARRAY[]::text[];
        RETURN;
    ELSIF array_length(arr,1) < r THEN
        RETURN;
    ELSIF array_length(arr,1) = r THEN
        RETURN NEXT arr;
        RETURN;
    ELSE
        -- Include the first element and combine with combinations of size r-1 from the rest
        FOR result IN
            SELECT ARRAY[arr[1]] || comb
            FROM generate_combinations(arr[2:array_length(arr,1)], r-1) AS comb
        LOOP
            RETURN NEXT result;
        END LOOP;
        -- Exclude the first element and generate combinations of size r from the rest
        FOR result IN
            SELECT comb
            FROM generate_combinations(arr[2:array_length(arr,1)], r) AS comb
        LOOP
            RETURN NEXT result;
        END LOOP;
    END IF;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION generate_where_combinations(
    base_query TEXT,
    conditions TEXT[],
    row_limit INT DEFAULT 100,
    row_offset INT DEFAULT 0,
    order_by_clause TEXT DEFAULT NULL
)
RETURNS SETOF RECORD AS $$
DECLARE
    num_conditions INT := array_length(conditions, 1);
    k INT;
    comb_arr text[];
    clause text;
    clause_array text[] := ARRAY[]::text[];
    combined_clause text;
    full_query text;
    order_clause text := '';
    remaining_rows int := row_limit;
    current_offset int := row_offset;
    window_size int := 1000; -- Process in chunks for better memory management
    record_var record;
    query_cursor REFCURSOR;
    exists_result boolean;
BEGIN
    -- Set up the ORDER BY clause if provided
    IF order_by_clause IS NOT NULL AND order_by_clause != '' THEN
        order_clause := ' ORDER BY ' || order_by_clause;
    END IF;

    -- Step 1: Full AND combination (all conditions, N)
    IF remaining_rows > 0 THEN
        full_query := base_query || ' WHERE ' || array_to_string(conditions, ' AND ');
        
        -- Check if the query would return any results
        EXECUTE 'SELECT EXISTS(SELECT 1 FROM (' || full_query || ') t)' INTO exists_result;
        
        IF exists_result THEN
            -- Add ORDER BY clause here after checking existence
            OPEN query_cursor FOR EXECUTE full_query || 
                order_clause ||
                ' OFFSET ' || current_offset || 
                ' LIMIT ' || LEAST(window_size, remaining_rows);
                
            LOOP
                FETCH query_cursor INTO record_var;
                EXIT WHEN NOT FOUND OR remaining_rows <= 0;
                remaining_rows := remaining_rows - 1;
                RETURN NEXT record_var;
            END LOOP;
            CLOSE query_cursor;
            
            IF remaining_rows <= 0 THEN
                RETURN;
            END IF;
            current_offset := 0; -- Reset offset for the next layer
        END IF;
    END IF;

    -- Step 2: For each combination size from N-1 down to 2
    FOR k IN REVERSE num_conditions-1 .. 2 LOOP
        clause_array := ARRAY[]::text[]; -- Reset clause array for this layer
        
        FOR comb_arr IN SELECT * FROM generate_combinations(conditions, k) LOOP
            clause := '(' || array_to_string(comb_arr, ' AND ') || ')';
            clause_array := clause_array || clause;
        END LOOP;
        
        IF array_length(clause_array, 1) IS NOT NULL THEN
            combined_clause := array_to_string(clause_array, ' OR ');
            full_query := base_query || ' WHERE ' || combined_clause;
            
            -- Check if this combination would return any results
            EXECUTE 'SELECT EXISTS(SELECT 1 FROM (' || full_query || ') t)' INTO exists_result;
            
            IF exists_result THEN
                -- Add ORDER BY clause here after checking existence
                OPEN query_cursor FOR EXECUTE full_query || 
                    order_clause ||
                    ' OFFSET ' || current_offset || 
                    ' LIMIT ' || LEAST(window_size, remaining_rows);
                    
                LOOP
                    FETCH query_cursor INTO record_var;
                    EXIT WHEN NOT FOUND OR remaining_rows <= 0;
                    remaining_rows := remaining_rows - 1;
                    RETURN NEXT record_var;
                END LOOP;
                CLOSE query_cursor;
                
                IF remaining_rows <= 0 THEN
                    RETURN;
                END IF;
                current_offset := 0; -- Reset offset for next layer
            END IF;
        END IF;
    END LOOP;

    RETURN;
END;
$$ LANGUAGE plpgsql;
"""
