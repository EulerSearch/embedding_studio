# Documentation for `get_where_combinations_function`

## Functionality

The `get_where_combinations_function` returns SQL that defines two PostgreSQL functions. The first function, `generate_combinations`, generates array element combinations. The second function, `generate_where_combinations`, executes a query with progressively relaxed WHERE clauses. This is useful for search relaxation, where strict conditions might not return many results.

## Parameters

This function does not take any parameters.

## Usage

Use the function to generate SQL that creates the PostgreSQL functions. Run the returned SQL on your database to enable flexible search queries.

### Example

```sql
sql_query = get_where_combinations_function()
-- Execute sql_query in your PostgreSQL database.
```