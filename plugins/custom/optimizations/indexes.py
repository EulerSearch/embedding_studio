from sqlalchemy import text

from embedding_studio.vectordb.pgvector.optimization import (
    PgvectorObjectsOptimization,
)


class CreateOrderingIndexesOptimization(PgvectorObjectsOptimization):
    def __init__(self):
        super().__init__(name="CreateOrderingIndexesOptimization")

    def _get_statement(self, tablename: str):
        likes_index_name = f"{tablename}_l_idx"
        downloads_index_name = f"{tablename}_d_idx"
        popularity_index_name = f"{tablename}_p_idx"
        source_name_index_name = f"{tablename}_sn_idx"
        name_index_name = f"{tablename}_n_idx"
        created_at_index_name = f"{tablename}_ca_idx"
        modified_at_index_name = f"{tablename}_ca_idx"

        return text(
            f"""ALTER TABLE {tablename}
ADD COLUMN IF NOT EXISTS likes INTEGER,
ADD COLUMN IF NOT EXISTS downloads INTEGER,
ADD COLUMN IF NOT EXISTS popularity_pace FLOAT,
ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ,
ADD COLUMN IF NOT EXISTS modified_at TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS idx_{tablename}_likes ON {tablename}(likes);
CREATE INDEX IF NOT EXISTS idx_{tablename}_downloads ON {tablename}(downloads);
CREATE INDEX IF NOT EXISTS idx_{tablename}_popularity_pace ON {tablename}(popularity_pace);
CREATE INDEX IF NOT EXISTS idx_{tablename}_created_at ON {tablename}(created_at);
CREATE INDEX IF NOT EXISTS idx_{tablename}_modified_at ON {tablename}(modified_at);

CREATE OR REPLACE FUNCTION update_columns_{tablename}_from_payload()
RETURNS TRIGGER AS $$
BEGIN
  -- If the JSON payload contains the key 'likes', update the likes column.
  IF NEW.payload ? 'likes' THEN
    NEW.likes := (NEW.payload->>'likes')::INTEGER;
  END IF;
  -- Similarly for downloads.
  IF NEW.payload ? 'downloads' THEN
    NEW.downloads := (NEW.payload->>'downloads')::INTEGER;
  END IF;
  -- For popularity_pace.
  IF NEW.payload ? 'popularity_pace' THEN
    NEW.popularity_pace := (NEW.payload->>'popularity_pace')::FLOAT;
  END IF;
  -- For created_at from hf_created_at.
  IF NEW.payload ? 'hf_created_at' THEN
    NEW.created_at := (NEW.payload->>'hf_created_at')::TIMESTAMPTZ;
  END IF;
  -- For modified_at from hf_lastModified.
  IF NEW.payload ? 'hf_lastModified' THEN
    NEW.modified_at := (NEW.payload->>'hf_lastModified')::TIMESTAMPTZ;
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;


CREATE OR REPLACE TRIGGER trg_update_columns_{tablename}_from_payload
BEFORE INSERT OR UPDATE ON {tablename}
FOR EACH ROW
EXECUTE FUNCTION update_columns_{tablename}_from_payload();

CREATE OR REPLACE PROCEDURE batch_update_payload_{tablename}(
    IN batch_size integer DEFAULT 100000
)
LANGUAGE plpgsql
AS $$
DECLARE
    rows_affected  integer;
    total_affected bigint := 0;
BEGIN
    LOOP
        -- Start a chunk-level transaction each time

        WITH next_batch AS (
		    SELECT object_id
		    FROM {tablename}
		    WHERE likes IS NULL
		        OR downloads IS NULL
		        OR popularity_pace IS NULL
		        OR created_at IS NULL
		        OR modified_at IS NULL
		    ORDER BY modified_at NULLS FIRST
		    LIMIT batch_size
		    FOR UPDATE SKIP LOCKED
		)
		-- Use a JOIN instead of IN for better performance
		UPDATE {tablename} t
		SET 
		    likes = COALESCE(t.likes, (t.payload->>'likes')::INTEGER),
		    downloads = COALESCE(t.downloads, (t.payload->>'downloads')::INTEGER),
		    popularity_pace = COALESCE(t.popularity_pace, (t.payload->>'popularity_pace')::FLOAT),
		    created_at = COALESCE(t.created_at, (t.payload->>'hf_created_at')::TIMESTAMPTZ),
		    modified_at = COALESCE(t.modified_at, (t.payload->>'hf_lastModified')::TIMESTAMPTZ)
		FROM next_batch nb
		WHERE t.object_id = nb.object_id;

        -- How many rows got updated in this chunk?
        GET DIAGNOSTICS rows_affected = ROW_COUNT;
        total_affected := total_affected + rows_affected;

        RAISE NOTICE 'Updated % rows in this chunk. Running total: %',
            rows_affected,
            total_affected;

        -- If we updated fewer than batch_size rows, we’ve reached the end
        IF rows_affected < batch_size THEN
            EXIT;
        END IF;
    END LOOP;

    RAISE NOTICE 'Done updating. Total updated rows: %', total_affected;
END;
$$;


CALL batch_update_payload_{tablename}(100000);
        
-- Create function for integer extraction that is IMMUTABLE
CREATE OR REPLACE FUNCTION jsonb_extract_integer(jsonb, text) RETURNS integer
    LANGUAGE sql IMMUTABLE
    AS $_$SELECT ($1->>$2)::integer$_$;

-- Create function for float extraction that is IMMUTABLE
CREATE OR REPLACE FUNCTION jsonb_extract_float(jsonb, text) RETURNS float
    LANGUAGE sql IMMUTABLE
    AS $_$SELECT ($1->>$2)::float$_$;

-- Create function for timestamptz extraction that is IMMUTABLE
CREATE OR REPLACE FUNCTION jsonb_extract_timestamptz(jsonb, text) RETURNS timestamptz
    LANGUAGE sql IMMUTABLE
    AS $_$SELECT ($1->>$2)::timestamptz$_$;

-- Integer indexes using IMMUTABLE functions
CREATE INDEX IF NOT EXISTS {likes_index_name} ON {tablename} (jsonb_extract_integer(payload, 'likes'));
CREATE INDEX IF NOT EXISTS {downloads_index_name} ON {tablename} (jsonb_extract_integer(payload, 'downloads'));

-- Float index using IMMUTABLE function
CREATE INDEX IF NOT EXISTS {popularity_index_name} ON {tablename} (jsonb_extract_float(payload, 'popularity_pace'));

-- Date indexes using IMMUTABLE function
CREATE INDEX IF NOT EXISTS {created_at_index_name} ON {tablename} (jsonb_extract_timestamptz(payload, 'hf_created_at'));
CREATE INDEX IF NOT EXISTS {modified_at_index_name} ON {tablename} (jsonb_extract_timestamptz(payload, 'hf_lastModified'));

-- Text indexes (for reference) - these should work without changes
CREATE INDEX IF NOT EXISTS {source_name_index_name} ON {tablename} ((payload->>'source_name'));
CREATE INDEX IF NOT EXISTS {name_index_name} ON {tablename} ((payload->>'name'));
"""
        )
