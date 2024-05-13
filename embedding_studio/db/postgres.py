from sqlalchemy import create_engine

from embedding_studio.core.config import settings

pg_database = create_engine(settings.POSTGRES_DB_URI)
